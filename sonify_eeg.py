"""Live EEG sonification: hear your brainwaves through the ML201 speaker.

Connects to Muse 2, reads EEG in real-time, generates audio from
brain activity, and streams it to the Bluetooth speaker.

Audio mapping:
  - Alpha power (8-13 Hz)  -> base drone pitch (relaxed = lower, warm tone)
  - Beta power (13-30 Hz)  -> brightness / higher harmonics (focused = brighter)
  - Theta power (4-8 Hz)   -> slow amplitude modulation (dreamy pulsing)
  - Blinks (AF7/AF8 spike) -> percussive ping
  - Overall amplitude      -> volume

Usage:
    source env/bin/activate
    python sonify_eeg.py
"""

import asyncio
import time
from collections import defaultdict

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfilt_zi
from bleak import BleakClient, BleakScanner

# --- Config ---
MUSE_NAME = "Muse-31A9"
SAMPLE_RATE = 256       # Muse EEG sample rate
AUDIO_SR = 44100        # Audio sample rate
CHUNK_SECONDS = 0.5     # Generate audio every 0.5s
DURATION = 120          # Total session duration in seconds

EEG_UUIDS = {
    "TP9":  "273e0003-4c4d-454d-96be-f03bac821358",
    "AF7":  "273e0004-4c4d-454d-96be-f03bac821358",
    "AF8":  "273e0005-4c4d-454d-96be-f03bac821358",
    "TP10": "273e0006-4c4d-454d-96be-f03bac821358",
}
CONTROL_UUID = "273e0001-4c4d-454d-96be-f03bac821358"
CMD_RESUME = bytearray([0x02, 0x64, 0x0a])
CMD_HALT = bytearray([0x02, 0x68, 0x0a])

# --- EEG Ring Buffer ---
BUF_SECONDS = 5
BUF_LEN = SAMPLE_RATE * BUF_SECONDS

eeg_buffers = {ch: np.zeros(BUF_LEN) for ch in EEG_UUIDS}
eeg_write_pos = {ch: 0 for ch in EEG_UUIDS}
eeg_total = {ch: 0 for ch in EEG_UUIDS}


def decode_eeg(packet: bytearray) -> list[float]:
    bit_buffer = 0
    bit_count = 0
    samples = []
    for byte in packet[2:]:
        bit_buffer = (bit_buffer << 8) | byte
        bit_count += 8
        while bit_count >= 12:
            bit_count -= 12
            raw = (bit_buffer >> bit_count) & 0xFFF
            samples.append(raw * 0.48828125)
    return samples


def push_samples(channel: str, samples: list[float]):
    buf = eeg_buffers[channel]
    pos = eeg_write_pos[channel]
    for s in samples:
        buf[pos % BUF_LEN] = s
        pos += 1
    eeg_write_pos[channel] = pos
    eeg_total[channel] += len(samples)


def get_window(channel: str, seconds: float) -> np.ndarray:
    n = int(SAMPLE_RATE * seconds)
    total = eeg_total[channel]
    if total < n:
        n = total
    if n == 0:
        return np.zeros(1)
    pos = eeg_write_pos[channel]
    buf = eeg_buffers[channel]
    indices = np.arange(pos - n, pos) % BUF_LEN
    return buf[indices]


def make_callback(channel: str):
    def callback(sender, data):
        samples = decode_eeg(data)
        push_samples(channel, samples)
    return callback


# --- Streaming Bandpass Filter ---
class StreamingBandpass:
    def __init__(self, low, high, sr=SAMPLE_RATE, order=4):
        nyq = sr / 2
        self.sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
        self.zi = sosfilt_zi(self.sos)
        # Stack zi for proper initial conditions
        self.zi = np.zeros_like(self.zi)

    def process(self, data: np.ndarray) -> np.ndarray:
        if len(data) < 2:
            return data
        out, self.zi = sosfilt(self.sos, data, zi=self.zi)
        return out


# --- Band Power ---
def band_power(data: np.ndarray, low: float, high: float) -> float:
    if len(data) < SAMPLE_RATE:
        return 0.0
    nyq = SAMPLE_RATE / 2
    sos = butter(4, [low / nyq, high / nyq], btype="band", output="sos")
    filtered = sosfilt(sos, data)
    return float(np.sqrt(np.mean(filtered ** 2)))


# --- Blink Detection ---
last_blink_time = 0.0
BLINK_THRESHOLD = 100.0  # uV peak-to-peak
BLINK_DEBOUNCE = 0.4     # seconds


def detect_blink() -> float:
    """Returns blink intensity (0-1) or 0 if no blink."""
    global last_blink_time
    now = time.monotonic()
    if now - last_blink_time < BLINK_DEBOUNCE:
        return 0.0

    for ch in ("AF7", "AF8"):
        window = get_window(ch, 0.15)
        if len(window) < 10:
            continue
        centered = window - np.mean(window)
        ptp = np.ptp(centered)
        if ptp > BLINK_THRESHOLD:
            last_blink_time = now
            intensity = min(1.0, ptp / (BLINK_THRESHOLD * 3))
            return intensity
    return 0.0


# --- Audio Synthesis ---
phase = 0.0  # continuous phase accumulator


def generate_audio(alpha_pow, beta_pow, theta_pow, gamma_pow, blink) -> np.ndarray:
    """Generate a chunk of audio from EEG parameters."""
    global phase

    num_frames = int(AUDIO_SR * CHUNK_SECONDS)
    t = np.arange(num_frames) / AUDIO_SR

    # Map alpha to base frequency: more alpha = lower pitch (relaxing)
    # Range: 80 Hz (very relaxed) to 300 Hz (alert)
    alpha_norm = min(1.0, alpha_pow / 15.0)
    beta_norm = min(1.0, beta_pow / 15.0)
    theta_norm = min(1.0, theta_pow / 10.0)

    base_freq = 300 - alpha_norm * 200 + beta_norm * 100

    # Generate base tone with phase accumulation (glitch-free)
    phase_inc = 2 * np.pi * base_freq / AUDIO_SR
    phases = phase + np.cumsum(np.full(num_frames, phase_inc))
    phase = phases[-1] % (2 * np.pi)

    # Sine + harmonics controlled by beta (brightness)
    audio = np.sin(phases)
    audio += 0.3 * beta_norm * np.sin(2 * phases)    # 2nd harmonic
    audio += 0.15 * beta_norm * np.sin(3 * phases)   # 3rd harmonic
    audio += 0.1 * gamma_pow / 20.0 * np.sin(5 * phases)  # 5th harmonic (shimmer)

    # Theta modulation: slow amplitude pulsing
    theta_lfo = 1.0 - 0.3 * theta_norm * np.sin(2 * np.pi * 5 * t)
    audio *= theta_lfo

    # Overall amplitude from signal strength
    total_power = alpha_pow + beta_pow + theta_pow
    volume = np.clip(total_power / 30.0, 0.02, 0.25)
    audio *= volume

    # Blink: percussive ping
    if blink > 0.1:
        ping_freq = 800 + blink * 400
        ping_env = np.exp(-t * 15)  # fast decay
        ping = 0.6 * blink * ping_env * np.sin(2 * np.pi * ping_freq * t)
        audio += ping

    # Normalize to prevent clipping
    peak = np.max(np.abs(audio))
    if peak > 0.95:
        audio *= 0.95 / peak

    return audio.astype(np.float32)


# --- Audio Output (sounddevice via BlueALSA) ---
_audio_buf = []
_stream = None


def start_audio_stream():
    """Open a sounddevice OutputStream routed through BlueALSA."""
    global _stream, _audio_buf
    _audio_buf = []

    def callback(outdata, frames, time_info, status):
        if _audio_buf:
            block = _audio_buf.pop(0)
            n = min(len(block), frames)
            outdata[:n, 0] = block[:n]
            if n < frames:
                outdata[n:, 0] = 0.0
        else:
            outdata[:, 0] = 0.0

    _stream = sd.OutputStream(
        samplerate=AUDIO_SR,
        blocksize=int(AUDIO_SR * CHUNK_SECONDS),
        channels=1,
        dtype="float32",
        callback=callback,
    )
    _stream.start()


def play_audio_chunk(audio: np.ndarray):
    """Queue audio for playback through the ML201."""
    _audio_buf.append(audio.astype(np.float32))


def stop_audio_stream():
    """Stop and close the audio stream."""
    global _stream
    if _stream is not None:
        _stream.stop()
        _stream.close()
        _stream = None


# --- Main ---
async def main():
    print(f"Scanning for {MUSE_NAME}...")
    device = await BleakScanner.find_device_by_name(MUSE_NAME, timeout=10)
    if not device:
        print("Muse not found! Make sure it's in pairing mode.")
        return

    print(f"Found: {device.name} ({device.address})")
    import subprocess
    subprocess.run(["bluetoothctl", "trust", device.address],
                   capture_output=True, timeout=5)

    for attempt in range(3):
        try:
            print(f"Connecting (attempt {attempt + 1}/3)...")
            async with BleakClient(device, timeout=30) as client:
                print(f"Connected: {client.is_connected}")

                for ch, uuid in EEG_UUIDS.items():
                    await client.start_notify(uuid, make_callback(ch))
                await client.write_gatt_char(CONTROL_UUID, CMD_RESUME)

                print(f"\nSonifying EEG for {DURATION}s...")
                print("  Close your eyes for a warmer, lower tone")
                print("  Focus hard for a brighter, higher tone")
                print("  Blink for a percussive ping")
                print("  Press Ctrl+C to stop\n")

                # Start audio stream
                start_audio_stream()

                # Wait for buffer to fill a bit
                await asyncio.sleep(1.5)

                start_time = time.monotonic()
                chunk_num = 0

                try:
                    while time.monotonic() - start_time < DURATION:
                        # Compute band powers from 1s windows
                        af7 = get_window("AF7", 1.0)
                        af8 = get_window("AF8", 1.0)
                        avg = (af7[:min(len(af7), len(af8))] +
                               af8[:min(len(af7), len(af8))]) / 2

                        alpha = band_power(avg, 8, 13)
                        beta = band_power(avg, 13, 30)
                        theta = band_power(avg, 4, 8)
                        gamma = band_power(avg, 30, 50)
                        blink = detect_blink()

                        # Generate and play audio
                        audio = generate_audio(alpha, beta, theta, gamma, blink)
                        play_audio_chunk(audio)

                        chunk_num += 1
                        elapsed = time.monotonic() - start_time
                        if chunk_num % 4 == 0:
                            print(
                                f"  [{elapsed:5.1f}s] "
                                f"α={alpha:5.1f} β={beta:5.1f} "
                                f"θ={theta:5.1f} γ={gamma:5.1f}"
                                f"{'  *BLINK*' if blink > 0.1 else ''}"
                            )

                        await asyncio.sleep(CHUNK_SECONDS)

                except KeyboardInterrupt:
                    print("\nStopping...")

                stop_audio_stream()

                try:
                    await client.write_gatt_char(CONTROL_UUID, CMD_HALT)
                    for uuid in EEG_UUIDS.values():
                        await client.stop_notify(uuid)
                except Exception:
                    pass

            print("Done!")
            return

        except Exception as e:
            total = sum(eeg_total.values())
            if total > SAMPLE_RATE:
                print(f"  Disconnected after collecting {total} samples.")
                print("Done!")
                return
            print(f"  Failed: {e}")
            if attempt < 2:
                print("  Retrying in 2s...")
                await asyncio.sleep(2)

    print("All connection attempts failed.")


if __name__ == "__main__":
    asyncio.run(main())
