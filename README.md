# TheBox - Muse 2 EEG on Raspberry Pi

Connect a Muse 2 EEG headband to a Raspberry Pi and stream raw brainwave data over BLE.

## Hardware

- Raspberry Pi (tested on Pi 5, Debian Trixie, kernel 6.12, arm64)
- Muse 2 headband (model MU-02)
- Built-in Bluetooth (no dongle needed)

## Prerequisites

### System packages

```bash
sudo apt-get install -y python3 python3-venv cmake g++ git
```

### Python environment

```bash
cd /home/gian/projects/thebox
python3 -m venv env
source env/bin/activate
pip install muselsl bleak
```

### Build liblsl from source (required on arm64)

The `pylsl` package (dependency of `muselsl`) needs the native `liblsl` shared library. There are no pre-built arm64 wheels, so you must build it:

```bash
cd /tmp
git clone --depth 1 https://github.com/sccn/liblsl.git liblsl-build
cd liblsl-build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Copy the built library into pylsl's lib directory:

```bash
cp /tmp/liblsl-build/build/liblsl.so.* \
   ~/projects/thebox/env/lib/python3.13/site-packages/pylsl/lib/liblsl.so
```

Verify it works:

```bash
source ~/projects/thebox/env/bin/activate
python -c "from pylsl import StreamInfo; print('pylsl OK')"
```

## Connecting the Muse 2

### 1. Put the Muse in pairing mode

- Turn the Muse off completely (hold power button until LED goes off)
- **Disconnect / forget** the Muse from any phone or other device — it won't advertise if already connected elsewhere
- Hold the power button for ~5 seconds until the LED starts blinking rapidly and you hear ascending tones

### 2. Scan for the device

```bash
source ~/projects/thebox/env/bin/activate
python -c "
import asyncio
from bleak import BleakScanner

async def scan():
    devices = await BleakScanner.discover(timeout=10)
    for d in devices:
        if d.name and 'Muse' in d.name:
            print(f'{d.address}  {d.name}')

asyncio.run(scan())
"
```

You should see something like `00:55:DA:B8:31:A9  Muse-31A9`. Note the name — you'll use it in the test script.

### 3. If the address changes (new location / re-pairing)

The Muse 2 uses a **fixed public BLE address**, so it stays the same. However, if you need to update the name in the script:

1. Edit `MUSE_NAME` in `test_muse.py` to match the output from the scan above
2. That's it — the script scans by name, not hardcoded address

### 4. Run the test

```bash
python test_muse.py
```

Expected output:

```
Scanning for Muse-31A9...
Found: Muse-31A9 (00:55:DA:B8:31:A9)
Trusted device in BlueZ
Connecting (attempt 1/3)...
Connected: True
  Subscribed to TP9
  Subscribed to AF7
  Subscribed to AF8
  Subscribed to TP10
  Subscribed to AUX

Streaming EEG for 5 seconds...

  TP9: avg= 946.61 µV  (12 samples)
  AF7: avg= 952.03 µV  (12 samples)
  AF8: avg=1004.07 µV  (12 samples)
  TP10: avg=1246.22 µV  (12 samples)
  ...

Done! Received ~1848 total samples across all channels.
```

### 5. Record EEG with filtered plots

```bash
python record_eeg.py
```

This records EEG data (default 20s, configurable via `DURATION` on line 15) and generates two plots:

- `eeg_filtered.png` — all 4 channels, raw (gray) vs bandpass filtered 1-50 Hz (color)
- `eeg_bands.png` — AF7 channel broken down into delta/theta/alpha/beta/gamma frequency bands

## EEG Frequency Bands

| Band | Frequency | What it reflects |
|------|-----------|-----------------|
| Delta | 0.5-4 Hz | Deep sleep, unconscious processing |
| Theta | 4-8 Hz | Drowsiness, daydreaming, light meditation |
| Alpha | 8-13 Hz | Calm relaxation, eyes closed. Try closing your eyes — you'll see this band increase |
| Beta | 13-30 Hz | Active thinking, focus, concentration |
| Gamma | 30-50 Hz | High-level cognition, sensory processing (also picks up muscle noise) |

## Troubleshooting

### "Muse not found" during scan

- Make sure the Muse is in pairing mode (LED blinking rapidly)
- Forget / disconnect the Muse from any phones or tablets
- Try restarting Bluetooth: `sudo systemctl restart bluetooth`

### Connection times out or "failed to discover services"

This is a common BlueZ race condition. The script has retry logic (3 attempts). If it keeps failing:

```bash
# Remove stale device state from BlueZ
bluetoothctl remove <MUSE_ADDRESS>
# Then run the script again
```

### "Disconnected, but saved X samples"

This is normal and not a data loss issue. BlueZ (the Linux Bluetooth stack) has a known issue with BLE sessions — it sometimes drops the low-level connection a fraction of a second before the application tries to cleanly disconnect. The recording data is fully intact; only the cleanup step failed. No action needed.

### Muse rapidly connecting/disconnecting in a loop

This happens when multiple things are talking to Bluetooth at the same time. To fix:

1. **Close `bluetoothctl`** if it's open in any terminal
2. **Stop any other BLE apps** (phone apps, other scripts)
3. Remove stale state and try again:

```bash
bluetoothctl remove <MUSE_ADDRESS>
# Turn Muse off, wait 3s, then back on in pairing mode
python record_eeg.py
```

### Before every recording session

1. Close `bluetoothctl` in all terminals — only one thing should talk to Bluetooth at a time
2. Don't run other BLE apps simultaneously
3. Fresh pairing mode — if the Muse has been idle for a while it goes to sleep. Turn it off, then back on right before running
4. Keep still during recording — movement can cause electrode contact loss

### "LSL binary library file was not found"

You need to build liblsl from source — see the [build liblsl](#build-liblsl-from-source-required-on-arm64) section above.

### muselsl list fails with bluetoothctl errors

The `muselsl list` command uses a buggy `bluetoothctl` + `pexpect` backend. Use the bleak scan snippet from step 2 instead — it's more reliable.

## Bluetooth Audio Output (ML201 Speaker)

The Pi outputs audio to a Bluetooth speaker (ML201) for EEG sonification via BlueALSA.

### Setup

Install the Bluetooth ALSA backend:

```bash
sudo apt-get install -y bluez-alsa-utils libasound2-plugin-bluez portaudio19-dev
```

### Pairing the ML201

```bash
# Put the ML201 in pairing mode, then:
bluetoothctl pair C4:A9:B8:27:A4:5A
bluetoothctl trust C4:A9:B8:27:A4:5A
bluetoothctl connect C4:A9:B8:27:A4:5A
```

### How audio routing works

On Raspberry Pi OS, the `bluealsa` system service starts at boot and claims the Bluetooth A2DP (high-quality audio) profile. PipeWire cannot compete for this endpoint, so we route audio through BlueALSA instead.

The chain is: `sounddevice → PortAudio → ALSA → BlueALSA → BlueZ A2DP → ML201`

Two config files make this work:

1. **`~/.config/wireplumber/wireplumber.conf.d/disable-seat-monitoring.conf`** — disables PipeWire's Bluetooth monitor so it doesn't fight with BlueALSA:

```
wireplumber.profiles = {
  main = {
    monitor.bluez = disabled
    monitor.bluez-midi = disabled
    support.logind = disabled
  }
}
```

2. **`~/.asoundrc`** — routes the default ALSA device to BlueALSA A2DP:

```
pcm.!default {
    type plug
    slave.pcm {
        type bluealsa
        device "C4:A9:B8:27:A4:5A"
        profile "a2dp"
    }
}

ctl.!default {
    type bluealsa
}
```

After changing these, restart WirePlumber: `systemctl --user restart wireplumber`

### Test tone

```bash
# Quick test with aplay:
aplay -D bluealsa:DEV=C4:A9:B8:27:A4:5A /tmp/test_tone.wav

# Test the Python audio path:
source env/bin/activate
python scripts/test_audio.py
```

### Troubleshooting audio

**No sound from ML201:**
1. Check it's connected: `bluetoothctl info C4:A9:B8:27:A4:5A | grep Connected`
2. Check BlueALSA sees it: `bluealsa-aplay -L` — should list the ML201
3. If not listed, disconnect and reconnect: `bluetoothctl disconnect C4:A9:B8:27:A4:5A && bluetoothctl connect C4:A9:B8:27:A4:5A`
4. Verify `bluealsa` service is running: `systemctl status bluealsa`

**PipeWire grabbing Bluetooth instead of BlueALSA:**
Make sure WirePlumber's Bluetooth monitor is disabled (see config above) and restart: `systemctl --user restart wireplumber`

**Hearing "press Control+Alt..." on connect:**
This is the Raspberry Pi desktop accessibility prompt playing through the speaker when Bluetooth connects. It's harmless — it stops after the announcement.

## EEG Channels

| Channel | Location | Description |
|---------|----------|-------------|
| TP9 | Left ear | Behind left ear |
| AF7 | Left forehead | Above left eyebrow |
| AF8 | Right forehead | Above right eyebrow |
| TP10 | Right ear | Behind right ear |
| AUX | — | Auxiliary (reference) |

## Key Commands Reference

| Command | What it does |
|---------|-------------|
| `source env/bin/activate` | Activate the Python virtual environment |
| `python test_muse.py` | Connect and stream 5s of EEG data |
| `bluetoothctl remove <ADDR>` | Clear stale BLE device from cache |
| `sudo systemctl restart bluetooth` | Restart the Bluetooth service |
| `bluetoothctl show` | Show Bluetooth adapter status |
