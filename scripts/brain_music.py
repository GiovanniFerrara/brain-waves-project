#!/usr/bin/env python3
"""Live music from your brain waves.

    python scripts/brain_music.py                        # live from the Muse
    python scripts/brain_music.py --seconds 300          # stop by itself after 5 min
    python scripts/brain_music.py --guided               # spoken instructions, 2.5 min
    python scripts/brain_music.py --replay REC.npz       # replay a session through the speakers
    python scripts/brain_music.py --replay REC.npz --render OUT.wav   # offline, to a WAV

Live sessions are saved to output/music_<timestamp>.npz; replay or render
them later (the music is reproducible from the EEG).

Play it: the first ~23 s calibrate to your baseline — sit still, eyes open.
Then close your eyes and relax for more melody, focus for more rhythm,
blink for a chime, clench your jaw for a boom and a drum fill.
"""

import argparse
import asyncio
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import sounddevice as sd

from thebox.ble.connection import MuseConnection
from thebox.config import TheBoxConfig
from thebox.eeg.recording import Recording
from thebox.music.session import AUDIO_SR, HOP, BrainMusic, render_recording


# (seconds after the music starts, phase name, spoken cue) — macOS `say`
GUIDE = [
    (0, "calibration", "Calibrazione. Resta fermo, occhi aperti, sguardo davanti a te."),
    (26, "eyes_closed", "Ora chiudi gli occhi e rilassati."),
    (56, "focus", "Apri gli occhi. Conta all'indietro da cento, togliendo sette ogni volta."),
    (86, "rest", "Rilassati, occhi aperti."),
    (96, "blinks", "Sbatti le palpebre con decisione, tre volte."),
    (106, "clench", "Ora stringi forte la mascella, per un secondo."),
    (114, "clench", "Ancora: stringi la mascella."),
    (122, "eyes_closed", "Chiudi gli occhi e respira lentamente, fino alla fine."),
    (150, "end", "Fine. Grazie."),
]


def speak(text: str) -> None:
    subprocess.Popen(["say", "-v", "Alice", text])


def start_audio(music: BrainMusic) -> sd.OutputStream:
    def callback(outdata, frames, _time, status):
        outdata[:] = music.engine.render(frames)

    stream = sd.OutputStream(samplerate=AUDIO_SR, channels=2, blocksize=1024,
                             dtype="float32", latency="high", callback=callback)
    stream.start()
    return stream


async def run(music: BrainMusic, seconds: float | None, replay: Recording | None,
              guided: bool = False) -> Recording | None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, stop.set)

    rec, conn = None, None
    if replay is None:
        rec = Recording()
        conn = MuseConnection(TheBoxConfig().device_name, max_retries=5)
        conn.on_eeg(music.push)
        conn.on_eeg(rec.append)
        conn.on_disconnect(stop.set)
        await conn.connect()

    stream = start_audio(music)
    print("\nPlaying — Ctrl+C to stop.\n")
    if seconds:
        loop.call_later(seconds, stop.set)
    phases = []
    if guided:
        t0 = music.features.time  # EEG seconds already received when the music started

        def cue(name: str, text: str) -> None:
            phases.append((round(music.features.time - t0, 2), round(music.features.time, 2), name))
            speak(text)
            if name == "end":
                loop.call_later(3, stop.set)

        for at, name, text in GUIDE:
            loop.call_later(at, cue, name, text)
    pos, hop = 0, int(HOP * 256)
    try:
        while not stop.is_set():
            if replay is not None:
                if pos + hop > replay.n_samples:
                    break
                for i, ch in enumerate(replay.channels):
                    music.push(ch, replay.data[i, pos:pos + hop], replay.valid[i, pos:pos + hop])
                pos += hop
            music.tick()
            print(f"\r{music.status()}\033[K", end="", flush=True)
            try:
                await asyncio.wait_for(stop.wait(), HOP)
            except asyncio.TimeoutError:
                pass
    finally:
        print()
        if conn:
            await conn.disconnect()
        # Let the reverb tails ring out
        await asyncio.sleep(1.5)
        stream.stop()
        stream.close()
    if rec is not None and phases:
        rec.meta["phases"] = phases  # (music time, EEG time, phase)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", type=Path, help="play a saved .npz session instead of the Muse")
    ap.add_argument("--render", type=Path, help="with --replay: write a WAV instead of playing")
    ap.add_argument("--seconds", type=float, help="stop automatically after this long")
    ap.add_argument("--guided", action="store_true", help="speak a 2.5-minute routine of instructions")
    args = ap.parse_args()

    if args.render:
        if not args.replay:
            sys.exit("--render needs --replay")
        out = render_recording(Recording.load(args.replay), args.render)
        print(f"Wrote {out}")
        return

    replay = Recording.load(args.replay) if args.replay else None
    music = BrainMusic()
    rec = asyncio.run(run(music, args.seconds, replay, args.guided))

    if rec is not None and rec.n_samples:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = rec.save(f"output/music_{stamp}.npz")
        print(f"Saved session {path} ({rec.duration:.0f}s)")
        print(f"  replay:  python scripts/brain_music.py --replay {path}")
        print(f"  to WAV:  python scripts/brain_music.py --replay {path} --render output/music_{stamp}.wav")


if __name__ == "__main__":
    main()
