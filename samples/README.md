# Sample kit

`scripts/brain_music.py` plays these WAVs. Any missing file is synthesised on
start-up, so delete one to get the default back.

To use your own sound, save a WAV (any sample rate, mono or stereo) with the
same name. Pitched samples are transposed by resampling, so they must be tuned
to the note below — or change the `root` for that entry in
`thebox/music/kit.py`.

| File | Tuned to | Plays when |
|---|---|---|
| `sub.wav` | D2 | always: bass drone, louder with delta |
| `pad.wav` | D4 | each chord change; louder with theta, brighter with alpha |
| `bell.wav` | D5 | melody when alpha ≥ beta (calm) |
| `pluck.wav` | D4 | melody when beta > alpha (alert) |
| `kick.wav` | — | beta above baseline |
| `hat.wav` | — | beta rising |
| `clap.wav` | — | high beta: backbeat |
| `tick.wav` | — | gamma (muscle tension): ghost clicks |
| `texture.wav` | — | looped wind layer, louder with gamma; must loop cleanly |
| `chime.wav` | — | blink |
| `boom.wav` | — | jaw clench |
