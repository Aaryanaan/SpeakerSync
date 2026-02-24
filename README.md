# SpeakerSync-Mac

Free, local alternative to [Airfoil](https://rogueamoeba.com/airfoil/). Routes system audio to two Bluetooth speakers simultaneously with per-speaker delay lines for manual phase alignment, stereo L/R split, bass boost, and stereo width control.

Built for syncing speakers with different DSP latencies (e.g., JBL Charge 4 + Charge 6) without relying on proprietary multi-speaker protocols.

## Features

- **Multi-speaker routing** — Captures system audio via virtual loopback (BlackHole / VB-Cable / PipeWire) and fans out to two independent Bluetooth outputs
- **Configurable delay lines** — Lock-free ring buffers with 0–500ms per-speaker delay, adjustable on the fly
- **Stereo split** — Left channel → Speaker A, Right channel → Speaker B, with adjustable crossfeed
- **Bass boost** — Low-shelf biquad filter (+/- 12dB at 150Hz)
- **Stereo width** — Mid/side processing for mono→extra-wide control
- **Drift compensation** — Automatic single-sample correction for DAC clock drift over long sessions
- **Presets** — `party`, `chill`, `flat` one-word presets from the CLI

## Quick Start

```bash
pip install sounddevice numpy
python3 bt_audio_sync.py --input 2 --output-a 0 --output-b 1 --delay-b 60
```

Use `python3 -c "import sounddevice; print(sounddevice.query_devices())"` to find your device indices.

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for virtual audio device setup on macOS, Windows, and Linux.

## Requirements

- Python 3.8+
- [PortAudio](http://www.portaudio.com/) system library
- A virtual audio loopback device ([BlackHole](https://existential.audio/blackhole/) on macOS, [VB-Cable](https://vb-audio.com/Cable/) on Windows, or PipeWire on Linux)
- Two Bluetooth speakers paired simultaneously

## License

MIT
