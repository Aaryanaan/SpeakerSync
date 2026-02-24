# bt-audio-sync — Complete Setup & Usage Guide

Everything you need to go from zero to synchronized stereo Bluetooth audio.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Install Python Dependencies](#2-install-python-dependencies)
3. [Set Up the Virtual Audio Loopback](#3-set-up-the-virtual-audio-loopback)
4. [Pair Your Bluetooth Speakers](#4-pair-your-bluetooth-speakers)
5. [Find Your Device Indices](#5-find-your-device-indices)
6. [Launch the App](#6-launch-the-app)
7. [How It Works](#7-how-it-works)
8. [CLI Command Reference](#8-cli-command-reference)
9. [Launch Flags Reference](#9-launch-flags-reference)
10. [Syncing Your Speakers (Delay Tuning)](#10-syncing-your-speakers-delay-tuning)
11. [Stereo & DSP Guide](#11-stereo--dsp-guide)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites

You need three things:

- **Python 3.8+** — check with `python3 --version`
- **Two Bluetooth speakers** — paired and connected to your computer simultaneously
- **A virtual audio loopback driver** — this is how the app intercepts system audio before it hits your hardware

---

## 2. Install Python Dependencies

```bash
pip install sounddevice numpy
```

On Linux you also need the PortAudio system library:

```bash
# Debian / Ubuntu
sudo apt install libportaudio2 portaudio19-dev

# Fedora
sudo dnf install portaudio portaudio-devel
```

---

## 3. Set Up the Virtual Audio Loopback

The app doesn't tap into your speakers directly — it reads from a virtual audio device that your OS sends all system audio into. Think of it as an invisible pipe between your apps and this tool.

### macOS — BlackHole (free, open source)

**Install:**

```bash
brew install blackhole-2ch
```

Or download from https://existential.audio/blackhole/

**Configure so you can still hear audio on your laptop:**

1. Open **Audio MIDI Setup** (Spotlight → type "Audio MIDI Setup")
2. Click the **+** button at the bottom-left → **Create Multi-Output Device**
3. Check the boxes for:
   - ✅ **BlackHole 2ch**
   - ✅ **MacBook Pro Speakers** (or whatever your built-in output is) — this is optional, for local monitoring
4. Go to **System Settings → Sound → Output** and select the **Multi-Output Device** you just created

Now all system audio goes to both BlackHole (which the app reads from) and your laptop speakers (so you can still hear things locally).

### Windows — VB-Cable (free)

**Install:**

1. Download from https://vb-audio.com/Cable/
2. Run the installer as **Administrator**
3. **Reboot**

**Configure:**

1. Right-click the speaker icon in your taskbar → **Sound Settings**
2. Set **Output** to **CABLE Input (VB-Audio Virtual Cable)**

All system audio now routes through VB-Cable. The app reads from **CABLE Output**.

**Optional — hear audio locally too:**

Right-click **CABLE Output** in Sound settings → Properties → Listen tab → check "Listen to this device" and pick your headphones/speakers.

### Linux — PipeWire

Most modern distros (Ubuntu 22.04+, Fedora 34+) already have PipeWire.

```bash
# Create a virtual sink
pactl load-module module-null-sink sink_name=virtual_out \
  sink_properties=device.description="BT_Sync_Virtual"

# Set it as your default output
pactl set-default-sink virtual_out
```

The app reads from **"Monitor of BT_Sync_Virtual"**.

To make this survive reboots, add to `~/.config/pipewire/pipewire.conf.d/virtual-sink.conf`:

```
context.modules = [
    {
        name = libpipewire-module-loopback
        args = {
            node.name = "bt_sync_virtual"
            node.description = "BT Sync Virtual Output"
            capture.props = { media.class = "Audio/Sink" }
            playback.props = { node.target = "" }
        }
    }
]
```

---

## 4. Pair Your Bluetooth Speakers

Both speakers must be paired and connected simultaneously.

**macOS:** System Settings → Bluetooth → pair each speaker. Both should show "Connected" at the same time.

**Windows:** Settings → Bluetooth & devices → Add device for each.

**Linux:**

```bash
bluetoothctl
> scan on
> pair <MAC_ADDRESS_SPEAKER_1>
> connect <MAC_ADDRESS_SPEAKER_1>
> pair <MAC_ADDRESS_SPEAKER_2>
> connect <MAC_ADDRESS_SPEAKER_2>
```

**Important:** Do NOT use JBL Connect+ or Auracast (JBL's built-in multi-speaker mode). We're bypassing that entirely so we have direct control over each speaker's audio and timing.

**Can't connect both?** Some Bluetooth adapters only support one A2DP connection at a time. You may need a USB Bluetooth 5.0+ dongle.

---

## 5. Find Your Device Indices

Run this to see all audio devices your system knows about:

```bash
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

You'll see something like:

```
  0 JBL Charge 4, Core Audio (0 in, 2 out)
  1 JBL Charge 6, Core Audio (0 in, 2 out)
  2 BlackHole 2ch, Core Audio (2 in, 2 out)
  3 MacBook Pro Microphone, Core Audio (1 in, 0 out)
  4 MacBook Pro Speakers, Core Audio (0 in, 2 out)
```

Write down three numbers:

| What | Device Name | Index (example) |
|------|-------------|-----------------|
| **Input** (virtual loopback) | BlackHole 2ch | 2 |
| **Output A** (Speaker A) | JBL Charge 4 | 0 |
| **Output B** (Speaker B) | JBL Charge 6 | 1 |

Your indices will vary. Use whatever numbers appear on your system.

---

## 6. Launch the App

### Basic launch (using device indices)

```bash
python3 bt_audio_sync.py --input 2 --output-a 0 --output-b 1
```

### With an initial delay on Speaker B

```bash
python3 bt_audio_sync.py --input 2 --output-a 0 --output-b 1 --delay-b 60
```

### Using name matching instead of indices

```bash
python3 bt_audio_sync.py --input-name "BlackHole" --output-a-name "Charge 4" --output-b-name "Charge 6"
```

### Interactive mode (pick devices from a list)

```bash
python3 bt_audio_sync.py
```

This shows all devices and prompts you to enter the index numbers.

### Then what?

Once the app starts, you'll see the `bt-sync>` prompt. Now just **play music from any app** — Spotify, Apple Music, YouTube, anything. The audio flows:

```
Your app → OS audio → Virtual loopback → bt_audio_sync → Both speakers
```

Use the commands below to adjust delay, stereo, bass, etc. while music is playing.

---

## 7. How It Works

The app captures raw audio from the virtual loopback device and writes it into two independent ring buffers (one per speaker). Each speaker has its own output stream reading from its ring buffer via non-blocking PortAudio callbacks.

**Delay** is not a sleep — it's a gap of silence at the front of the ring buffer. The write pointer starts ahead of the read pointer by exactly `delay_frames`. Changing the delay repositions the pointer.

**Stereo split** happens in the output callbacks. Speaker A gets a left-dominant mix, Speaker B gets a right-dominant mix, with crossfeed blending so you don't lose instruments panned to one side.

**Drift compensation** runs automatically. Because each speaker has its own DAC clock, they'll slowly drift apart over time (~1 sample/second). The app monitors buffer fill levels and drops or duplicates a single sample when needed — completely inaudible.

---

## 8. CLI Command Reference

Once the app is running, type these at the `bt-sync>` prompt. All changes take effect immediately.

### Delay

These **set** the delay to the given value (they don't add to the current delay).

| Command | What it does | Example |
|---------|-------------|---------|
| `da <ms>` | Set Speaker A delay (0–500 ms) | `da 120` sets A to 120ms |
| `db <ms>` | Set Speaker B delay (0–500 ms) | `db 85` sets B to 85ms |

**Speaker mapping:**
- `da` = whatever you passed as `--output-a` (e.g., JBL Charge 4)
- `db` = whatever you passed as `--output-b` (e.g., JBL Charge 6)

### Volume

| Command | What it does | Example |
|---------|-------------|---------|
| `va <0-100>` | Set Speaker A volume (percentage) | `va 80` sets A to 80% |
| `vb <0-100>` | Set Speaker B volume (percentage) | `vb 60` sets B to 60% |
| `ma` | Toggle mute on Speaker A | `ma` (mute), `ma` again (unmute) |
| `mb` | Toggle mute on Speaker B | `mb` |

### Stereo Mode

| Command | What it does |
|---------|-------------|
| `stereo` | **Default.** Speaker A gets the left channel, Speaker B gets the right channel, with crossfeed blending. This is the mode that makes two speakers sound like a real stereo pair. |
| `mono` | Both speakers get identical audio (L+R summed). Use this if the speakers are right next to each other. |
| `full` | Both speakers get the complete unmodified stereo signal. Each speaker plays both L and R. |

### DSP Controls

| Command | What it does | Range | Example |
|---------|-------------|-------|---------|
| `crossfeed <n>` | How much of the opposite channel bleeds in during stereo mode. Lower = more separation between speakers. | 0–50 | `crossfeed 20` (tighter split) |
| `bass <dB>` | Bass boost/cut. Applies a low-shelf filter at 150Hz. | -12 to +12 | `bass 6` (+6dB boost) |
| `width <n>` | Stereo width. Widens or narrows the stereo image using mid/side processing. | 0.0–2.0 | `width 1.5` (wider than original) |

**Crossfeed explained:**
- `crossfeed 0` — hard L/R split. Speaker A gets pure left, Speaker B gets pure right. Extreme separation.
- `crossfeed 30` — natural (default). Speaker A gets 70% left + 30% right. Sounds like sitting in a room with real speakers.
- `crossfeed 50` — effectively mono. No separation.

**Width explained:**
- `width 0` — collapses to mono
- `width 1.0` — original stereo (default)
- `width 1.5` — wider than the original recording
- `width 2.0` — maximum width (can sound unnatural, use sparingly)

**Bass explained:**
- `bass 0` — no boost (off)
- `bass 3` — subtle warmth
- `bass 6` — noticeable thump, good for pop/hip-hop
- `bass 8` — heavy, good for outdoor use where bass gets lost
- `bass -3` — bass cut (if the speakers sound boomy in a small room)

### Presets

One-word commands that set stereo mode, crossfeed, bass, and width all at once.

| Command | Stereo Mode | Crossfeed | Bass | Width | Good for |
|---------|-------------|-----------|------|-------|----------|
| `party` | stereo | 20% | +6 dB | 1.4x | Filling a room, outdoor hangouts |
| `chill` | stereo | 30% | +3 dB | 1.0x | Relaxed listening, background music |
| `flat` | stereo | 30% | off | 1.0x | Accurate playback, no coloring |

Your delay and volume settings are NOT changed by presets.

### System

| Command | What it does |
|---------|-------------|
| `status` | Shows buffer fill levels, underrun count, drift corrections, current DSP settings |
| `devices` | Re-lists all audio devices and their indices |
| `help` | Shows the quick-reference command list |
| `quit` / `q` | Stops audio and exits |

---

## 9. Launch Flags Reference

All flags are optional. If you don't specify devices, the app will prompt you interactively.

### Device Selection (by index)

| Flag | Description | Example |
|------|-------------|---------|
| `--input <n>` | Device index for virtual loopback input | `--input 2` |
| `--output-a <n>` | Device index for Speaker A | `--output-a 0` |
| `--output-b <n>` | Device index for Speaker B | `--output-b 1` |

### Device Selection (by name)

| Flag | Description | Example |
|------|-------------|---------|
| `--input-name <str>` | Match input device by name substring | `--input-name "BlackHole"` |
| `--output-a-name <str>` | Match Speaker A by name substring | `--output-a-name "Charge 4"` |
| `--output-b-name <str>` | Match Speaker B by name substring | `--output-b-name "Charge 6"` |

### Audio Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--sample-rate <Hz>` | 48000 | Sample rate. 48000 is standard for Bluetooth A2DP. Try 44100 if you get issues. |
| `--channels <n>` | 2 | Number of audio channels. Leave at 2. |
| `--blocksize <n>` | 512 | Frames per audio callback. Lower = less latency but more CPU. Increase to 1024 or 2048 if you get clicks/pops. |

### Initial Delays

| Flag | Default | Description |
|------|---------|-------------|
| `--delay-a <ms>` | 0 | Starting delay for Speaker A in milliseconds |
| `--delay-b <ms>` | 0 | Starting delay for Speaker B in milliseconds |

---

## 10. Syncing Your Speakers (Delay Tuning)

The whole point of this app. Because the Charge 4 and Charge 6 have different DSP chips, they process Bluetooth audio at different speeds. One speaker will always be slightly ahead of the other. The delay line lets you slow down the faster speaker so they match.

### Step by Step

1. Launch the app and start playing music (anything with a strong beat works)
2. Stand **equidistant** between both speakers
3. Listen for a "flam" — two distinct hits instead of one crisp hit on drums/percussion
4. If Speaker A (Charge 4) sounds **early** → increase `da` (e.g., `da 10`)
5. If Speaker B (Charge 6) sounds **early** → increase `db` (e.g., `db 10`)
6. Adjust in **5ms steps** until the hits merge into one

### Starting Point

The Charge 6 usually has lower latency than the Charge 4 (newer DSP), so a good starting command is:

```bash
python3 bt_audio_sync.py --input 2 --output-a 0 --output-b 1 --delay-b 60
```

Then fine-tune from there with `db 55`, `db 65`, etc.

### Alternative: Sine Wave Test

1. Play a 1kHz tone through your system
2. Walk between the speakers
3. If the tone "wobbles" or sounds hollow in the overlap zone, the speakers are out of phase
4. Adjust delay until the tone is smooth and consistent everywhere between the speakers

### Important

- `da 50` **sets** Speaker A's delay to 50ms. It does **not** add 50ms.
- Only delay one speaker at a time. If A is at 0 and B is at 60, don't move both — just adjust the one that's ahead.
- Your ideal delay value depends on your specific Bluetooth adapter, distance, and room. It won't match anyone else's setup exactly.

---

## 11. Stereo & DSP Guide

### What is stereo split?

By default, the app runs in `stereo` mode. This means:

- **Speaker A** gets the **left channel** of your music (with a little right mixed in via crossfeed)
- **Speaker B** gets the **right channel** (with a little left mixed in)

Place Speaker A on your left and Speaker B on your right. You'll hear instruments panned across the room — guitars on one side, keyboards on the other, vocals in the center. This is how music is meant to be heard.

### Speaker Placement

For the best stereo image:

```
        You
        🧑
       / | \
      /  |  \
     /   |   \
    🔊   |   🔊
   (A)   |   (B)
  LEFT       RIGHT
```

Aim for an equilateral triangle — speakers about as far apart as they are from you. Angle them slightly inward (toward you, not parallel).

### Crossfeed

If the speakers are close together, increase crossfeed (`crossfeed 40`). If they're far apart, decrease it (`crossfeed 15`) for more dramatic separation.

### Stereo Width

Width is applied before the stereo split. Increasing it exaggerates the difference between the left and right channels, making the stereo image feel wider. It's most noticeable on well-produced studio recordings. Live recordings or podcasts won't benefit much.

### Bass Boost

The low-shelf filter boosts everything below ~150Hz. The JBL Charge series has good bass drivers, especially the Charge 6, so even moderate boost (+3 to +6 dB) sounds clean. At +8 to +12 dB you'll feel it in your chest outdoors. Back off if it sounds muddy.

### Recipe: Outdoor Party

```
party
da 0
db 60
va 100
vb 100
```

### Recipe: Desktop Listening

```
chill
da 0
db 40
va 70
vb 70
```

### Recipe: Podcast / Spoken Word

```
mono
bass 0
width 1.0
```

---

## 12. Troubleshooting

### No audio coming through

- **Most common cause:** Your Mac's Sound Output is set to the speakers directly instead of the Multi-Output Device. Go to System Settings → Sound → Output and select the Multi-Output Device that includes BlackHole.
- Run `status` in the CLI. If "Input callbacks" is incrementing, audio is flowing in. If "Fill" on both speakers is 0, the output devices are wrong.
- Check your OS volume isn't muted.

### Audio clicks or pops

- Increase blocksize: launch with `--blocksize 1024` or `--blocksize 2048`
- Close CPU-heavy apps (browsers with many tabs, video editing, etc.)
- On macOS, make sure you're using **BlackHole 2ch**, not 16ch or 64ch

### One speaker is louder than the other

- Use `va` and `vb` to balance (e.g., `va 85` / `vb 100`)
- Also check the physical volume buttons on each speaker

### Speakers drift apart after 30+ minutes

- The drift compensator handles this automatically. Run `status` to check the "Drift corrections" count — a few corrections per hour is normal.
- If corrections are accumulating rapidly (>10/minute), try `--sample-rate 44100`

### Speakers won't connect simultaneously

- Your Bluetooth adapter may only support one A2DP connection
- Try a **USB Bluetooth 5.0+** dongle
- On Linux, check that PipeWire (not PulseAudio alone) is managing Bluetooth: `systemctl --user status pipewire`

### "PortAudio library not found"

```bash
# macOS
brew install portaudio

# Linux
sudo apt install libportaudio2 portaudio19-dev
```

### Bass boost sounds distorted

- You're clipping. Lower the volume (`va 70` / `vb 70`) or reduce bass (`bass 4`)
- High bass boost + high volume can push samples above 1.0 (digital clipping)

### I changed my delay and heard a click

- Normal. Changing delay repositions the ring buffer pointer, which can cause a single-sample discontinuity. It's one click, not ongoing. Adjust while music is playing and you likely won't notice.

### How do I stop the app?

- Type `quit` or `q` at the prompt
- Or press `Ctrl+C`
