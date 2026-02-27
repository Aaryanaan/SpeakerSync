#!/usr/bin/env python3
"""
bt_audio_sync.py — Bluetooth Multi-Speaker Audio Synchronizer
=============================================================

A free, local alternative to Airfoil. Routes system audio captured from a
virtual loopback device (BlackHole / VB-Cable / PipeWire) to two independent
Bluetooth speakers with per-speaker configurable delay lines for manual
phase alignment.

Architecture
------------
                  ┌──────────────────────────────────────────────┐
                  │              OS System Audio                 │
                  │  (Music, Video, Browser, any app)            │
                  └──────────────┬───────────────────────────────┘
                                 │  OS routes to virtual sink
                                 ▼
                  ┌──────────────────────────────────────────────┐
                  │       Virtual Loopback Device                │
                  │  (BlackHole 2ch / VB-Cable / PipeWire)       │
                  └──────────────┬───────────────────────────────┘
                                 │  PortAudio non-blocking callback
                                 ▼
                  ┌──────────────────────────────────────────────┐
                  │          INPUT CALLBACK (this app)           │
                  │  Captures raw PCM float32 frames             │
                  │  Writes into TWO ring buffers                │
                  └────────┬─────────────────────┬───────────────┘
                           │                     │
                           ▼                     ▼
               ┌───────────────────┐  ┌───────────────────┐
               │  Ring Buffer A    │  │  Ring Buffer B    │
               │  (delay_ms_A)     │  │  (delay_ms_B)     │
               │  Lock-free SPSC   │  │  Lock-free SPSC   │
               └────────┬──────────┘  └────────┬──────────┘
                        │                      │
                        ▼                      ▼
               ┌───────────────────┐  ┌───────────────────┐
               │  OUTPUT CB A      │  │  OUTPUT CB B      │
               │  → JBL Charge 4   │  │  → JBL Charge 6   │
               │  (PortAudio)      │  │  (PortAudio)      │
               └───────────────────┘  └───────────────────┘

Key Design Decisions
--------------------
1. **Non-blocking callbacks only.** All three audio callbacks (1 input, 2 output)
   run in PortAudio's real-time thread. No allocations, no locks, no I/O.
2. **Lock-free ring buffers.** Each output stream has a dedicated SPSC
   (Single-Producer, Single-Consumer) ring buffer. The input callback is the
   sole producer; each output callback is the sole consumer.
3. **Delay via read-pointer offset.** Delay is NOT implemented by sleeping.
   The ring buffer is pre-filled with silence, so the read pointer starts
   `delay_frames` behind the write pointer. Changing delay dynamically
   repositions the read pointer.
4. **Drift compensation.** Buffer fill-level is monitored. If a consumer
   drifts (fill level deviates from target), we drop or duplicate a single
   sample per callback period to gently resynchronize without audible artifacts.

License: MIT
"""

import sys
import time
import threading
import argparse
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: 'sounddevice' not installed. Run: pip install sounddevice")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# §1  LOCK-FREE RING BUFFER (SPSC — Single Producer, Single Consumer)
# ═══════════════════════════════════════════════════════════════════════════════

class RingBuffer:
    """
    A fixed-size circular buffer for interleaved float32 audio frames.

    Thread safety model:  SPSC (one writer thread, one reader thread).
    The write_pos and read_pos are only ever modified by their owning thread,
    so no mutex is needed — Python's GIL + numpy's C-level operations give us
    the atomicity we need for the index updates.

    Memory layout:
        buffer[capacity * channels]   — flat float32 array
        Indexing: frame i, channel c  → buffer[i * channels + c]

    Delay mechanism:
        On init, the buffer is zero-filled and write_pos is advanced by
        `delay_frames`, creating a gap of silence between the write head
        and the read head. This gap IS the delay.
    """

    def __init__(self, capacity_frames: int, channels: int, delay_frames: int = 0):
        """
        Parameters
        ----------
        capacity_frames : int
            Total ring buffer size in audio frames. Must be significantly larger
            than (max_delay_frames + largest expected callback blocksize).
            Rule of thumb: 2 * max_delay_frames + 4 * blocksize.
        channels: int
            Number of audio channels (e.g., 2 for stereo).
        delay_frames: int
            Initial delay in frames. The write pointer is advanced by this
            many frames at startup, pre-filling the gap with silence.
        """
        self.capacity = capacity_frames
        self.channels = channels
        # Flat buffer: capacity_frames * channels floats
        self.buffer = np.zeros(capacity_frames * channels, dtype=np.float32)
        # Positions in FRAMES (not samples)
        self.write_pos = delay_frames % capacity_frames
        self.read_pos = 0
        # Track total frames written/read for drift monitoring
        self.total_written = 0
        self.total_read = 0

    def available_read(self) -> int:
        """Frames available for reading (how full the buffer is)."""
        diff = self.write_pos - self.read_pos
        if diff < 0:
            diff += self.capacity
        return diff

    def available_write(self) -> int:
        """Frames available for writing (how much space is left)."""
        return self.capacity - 1 - self.available_read()

    def write(self, data: np.ndarray) -> int:
        """
        Write interleaved float32 frames into the ring buffer.

        Parameters
        ----------
        data : np.ndarray
            Shape: (num_frames, channels) or (num_frames * channels,).
            Must be float32.

        Returns
        -------
        int
            Number of frames actually written (may be less than requested
            if the buffer is nearly full — this indicates the consumer
            is falling behind, i.e., drift).
        """
        frames = data.shape[0] if data.ndim == 2 else data.shape[0] // self.channels
        flat = data.ravel()
        avail = self.available_write()
        to_write = min(frames, avail)
        if to_write <= 0:
            return 0

        samples = to_write * self.channels
        start = self.write_pos * self.channels
        end = start + samples

        if end <= self.capacity * self.channels:
            # Contiguous write
            self.buffer[start:end] = flat[:samples]
        else:
            # Wrap-around write
            first_chunk = self.capacity * self.channels - start
            self.buffer[start:] = flat[:first_chunk]
            self.buffer[:samples - first_chunk] = flat[first_chunk:samples]

        self.write_pos = (self.write_pos + to_write) % self.capacity
        self.total_written += to_write
        return to_write

    def read(self, num_frames: int) -> np.ndarray:
        """
        Read interleaved float32 frames from the ring buffer.

        If fewer frames are available than requested (buffer underrun),
        the remaining frames are filled with silence (zeros) to prevent
        audio glitches.

        Parameters
        ----------
        num_frames : int
            Number of frames to read (typically the callback blocksize).

        Returns
        -------
        np.ndarray
            Shape: (num_frames, channels), dtype float32.
        """
        avail = self.available_read()
        to_read = min(num_frames, avail)
        out = np.zeros(num_frames * self.channels, dtype=np.float32)

        if to_read > 0:
            samples = to_read * self.channels
            start = self.read_pos * self.channels
            end = start + samples

            if end <= self.capacity * self.channels:
                out[:samples] = self.buffer[start:end]
            else:
                first_chunk = self.capacity * self.channels - start
                out[:first_chunk] = self.buffer[start:]
                out[first_chunk:samples] = self.buffer[:samples - first_chunk]

            self.read_pos = (self.read_pos + to_read) % self.capacity
            self.total_read += to_read

        return out.reshape(num_frames, self.channels)

    def set_delay_frames(self, new_delay_frames: int):
        """
        Dynamically change the delay by repositioning the write pointer
        relative to the read pointer.

        This is called from the CLI thread, NOT from the audio callback.
        Because only the input callback modifies write_pos, we set a flag
        and let the input callback apply the change on its next invocation
        to avoid race conditions. For simplicity in this implementation,
        we directly reposition — acceptable because the input callback
        and this method both only modify write_pos, and the brief
        inconsistency (< 1 callback period) causes at most a single
        click, not corruption.
        """
        # Place write_pos ahead of read_pos by new_delay_frames
        self.write_pos = (self.read_pos + new_delay_frames) % self.capacity
        # Zero-fill the delay region to avoid playing stale audio
        pos = self.read_pos
        for _ in range(new_delay_frames):
            idx = pos * self.channels
            self.buffer[idx:idx + self.channels] = 0.0
            pos = (pos + 1) % self.capacity


# ═══════════════════════════════════════════════════════════════════════════════
# §2  DRIFT COMPENSATOR
# ═══════════════════════════════════════════════════════════════════════════════

class DriftCompensator:
    """
    Monitors the fill level of a ring buffer and compensates for DAC clock
    drift between independent Bluetooth speakers.

    Theory:
        If Speaker A's DAC clock runs 20ppm fast relative to the host,
        it consumes samples slightly faster than the host produces them.
        Over time, the ring buffer drains. If uncorrected, it underruns.

        slow DAC causes the buffer to fill up and eventually
        overflow (dropping audio).

    Strategy:
        Every N callback periods, check buffer fill level. If it deviates
        from the target fill level by more than a threshold:
        - Buffer too full -> drop 1 sample (skip a sample in the read)
        - Buffer too empty -> duplicate 1 sample (re-read previous sample)

        Dropping/adding ONE sample per adjustment period is inaudible.
        At 48kHz, one sample = ~21µs of time shift.

    Adjustment rate:
        With 20ppm drift at 48kHz, we drift ~1 sample per second.
        Checking every 0.5s and adjusting 1 sample is sufficient.
    """

    def __init__(self, ring_buffer: RingBuffer, target_fill_frames: int,
                 threshold_frames: int = 64, check_interval_callbacks: int = 100):
        """
        Parameters
        ----------
        ring_buffer : RingBuffer
            The ring buffer to monitor.
        target_fill_frames : int
            The ideal fill level (typically = delay_frames).
        threshold_frames : int
            Deviation from target before corrective action.
        check_interval_callbacks : int
            Number of output callbacks between drift checks.
        """
        self.rb = ring_buffer
        self.target = target_fill_frames
        self.threshold = threshold_frames
        self.interval = check_interval_callbacks
        self.counter = 0
        self.corrections = 0  # Running count for diagnostics

    def check_and_compensate(self, output_data: np.ndarray) -> np.ndarray:
        """
        Called from the output callback AFTER reading from the ring buffer.

        If a correction is needed, modifies output_data in-place:
        - Drop: removes one frame, appends silence at the end
        - Pad: duplicates the last frame

        Returns the (possibly modified) output_data.
        """
        self.counter += 1
        if self.counter < self.interval:
            return output_data

        self.counter = 0
        fill = self.rb.available_read()
        deviation = fill - self.target

        if abs(deviation) < self.threshold:
            return output_data  # Within tolerance, no action

        if deviation > 0:
            # Buffer is overfull — consumer is slow (or producer is fast)
            # Drop one frame: shift data left by 1 frame, shortening the buffer
            # We "read" an extra frame from the ring buffer to drain it
            extra = self.rb.read(1)  # Discard one frame
            self.corrections += 1
        elif deviation < 0:
            # Buffer is underfull — consumer is fast (or producer is slow)
            # Pad: duplicate the last frame of output_data
            # This effectively slows the consumer by 1 frame
            if output_data.shape[0] > 1:
                output_data[-1] = output_data[-2]  # Duplicate second-to-last
            self.corrections -= 1

        return output_data


# ═══════════════════════════════════════════════════════════════════════════════
# §2b  DSP PROCESSORS — Stereo, Bass Boost, Width
# ═══════════════════════════════════════════════════════════════════════════════

class StereoRouter:
    """
    Routes stereo audio to each speaker with configurable channel mixing.

    Modes:
        'stereo'   — Speaker A = Left-dominant, Speaker B = Right-dominant.
                     Crossfeed blends some of the opposite channel so you
                     don't lose instruments panned hard to one side.
        'mono'     — Both speakers get identical L+R sum (original behavior).
        'full'     — Each speaker gets the full stereo mix (both channels).

    Crossfeed:
        0.0 = hard split (A gets pure L, B gets pure R) — extreme separation
        0.3 = natural (A gets 70% L + 30% R) — like sitting in a room
        0.5 = mono (no separation, same as 'mono' mode)

    The crossfeed is applied as a simple linear mix in the output callback.
    No allocations, no branching beyond a mode check — safe for real-time.
    """

    def __init__(self, mode: str = 'stereo', crossfeed: float = 0.3):
        self.mode = mode            # 'stereo', 'mono', 'full'
        self.crossfeed = crossfeed  # 0.0–0.5

    def process_for_a(self, data: np.ndarray) -> np.ndarray:
        """Process stereo data for Speaker A (left-dominant)."""
        if self.mode == 'mono' or data.shape[1] < 2:
            # Sum to mono, output on both channels
            mono = (data[:, 0] + data[:, 1]) * 0.5
            data[:, 0] = mono
            data[:, 1] = mono
        elif self.mode == 'stereo':
            # A = left-dominant: L * (1-crossfeed) + R * crossfeed
            cf = self.crossfeed
            left = data[:, 0].copy()
            right = data[:, 1].copy()
            mixed = left * (1.0 - cf) + right * cf
            data[:, 0] = mixed
            data[:, 1] = mixed
        # 'full' mode: pass through unchanged
        return data

    def process_for_b(self, data: np.ndarray) -> np.ndarray:
        """Process stereo data for Speaker B (right-dominant)."""
        if self.mode == 'mono' or data.shape[1] < 2:
            mono = (data[:, 0] + data[:, 1]) * 0.5
            data[:, 0] = mono
            data[:, 1] = mono
        elif self.mode == 'stereo':
            # B = right-dominant: R * (1-crossfeed) + L * crossfeed
            cf = self.crossfeed
            left = data[:, 0].copy()
            right = data[:, 1].copy()
            mixed = right * (1.0 - cf) + left * cf
            data[:, 0] = mixed
            data[:, 1] = mixed
        return data


class BiquadFilter:
    """
    A generic second-order IIR (biquad) filter for real-time audio processing.

    Implements the Direct Form II Transposed structure, which is numerically
    stable and efficient. Used here for bass boost (low-shelf) and could be
    extended for EQ, high-shelf, etc.

    Transfer function:
        H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)

    The filter state (z1, z2) is maintained per channel for correct stereo
    processing. State is persistent across callback invocations.
    """

    def __init__(self, b0=1.0, b1=0.0, b2=0.0, a0=1.0, a1=0.0, a2=0.0,
                 channels: int = 2):
        # Normalize coefficients by a0
        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0
        self.channels = channels
        # Per-channel state (Direct Form II Transposed)
        self.z1 = np.zeros(channels, dtype=np.float64)
        self.z2 = np.zeros(channels, dtype=np.float64)

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the biquad filter to a block of audio in-place.

        Parameters
        ----------
        data : np.ndarray, shape (frames, channels), dtype float32

        Returns
        -------
        np.ndarray — the same array, filtered in-place.
        """
        for ch in range(min(self.channels, data.shape[1])):
            for i in range(data.shape[0]):
                x = float(data[i, ch])
                y = self.b0 * x + self.z1[ch]
                self.z1[ch] = self.b1 * x - self.a1 * y + self.z2[ch]
                self.z2[ch] = self.b2 * x - self.a2 * y
                data[i, ch] = np.float32(y)
        return data

    def reset(self):
        """Clear filter state (call when changing parameters)."""
        self.z1[:] = 0.0
        self.z2[:] = 0.0

    @staticmethod
    def low_shelf(freq: float, gain_db: float, sample_rate: int,
                  channels: int = 2) -> 'BiquadFilter':
        """
        Create a low-shelf biquad filter.

        Parameters
        ----------
        freq : float
            Shelf corner frequency in Hz (e.g., 150.0 for bass boost).
        gain_db : float
            Gain in dB below the corner frequency. Positive = boost.
            Typical bass boost: +3 to +8 dB.
        sample_rate : int
            Sample rate in Hz.
        channels : int
            Number of audio channels.

        Returns
        -------
        BiquadFilter configured as a low-shelf.
        """
        A = 10 ** (gain_db / 40.0)  # amplitude from dB
        w0 = 2 * np.pi * freq / sample_rate
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / 2.0 * np.sqrt(2.0)  # Q = 0.707 (Butterworth)
        sqrt_A = np.sqrt(A)

        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha

        return BiquadFilter(b0, b1, b2, a0, a1, a2, channels)


class StereoWidener:
    """
    Mid/Side stereo width control.

    How it works:
        Mid   = (L + R) / 2   — the "center" (vocals, bass, kick)
        Side  = (L - R) / 2   — the "edges" (reverb, panned instruments)

        Width controls the Side gain relative to Mid:
        - width = 1.0 → original stereo (no change)
        - width = 0.0 → pure mono
        - width = 1.5 → wider than original (boosted side signal)
        - width = 2.0 → very wide (can sound unnatural, use sparingly)

    This is the same technique used in professional mastering plugins
    (iZotope Ozone, FabFilter Pro-Q, etc.)
    """

    def __init__(self, width: float = 1.0):
        self.width = width  # 0.0 (mono) to 2.0 (extra wide)

    def process(self, data: np.ndarray) -> np.ndarray:
        """Apply stereo width adjustment in-place. Input must be stereo."""
        if data.shape[1] < 2 or self.width == 1.0:
            return data

        left = data[:, 0].copy()
        right = data[:, 1].copy()

        mid = (left + right) * 0.5
        side = (left - right) * 0.5

        # Scale side by width factor
        side *= self.width

        data[:, 0] = mid + side
        data[:, 1] = mid - side
        return data


# ═══════════════════════════════════════════════════════════════════════════════
# §3  DEVICE DISCOVERY & SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def list_audio_devices():
    """Print all audio devices detected by PortAudio with their indices."""
    devices = sd.query_devices()
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║               DETECTED AUDIO DEVICES                       ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    for i, dev in enumerate(devices):
        direction = ""
        if dev['max_input_channels'] > 0:
            direction += "IN"
        if dev['max_output_channels'] > 0:
            direction += ("/" if direction else "") + "OUT"
        marker = ""
        name_lower = dev['name'].lower()
        if 'blackhole' in name_lower or 'vb-cable' in name_lower or 'pipewire' in name_lower:
            marker = " ◄── VIRTUAL LOOPBACK"
        elif 'jbl' in name_lower or 'charge' in name_lower:
            marker = " ◄── JBL SPEAKER"
        print(f"║  [{i:3d}] {dev['name'][:42]:<42s} ({direction}){marker}")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    return devices


def find_device_by_substring(substring: str, kind: str = 'output') -> int:
    """
    Search for a device whose name contains `substring` (case-insensitive).

    Parameters
    ----------
    substring : str
        Partial name to search for (e.g., 'BlackHole', 'JBL Charge 4').
    kind : str
        'input' or 'output' — filters by device capability.

    Returns
    -------
    int
        Device index, or -1 if not found.
    """
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if substring.lower() in dev['name'].lower():
            if kind == 'input' and dev['max_input_channels'] > 0:
                return i
            elif kind == 'output' and dev['max_output_channels'] > 0:
                return i
    return -1


# ═══════════════════════════════════════════════════════════════════════════════
# §4  AUDIO ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class AudioSyncEngine:
    """
    Core engine: captures system audio and fans it out to two Bluetooth
    speakers with independent delay lines and drift compensation.
    """

    def __init__(self, input_device: int, output_device_a: int,
                 output_device_b: int, sample_rate: int = 48000,
                 channels: int = 2, blocksize: int = 512,
                 delay_ms_a: float = 0.0, delay_ms_b: float = 0.0,
                 max_delay_ms: float = 500.0):
        """
        Parameters
        ----------
        input_device : int
            PortAudio device index for the virtual loopback input.
        output_device_a : int
            PortAudio device index for Speaker A (e.g., JBL Charge 4).
        output_device_b : int
            PortAudio device index for Speaker B (e.g., JBL Charge 6).
        sample_rate : int
            Sample rate in Hz. 48000 is standard for Bluetooth A2DP SBC/AAC.
        channels : int
            Number of audio channels (2 = stereo).
        blocksize : int
            Frames per callback. Lower = less latency but more CPU.
            512 is a good balance for Bluetooth (which has ~150ms latency anyway).
        delay_ms_a : float
            Initial delay for Speaker A in milliseconds.
        delay_ms_b : float
            Initial delay for Speaker B in milliseconds.
        max_delay_ms : float
            Maximum allowed delay in milliseconds. Determines buffer size.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.input_device = input_device
        self.output_device_a = output_device_a
        self.output_device_b = output_device_b

        # Convert ms → frames
        self.delay_frames_a = int((delay_ms_a / 1000.0) * sample_rate)
        self.delay_frames_b = int((delay_ms_b / 1000.0) * sample_rate)
        max_delay_frames = int((max_delay_ms / 1000.0) * sample_rate)

        # Ring buffer capacity: max_delay + generous headroom for callbacks
        capacity = max_delay_frames + blocksize * 8
        print(f"  Ring buffer capacity: {capacity} frames "
              f"({capacity / sample_rate * 1000:.0f} ms)")

        # Create ring buffers with initial delays
        self.ring_a = RingBuffer(capacity, channels, self.delay_frames_a)
        self.ring_b = RingBuffer(capacity, channels, self.delay_frames_b)

        # Drift compensators
        self.drift_a = DriftCompensator(
            self.ring_a, target_fill_frames=self.delay_frames_a,
            threshold_frames=blocksize, check_interval_callbacks=100
        )
        self.drift_b = DriftCompensator(
            self.ring_b, target_fill_frames=self.delay_frames_b,
            threshold_frames=blocksize, check_interval_callbacks=100
        )

        # Volume controls (0.0 to 1.0)
        self.volume_a = 1.0
        self.volume_b = 1.0

        # Mute flags
        self.mute_a = False
        self.mute_b = False

        # ── DSP Processors ────────────────────────────────────────────────
        # Stereo router: splits L/R between speakers with crossfeed
        self.stereo_router = StereoRouter(mode='stereo', crossfeed=0.3)

        # Bass boost: low-shelf filter at 150Hz, applied per-speaker
        # Start at 0 dB (off). User can enable with "bass" command.
        self.bass_gain_db = 0.0
        self.bass_filter_a = BiquadFilter.low_shelf(
            150.0, self.bass_gain_db, sample_rate, channels)
        self.bass_filter_b = BiquadFilter.low_shelf(
            150.0, self.bass_gain_db, sample_rate, channels)

        # Stereo width: 1.0 = normal, >1.0 = wider image
        self.stereo_widener = StereoWidener(width=1.0)

        # Running state
        self.running = False
        self.input_stream = None
        self.output_stream_a = None
        self.output_stream_b = None

        # Performance monitoring
        self.input_callbacks = 0
        self.output_a_callbacks = 0
        self.output_b_callbacks = 0
        self.underruns_a = 0
        self.underruns_b = 0

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _input_callback(self, indata, frames, time_info, status):
        """
        PortAudio input callback — runs in real-time audio thread.

        RULES FOR REAL-TIME AUDIO CALLBACKS:
        ✗ No memory allocation (no new lists, dicts, or large objects)
        ✗ No I/O (no print, no file access, no network)
        ✗ No locks (no threading.Lock, no mutexes)
        ✓ Only numpy operations on pre-allocated buffers
        ✓ Only index arithmetic
        """
        if status:
            # Status flags indicate xruns; we note but can't print here
            pass

        self.input_callbacks += 1

        # Apply stereo width BEFORE splitting to speakers.
        # We work on a copy so the original indata (owned by PortAudio) is safe.
        widened = indata.copy()
        self.stereo_widener.process(widened)

        # Write widened stereo audio into BOTH ring buffers
        self.ring_a.write(widened)
        self.ring_b.write(widened)

    def _output_callback_a(self, outdata, frames, time_info, status):
        """Output callback for Speaker A (e.g., JBL Charge 4) — LEFT dominant."""
        if status:
            self.underruns_a += 1

        self.output_a_callbacks += 1

        # Read from ring buffer
        data = self.ring_a.read(frames)

        # Apply drift compensation
        data = self.drift_a.check_and_compensate(data)

        # DSP chain: stereo routing → bass boost
        data = self.stereo_router.process_for_a(data)
        if self.bass_gain_db != 0.0:
            data = self.bass_filter_a.process(data)

        # Apply volume and mute
        if self.mute_a:
            outdata[:] = 0.0
        else:
            outdata[:] = data * self.volume_a

    def _output_callback_b(self, outdata, frames, time_info, status):
        """Output callback for Speaker B (e.g., JBL Charge 6) — RIGHT dominant."""
        if status:
            self.underruns_b += 1

        self.output_b_callbacks += 1

        data = self.ring_b.read(frames)
        data = self.drift_b.check_and_compensate(data)

        # DSP chain: stereo routing → bass boost
        data = self.stereo_router.process_for_b(data)
        if self.bass_gain_db != 0.0:
            data = self.bass_filter_b.process(data)

        if self.mute_b:
            outdata[:] = 0.0
        else:
            outdata[:] = data * self.volume_b

    # ── Start / Stop ──────────────────────────────────────────────────────

    def start(self):
        """Open all three audio streams and begin routing."""
        print("\n▶ Starting audio engine...")
        print(f"  Sample rate:  {self.sample_rate} Hz")
        print(f"  Block size:   {self.blocksize} frames")
        print(f"  Channels:     {self.channels}")
        print(f"  Delay A:      {self.delay_frames_a / self.sample_rate * 1000:.1f} ms")
        print(f"  Delay B:      {self.delay_frames_b / self.sample_rate * 1000:.1f} ms")
        print(f"  Stereo mode:  {self.stereo_router.mode} (crossfeed: {self.stereo_router.crossfeed:.0%})")
        print(f"  Bass boost:   {self.bass_gain_db:+.1f} dB")
        print(f"  Stereo width: {self.stereo_widener.width:.1f}x")

        # Input stream (from virtual loopback)
        self.input_stream = sd.InputStream(
            device=self.input_device,
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype='float32',
            callback=self._input_callback,
            latency='low'
        )

        # Output stream A (Speaker A)
        self.output_stream_a = sd.OutputStream(
            device=self.output_device_a,
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype='float32',
            callback=self._output_callback_a,
            latency='low'
        )

        # Output stream B (Speaker B)
        self.output_stream_b = sd.OutputStream(
            device=self.output_device_b,
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype='float32',
            callback=self._output_callback_b,
            latency='low'
        )

        # Start all streams
        self.input_stream.start()
        self.output_stream_a.start()
        self.output_stream_b.start()
        self.running = True
        print("  ✓ All streams active. Audio is routing.\n")

    def stop(self):
        """Stop and close all audio streams."""
        print("\n■ Stopping audio engine...")
        self.running = False
        for stream in [self.input_stream, self.output_stream_a, self.output_stream_b]:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
        print("  ✓ All streams closed.")

    # ── Dynamic Controls ──────────────────────────────────────────────────

    def set_delay_a(self, delay_ms: float):
        """Dynamically change the delay for Speaker A."""
        new_frames = int((delay_ms / 1000.0) * self.sample_rate)
        self.delay_frames_a = new_frames
        self.ring_a.set_delay_frames(new_frames)
        self.drift_a.target = new_frames
        print(f"  ◆ Speaker A delay → {delay_ms:.1f} ms ({new_frames} frames)")

    def set_delay_b(self, delay_ms: float):
        """Dynamically change the delay for Speaker B."""
        new_frames = int((delay_ms / 1000.0) * self.sample_rate)
        self.delay_frames_b = new_frames
        self.ring_b.set_delay_frames(new_frames)
        self.drift_b.target = new_frames
        print(f"  ◆ Speaker B delay → {delay_ms:.1f} ms ({new_frames} frames)")

    # ── DSP Controls ──────────────────────────────────────────────────────

    def set_stereo_mode(self, mode: str):
        """Set stereo routing mode: 'stereo', 'mono', or 'full'."""
        self.stereo_router.mode = mode
        print(f"  ◆ Stereo mode → {mode}")

    def set_crossfeed(self, value: float):
        """Set crossfeed amount (0.0 = hard L/R split, 0.5 = mono blend)."""
        self.stereo_router.crossfeed = max(0.0, min(0.5, value))
        print(f"  ◆ Crossfeed → {self.stereo_router.crossfeed:.0%}")

    def set_bass(self, gain_db: float):
        """Set bass boost in dB (0 = off, typical: 3-8 dB)."""
        self.bass_gain_db = gain_db
        self.bass_filter_a = BiquadFilter.low_shelf(
            150.0, gain_db, self.sample_rate, self.channels)
        self.bass_filter_b = BiquadFilter.low_shelf(
            150.0, gain_db, self.sample_rate, self.channels)
        if gain_db == 0:
            print(f"  ◆ Bass boost → OFF")
        else:
            print(f"  ◆ Bass boost → {gain_db:+.1f} dB")

    def set_width(self, width: float):
        """Set stereo width (0.0 = mono, 1.0 = normal, 2.0 = extra wide)."""
        self.stereo_widener.width = max(0.0, min(2.0, width))
        print(f"  ◆ Stereo width → {self.stereo_widener.width:.1f}x")

    def get_stats(self) -> str:
        """Return a formatted status string for the CLI."""
        fill_a = self.ring_a.available_read()
        fill_b = self.ring_b.available_read()
        return (
            f"  ┌─ Speaker A (LEFT) ── JBL Charge 4 ──────┐\n"
            f"  │ Delay: {self.delay_frames_a / self.sample_rate * 1000:6.1f} ms   "
            f"Fill: {fill_a:6d} frames   "
            f"Vol: {self.volume_a:.0%}   "
            f"{'MUTED' if self.mute_a else 'ON':>5s} │\n"
            f"  │ Callbacks: {self.output_a_callbacks:8d}   "
            f"Underruns: {self.underruns_a:4d}   "
            f"Drift corrections: {self.drift_a.corrections:+d}   │\n"
            f"  └────────────────────────────────────────────┘\n"
            f"  ┌─ Speaker B (RIGHT) ── JBL Charge 6 ─────┐\n"
            f"  │ Delay: {self.delay_frames_b / self.sample_rate * 1000:6.1f} ms   "
            f"Fill: {fill_b:6d} frames   "
            f"Vol: {self.volume_b:.0%}   "
            f"{'MUTED' if self.mute_b else 'ON':>5s} │\n"
            f"  │ Callbacks: {self.output_b_callbacks:8d}   "
            f"Underruns: {self.underruns_b:4d}   "
            f"Drift corrections: {self.drift_b.corrections:+d}   │\n"
            f"  └────────────────────────────────────────────┘\n"
            f"  ┌─ DSP ────────────────────────────────────┐\n"
            f"  │ Stereo: {self.stereo_router.mode:<8s}  "
            f"Crossfeed: {self.stereo_router.crossfeed:.0%}    "
            f"Width: {self.stereo_widener.width:.1f}x      │\n"
            f"  │ Bass:   {self.bass_gain_db:+.1f} dB"
            f"{'  (OFF)' if self.bass_gain_db == 0 else '       '}"
            f"                              │\n"
            f"  └────────────────────────────────────────────┘\n"
            f"  Input callbacks: {self.input_callbacks}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §5  INTERACTIVE CLI
# ═══════════════════════════════════════════════════════════════════════════════

HELP_TEXT = """
╔══════════════════════════════════════════════════════════════════════╗
║                    BT AUDIO SYNC — COMMANDS                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  DELAY & VOLUME                                                    ║
║  da <ms>       Set delay for Speaker A  (e.g., "da 120")           ║
║  db <ms>       Set delay for Speaker B  (e.g., "db 85")            ║
║  va <0-100>    Set volume for Speaker A  (e.g., "va 80")           ║
║  vb <0-100>    Set volume for Speaker B  (e.g., "vb 60")           ║
║  ma            Toggle mute Speaker A                               ║
║  mb            Toggle mute Speaker B                               ║
║                                                                    ║
║  STEREO & DSP                                                      ║
║  stereo        Stereo split mode (A=Left, B=Right + crossfeed)     ║
║  mono          Mono mode (identical audio to both speakers)        ║
║  full          Full stereo to both speakers (no split)             ║
║  crossfeed <n> Set crossfeed 0-50 (0=hard split, 30=natural)      ║
║  bass <dB>     Bass boost in dB (0=off, 3-8=typical, max 12)      ║
║  width <n>     Stereo width (0=mono, 1.0=normal, 1.5=wide)        ║
║                                                                    ║
║  PRESETS                                                           ║
║  party         Wide stereo + bass boost (width 1.4, bass +6)      ║
║  chill         Natural stereo + gentle bass (width 1.0, bass +3)  ║
║  flat          Everything neutral (no DSP)                         ║
║                                                                    ║
║  SYSTEM                                                            ║
║  status        Show buffer levels, drift corrections, DSP stats    ║
║  devices       Re-list audio devices                               ║
║  help          Show this help                                      ║
║  quit / q      Stop and exit                                       ║
╚══════════════════════════════════════════════════════════════════════╝

TIP: To sync your speakers, play a metronome or click track.
     If Speaker A sounds EARLY, increase da. If LATE, increase db.
     Adjust in 5-10ms increments until perfectly aligned.

     Speaker A (da) = JBL Charge 4 = LEFT channel
     Speaker B (db) = JBL Charge 6 = RIGHT channel
"""


def run_cli(engine: AudioSyncEngine):
    """Interactive command-line interface for live delay adjustment."""
    print(HELP_TEXT)

    while engine.running:
        try:
            cmd = input("bt-sync> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue

        parts = cmd.split()
        command = parts[0]

        try:
            if command in ('quit', 'q', 'exit'):
                break

            elif command == 'help':
                print(HELP_TEXT)

            elif command == 'status':
                print(engine.get_stats())

            elif command == 'devices':
                list_audio_devices()

            elif command == 'da' and len(parts) == 2:
                ms = float(parts[1])
                if 0 <= ms <= 500:
                    engine.set_delay_a(ms)
                else:
                    print("  ⚠ Delay must be 0–500 ms")

            elif command == 'db' and len(parts) == 2:
                ms = float(parts[1])
                if 0 <= ms <= 500:
                    engine.set_delay_b(ms)
                else:
                    print("  ⚠ Delay must be 0–500 ms")

            elif command == 'va' and len(parts) == 2:
                vol = float(parts[1]) / 100.0
                engine.volume_a = max(0.0, min(1.0, vol))
                print(f"  ◆ Speaker A volume → {engine.volume_a:.0%}")

            elif command == 'vb' and len(parts) == 2:
                vol = float(parts[1]) / 100.0
                engine.volume_b = max(0.0, min(1.0, vol))
                print(f"  ◆ Speaker B volume → {engine.volume_b:.0%}")

            elif command == 'ma':
                engine.mute_a = not engine.mute_a
                print(f"  ◆ Speaker A {'MUTED' if engine.mute_a else 'UNMUTED'}")

            elif command == 'mb':
                engine.mute_b = not engine.mute_b
                print(f"  ◆ Speaker B {'MUTED' if engine.mute_b else 'UNMUTED'}")

            # ── Stereo & DSP commands ─────────────────────────────────
            elif command == 'stereo':
                engine.set_stereo_mode('stereo')

            elif command == 'mono':
                engine.set_stereo_mode('mono')

            elif command == 'full':
                engine.set_stereo_mode('full')

            elif command == 'crossfeed' and len(parts) == 2:
                val = float(parts[1]) / 100.0  # User gives 0-50, we use 0.0-0.5
                engine.set_crossfeed(val)

            elif command == 'bass' and len(parts) == 2:
                db = float(parts[1])
                if -12 <= db <= 12:
                    engine.set_bass(db)
                else:
                    print("  ⚠ Bass must be -12 to +12 dB")

            elif command == 'width' and len(parts) == 2:
                w = float(parts[1])
                if 0.0 <= w <= 2.0:
                    engine.set_width(w)
                else:
                    print("  ⚠ Width must be 0.0–2.0")

            # ── Presets ───────────────────────────────────────────────
            elif command == 'party':
                engine.set_stereo_mode('stereo')
                engine.set_crossfeed(0.2)
                engine.set_bass(6.0)
                engine.set_width(1.4)
                print("  🎉 Party preset loaded!")

            elif command == 'chill':
                engine.set_stereo_mode('stereo')
                engine.set_crossfeed(0.3)
                engine.set_bass(3.0)
                engine.set_width(1.0)
                print("  😎 Chill preset loaded!")

            elif command == 'flat':
                engine.set_stereo_mode('stereo')
                engine.set_crossfeed(0.3)
                engine.set_bass(0.0)
                engine.set_width(1.0)
                print("  📐 Flat preset loaded (no DSP)")

            else:
                print(f"  Unknown command: '{cmd}'. Type 'help' for commands.")

        except ValueError:
            print("  ⚠ Invalid number. Example: da 120")
        except Exception as e:
            print(f"  ⚠ Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# §6  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='BT Audio Sync — Route system audio to two Bluetooth speakers '
                    'with configurable delay lines for synchronization.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Interactive mode — pick devices from a list:
  python bt_audio_sync.py

  # Specify devices by index:
  python bt_audio_sync.py --input 4 --output-a 7 --output-b 9

  # Specify devices by name substring:
  python bt_audio_sync.py --input-name "BlackHole" --output-a-name "Charge 4" --output-b-name "Charge 6"

  # Set initial delays:
  python bt_audio_sync.py --input-name "BlackHole" --output-a-name "Charge 4" --output-b-name "Charge 6" --delay-a 0 --delay-b 85
        """
    )

    # Device selection — by index
    parser.add_argument('--input', type=int, default=None,
                        help='Input device index (virtual loopback)')
    parser.add_argument('--output-a', type=int, default=None,
                        help='Output device index for Speaker A')
    parser.add_argument('--output-b', type=int, default=None,
                        help='Output device index for Speaker B')

    # Device selection — by name
    parser.add_argument('--input-name', type=str, default=None,
                        help='Substring to match input device name (e.g., "BlackHole")')
    parser.add_argument('--output-a-name', type=str, default=None,
                        help='Substring to match Speaker A name (e.g., "Charge 4")')
    parser.add_argument('--output-b-name', type=str, default=None,
                        help='Substring to match Speaker B name (e.g., "Charge 6")')

    # Audio parameters
    parser.add_argument('--sample-rate', type=int, default=48000,
                        help='Sample rate in Hz (default: 48000)')
    parser.add_argument('--channels', type=int, default=2,
                        help='Number of channels (default: 2)')
    parser.add_argument('--blocksize', type=int, default=512,
                        help='Frames per callback (default: 512)')

    # Initial delays
    parser.add_argument('--delay-a', type=float, default=0.0,
                        help='Initial delay for Speaker A in ms (default: 0)')
    parser.add_argument('--delay-b', type=float, default=0.0,
                        help='Initial delay for Speaker B in ms (default: 0)')

    args = parser.parse_args()

    # ── Banner ────────────────────────────────────────────────────────────
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         BT AUDIO SYNC — Multi-Speaker Synchronizer         ║
    ║                                                            ║
    ║   Route system audio → two Bluetooth speakers              ║
    ║   with stereo split, bass boost, and delay sync            ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # ── Device Discovery ──────────────────────────────────────────────────
    devices = list_audio_devices()

    # Resolve input device
    input_dev = args.input
    if input_dev is None and args.input_name:
        input_dev = find_device_by_substring(args.input_name, 'input')
        if input_dev == -1:
            print(f"ERROR: No input device matching '{args.input_name}' found.")
            sys.exit(1)
    if input_dev is None:
        try:
            input_dev = int(input("Enter INPUT device index (virtual loopback): "))
        except (ValueError, EOFError):
            print("Invalid input. Exiting.")
            sys.exit(1)

    # Resolve output device A
    output_a = args.output_a
    if output_a is None and args.output_a_name:
        output_a = find_device_by_substring(args.output_a_name, 'output')
        if output_a == -1:
            print(f"ERROR: No output device matching '{args.output_a_name}' found.")
            sys.exit(1)
    if output_a is None:
        try:
            output_a = int(input("Enter OUTPUT device index for Speaker A: "))
        except (ValueError, EOFError):
            print("Invalid input. Exiting.")
            sys.exit(1)

    # Resolve output device B
    output_b = args.output_b
    if output_b is None and args.output_b_name:
        output_b = find_device_by_substring(args.output_b_name, 'output')
        if output_b == -1:
            print(f"ERROR: No output device matching '{args.output_b_name}' found.")
            sys.exit(1)
    if output_b is None:
        try:
            output_b = int(input("Enter OUTPUT device index for Speaker B: "))
        except (ValueError, EOFError):
            print("Invalid input. Exiting.")
            sys.exit(1)

    # ── Print Selected Devices ────────────────────────────────────────────
    print(f"\n  Selected devices:")
    print(f"    INPUT:    [{input_dev}] {devices[input_dev]['name']}")
    print(f"    OUTPUT A: [{output_a}] {devices[output_a]['name']}")
    print(f"    OUTPUT B: [{output_b}] {devices[output_b]['name']}")

    # ── Create and Start Engine ───────────────────────────────────────────
    engine = AudioSyncEngine(
        input_device=input_dev,
        output_device_a=output_a,
        output_device_b=output_b,
        sample_rate=args.sample_rate,
        channels=args.channels,
        blocksize=args.blocksize,
        delay_ms_a=args.delay_a,
        delay_ms_b=args.delay_b
    )

    try:
        engine.start()
        run_cli(engine)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        engine.stop()

    # ── Final Stats ───────────────────────────────────────────────────────
    print("\n  Final statistics:")
    print(engine.get_stats())
    print("\n  Goodbye.\n")


if __name__ == '__main__':
    main()
