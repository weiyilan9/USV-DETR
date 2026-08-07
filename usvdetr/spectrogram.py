# -*- coding: utf-8 -*-
"""Audio loading, windowing and spectrogram rendering for USV-DETR.

This module is the input stage of the pipeline. It turns a wav file into
fixed-length RGB spectrogram images that the detector consumes.

Frequency axis convention: row 0 of the returned array is the LOWEST
frequency. Downstream code must therefore display images with
origin="lower" and may map a pixel row directly to frequency.
"""

import numpy as np
import soundfile as sf
from scipy.signal import spectrogram as scipy_spectrogram
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Default parameters. Every function below takes these as keyword arguments,
# so a demo notebook can override any of them without editing this file.
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 250000    # expected wav sample rate, Hz
DEFAULT_LOW_FREQ_HZ = 20000     # lower edge of the displayed frequency band
DEFAULT_HIGH_FREQ_HZ = 100000   # upper edge of the displayed frequency band

DEFAULT_WINDOW_SEC = 2.0        # duration of one spectrogram image
DEFAULT_STEP_SEC = 2.0          # hop between consecutive images

DEFAULT_NPERSEG = 1024          # STFT window length in samples
DEFAULT_NOVERLAP = 768          # STFT overlap in samples
DEFAULT_NFFT = 2048             # FFT length, zero-padded

DEFAULT_GAMMA = 2.0             # contrast exponent, higher darkens background
DEFAULT_VMIN_PCT = 25.0         # percentile mapped to black
DEFAULT_VMAX_PCT = 99.8         # percentile mapped to full brightness

LOG_EPSILON = 1e-10             # guard against log10(0)


# ---------------------------------------------------------------------------
# Colormap
# ---------------------------------------------------------------------------

AUDITION_CMAP = LinearSegmentedColormap.from_list(
    "usvdetr_audition",
    [
        (0.00, "#000000"),
        (0.08, "#04000f"),
        (0.18, "#10001f"),
        (0.32, "#22003a"),
        (0.48, "#3c0052"),
        (0.66, "#6a004f"),
        (0.82, "#b30032"),
        (0.93, "#ff2a2a"),
        (1.00, "#ff6a35"),
    ],
)


# ---------------------------------------------------------------------------
# Audio loading and windowing
# ---------------------------------------------------------------------------

def load_audio(wav_path, expected_sample_rate=None):
    """Read a wav file and downmix it to mono.

    Args:
        wav_path: path to the wav file.
        expected_sample_rate: if given, raise when the file does not match.
            Pass None to accept whatever sample rate the file has.

    Returns:
        (audio, sample_rate) where audio is a 1-D float array.
    """
    audio, sample_rate = sf.read(wav_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if expected_sample_rate is not None and sample_rate != expected_sample_rate:
        raise ValueError(
            "Sample rate mismatch: file has %d Hz, expected %d Hz"
            % (sample_rate, expected_sample_rate)
        )

    return audio, sample_rate


def slice_audio(audio, sample_rate, start_sec=0.0, end_sec=None):
    """Cut a time range out of an audio array.

    Useful for previewing a few seconds of a long recording.

    Args:
        audio: 1-D audio array.
        sample_rate: samples per second.
        start_sec: start of the range in seconds.
        end_sec: end of the range in seconds, or None for end of file.

    Returns:
        The selected part of the array. The caller is responsible for
        remembering that its timestamps start at start_sec.
    """
    start_index = max(0, int(start_sec * sample_rate))

    if end_sec is None:
        end_index = len(audio)
    else:
        end_index = min(len(audio), int(end_sec * sample_rate))

    return audio[start_index:end_index]


def count_segments(
    n_samples,
    sample_rate,
    window_sec=DEFAULT_WINDOW_SEC,
    step_sec=DEFAULT_STEP_SEC,
    pad_tail=True,
):
    """Return how many windows iter_segments will yield.

    Kept in sync with iter_segments so a demo can size a progress bar
    before starting the loop.
    """
    samples_per_window = int(window_sec * sample_rate)
    samples_per_step = int(step_sec * sample_rate)

    if samples_per_window <= 0 or samples_per_step <= 0:
        raise ValueError("window_sec and step_sec must be positive")

    if pad_tail:
        return int(np.ceil(n_samples / samples_per_step))

    if n_samples < samples_per_window:
        return 0

    return (n_samples - samples_per_window) // samples_per_step + 1


def iter_segments(
    audio,
    sample_rate,
    window_sec=DEFAULT_WINDOW_SEC,
    step_sec=DEFAULT_STEP_SEC,
    pad_tail=True,
    time_offset_sec=0.0,
):
    """Cut the audio into fixed-length windows.

    Args:
        audio: 1-D audio array.
        sample_rate: samples per second.
        window_sec: duration of each window.
        step_sec: hop between windows. Equal to window_sec means no overlap.
        pad_tail: zero-pad the last incomplete window instead of dropping it.
        time_offset_sec: added to every reported timestamp. Use it when audio
            is a slice of a longer file so timestamps stay absolute.

    Yields:
        (start_sec, end_sec, segment) for each window.
    """
    samples_per_window = int(window_sec * sample_rate)
    samples_per_step = int(step_sec * sample_rate)

    if samples_per_window <= 0 or samples_per_step <= 0:
        raise ValueError("window_sec and step_sec must be positive")

    position = 0
    while position < len(audio):
        segment = audio[position:position + samples_per_window]

        if len(segment) < samples_per_window:
            if not pad_tail:
                break
            segment = np.pad(segment, (0, samples_per_window - len(segment)))

        start_sec = time_offset_sec + position / sample_rate
        yield start_sec, start_sec + window_sec, segment

        position += samples_per_step


# ---------------------------------------------------------------------------
# Spectrogram computation
# ---------------------------------------------------------------------------

def make_spectrogram(
    segment,
    sample_rate,
    low_freq_hz=DEFAULT_LOW_FREQ_HZ,
    high_freq_hz=DEFAULT_HIGH_FREQ_HZ,
    nperseg=DEFAULT_NPERSEG,
    noverlap=DEFAULT_NOVERLAP,
    nfft=DEFAULT_NFFT,
    gamma=DEFAULT_GAMMA,
    vmin_pct=DEFAULT_VMIN_PCT,
    vmax_pct=DEFAULT_VMAX_PCT,
):
    """Compute a contrast-enhanced spectrogram of one audio window.

    The magnitude is log scaled, clipped to a percentile range computed on
    this window only, rescaled to [0, 1] and raised to gamma. Percentiles are
    per window, so contrast adapts to the local noise floor.

    Args:
        segment: 1-D audio window.
        sample_rate: samples per second.
        low_freq_hz, high_freq_hz: frequency band to keep.
        nperseg, noverlap, nfft: STFT parameters.
        gamma: contrast exponent applied after normalisation.
        vmin_pct, vmax_pct: percentiles mapped to 0 and 1.

    Returns:
        (freqs_hz, times_sec, power_01) where power_01 has shape
        (n_freqs, n_times), values in [0, 1], and row 0 is the lowest
        frequency. times_sec is relative to the start of the segment.
    """
    freqs_hz, times_sec, magnitude = scipy_spectrogram(
        segment,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        mode="magnitude",
    )

    band_mask = (freqs_hz >= low_freq_hz) & (freqs_hz <= high_freq_hz)
    freqs_hz = freqs_hz[band_mask]
    magnitude = magnitude[band_mask, :]

    if magnitude.size == 0:
        raise ValueError(
            "Empty frequency band: no FFT bin between %d and %d Hz"
            % (low_freq_hz, high_freq_hz)
        )

    log_magnitude = np.log10(magnitude + LOG_EPSILON)

    vmin = np.percentile(log_magnitude, vmin_pct)
    vmax = np.percentile(log_magnitude, vmax_pct)
    log_magnitude = np.clip(log_magnitude, vmin, vmax)

    power_01 = (log_magnitude - vmin) / (vmax - vmin + 1e-12)
    power_01 = np.power(power_01, gamma)

    return freqs_hz, times_sec, power_01


def spectrogram_to_rgb(power_01, cmap=None):
    """Colour a normalised spectrogram into an 8-bit RGB image.

    Args:
        power_01: array with values in [0, 1], shape (n_freqs, n_times).
        cmap: any matplotlib colormap. Defaults to AUDITION_CMAP.

    Returns:
        uint8 array of shape (n_freqs, n_times, 3). Row 0 is the lowest
        frequency, so display it with origin="lower".
    """
    if cmap is None:
        cmap = AUDITION_CMAP

    rgba = cmap(power_01)
    return (rgba[..., :3] * 255).astype(np.uint8)


def segment_to_rgb(
    segment,
    sample_rate,
    low_freq_hz=DEFAULT_LOW_FREQ_HZ,
    high_freq_hz=DEFAULT_HIGH_FREQ_HZ,
    nperseg=DEFAULT_NPERSEG,
    noverlap=DEFAULT_NOVERLAP,
    nfft=DEFAULT_NFFT,
    gamma=DEFAULT_GAMMA,
    vmin_pct=DEFAULT_VMIN_PCT,
    vmax_pct=DEFAULT_VMAX_PCT,
    cmap=None,
):
    """Convenience wrapper: audio window in, RGB spectrogram image out.

    Returns:
        uint8 RGB array of shape (n_freqs, n_times, 3).
    """
    _, _, power_01 = make_spectrogram(
        segment,
        sample_rate,
        low_freq_hz=low_freq_hz,
        high_freq_hz=high_freq_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        gamma=gamma,
        vmin_pct=vmin_pct,
        vmax_pct=vmax_pct,
    )
    return spectrogram_to_rgb(power_01, cmap=cmap)
