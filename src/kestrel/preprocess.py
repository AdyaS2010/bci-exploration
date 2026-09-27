"""Filtering, epoching, and amplitude-based artifact rejection.

Artifact strategy, stated plainly: with four dry electrodes, ICA can't reliably
separate eye and muscle sources from brain sources. So the primary defense here is
(1) a peak-to-peak amplitude threshold on each epoch, and (2) deliberate "catch"
recordings (blink, clench) that measure how much each artifact type contaminates the
features and whether the threshold catches it. ICA via MNE may come later as an
extra check, never as the only one.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

from kestrel import config


def bandpass(data, fs, low=config.BANDPASS_HZ[0], high=config.BANDPASS_HZ[1], order=None):
    """Zero-phase Butterworth bandpass along the last axis.

    Zero-phase (forward-backward) filtering avoids shifting events in time, which
    matters when lining epochs up with task blocks. Second-order sections (sos) stay
    numerically stable at a 1 Hz cutoff, where the plain (b, a) form can go unstable.
    """
    order = config.FILTER_ORDER if order is None else order
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def notch(data, fs, freq=config.NOTCH_HZ, quality=config.NOTCH_QUALITY):
    """Remove mains interference at `freq` with a narrow zero-phase notch.

    Skipped when the notch frequency is at or above Nyquist (it can't exist in the data).
    """
    if freq >= fs / 2:
        return data
    b, a = iirnotch(freq, quality, fs=fs)
    return filtfilt(b, a, data, axis=-1)


def clean(data, fs):
    """Standard Kestrel filter chain: notch first, then 1-40 Hz bandpass."""
    return bandpass(notch(data, fs), fs)


def epoch(data, fs, epoch_seconds=config.EPOCH_SECONDS, overlap=config.EPOCH_OVERLAP):
    """Cut a continuous recording into fixed-length, overlapping windows.

    data: array (n_channels, n_samples).
    Returns (epochs with shape (n_epochs, n_channels, n_samples_per_epoch),
             start times in seconds).
    A trailing partial window is dropped rather than padded, since padding would
    add fake zeros to the spectrum.
    """
    size = int(round(epoch_seconds * fs))
    step = int(round(size * (1 - overlap)))
    if step <= 0:
        raise ValueError("overlap must be < 1")
    n_samples = data.shape[-1]
    starts = np.arange(0, n_samples - size + 1, step)
    if len(starts) == 0:
        return np.empty((0, data.shape[0], size)), np.empty(0)
    epochs = np.stack([data[:, s : s + size] for s in starts])
    return epochs, starts / fs


def peak_to_peak(epochs):
    """Max minus min per epoch and channel, shape (n_epochs, n_channels)."""
    return epochs.max(axis=-1) - epochs.min(axis=-1)


def reject_by_amplitude(epochs, threshold_uv=config.REJECT_PEAK_TO_PEAK_UV):
    """Keep-mask: True where every channel stays under the peak-to-peak threshold.

    One bad channel rejects the whole epoch, because features like frontal asymmetry
    combine channels, and a blink on AF7 alone would bias them.
    """
    return (peak_to_peak(epochs) < threshold_uv).all(axis=1)


def contamination_report(epochs, channels=None, threshold_uv=config.REJECT_PEAK_TO_PEAK_UV):
    """Summarise how artifact-heavy a set of epochs is.

    Run this on blink and clench catch blocks: a high rejection rate there means the
    threshold catches those artifacts. Run it on task blocks to report how much data
    each condition lost.
    """
    channels = config.CHANNELS if channels is None else channels
    ptp = peak_to_peak(epochs)
    keep = reject_by_amplitude(epochs, threshold_uv)
    n = len(epochs)
    return {
        "n_epochs": n,
        "n_rejected": int(n - keep.sum()),
        "fraction_rejected": float(1 - keep.mean()) if n else float("nan"),
        "median_ptp_uv": {ch: float(np.median(ptp[:, i])) for i, ch in enumerate(channels)},
        "fraction_over_by_channel": {
            ch: float((ptp[:, i] >= threshold_uv).mean()) for i, ch in enumerate(channels)
        },
    }


def preprocess_recording(data, fs):
    """Full chain for one raw recording: filter, epoch, flag artifacts.

    Filtering happens on the continuous signal, before epoching, so filter edge
    transients only affect the recording's ends, not every epoch.
    Returns (epochs, epoch start times in s, keep mask).
    """
    filtered = clean(data, fs)
    epochs, starts = epoch(filtered, fs)
    keep = reject_by_amplitude(epochs) if len(epochs) else np.zeros(0, dtype=bool)
    return epochs, starts, keep
