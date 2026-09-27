"""Tests for filtering, epoching, and amplitude rejection."""

import numpy as np
import pytest

from kestrel.preprocess import (
    bandpass,
    clean,
    contamination_report,
    epoch,
    notch,
    preprocess_recording,
    reject_by_amplitude,
)

FS = 256


def tone(freq_hz, seconds=10.0, amplitude=10.0, fs=FS):
    t = np.arange(int(seconds * fs)) / fs
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def rms(x):
    return np.sqrt(np.mean(x**2))


def middle(x, fs=FS, trim_s=1.0):
    """Ignore filter edge transients when judging steady-state behaviour."""
    n = int(trim_s * fs)
    return x[..., n:-n]


def test_bandpass_keeps_alpha_and_removes_drift():
    alpha = tone(10.0)
    drift = tone(0.1, amplitude=100.0)
    out = bandpass(alpha + drift, FS)
    residual = middle(out - alpha)
    assert rms(residual) < 0.02 * rms(middle(alpha))


def test_bandpass_alone_leaves_some_mains_so_notch_is_needed():
    # 60 Hz is only 1.5x the 40 Hz cutoff, so the low-pass removes most but not all of
    # a strong mains signal. This test documents why clean() adds the notch.
    mains = tone(60.0, amplitude=50.0)
    leaked = rms(middle(bandpass(mains, FS)))
    assert 0.001 * rms(mains) < leaked < 0.05 * rms(mains)
    assert rms(middle(clean(mains, FS))) < 0.2 * leaked


def test_clean_keeps_alpha_and_removes_drift_and_mains():
    alpha = tone(10.0)
    drift = tone(0.1, amplitude=100.0)
    mains = tone(60.0, amplitude=50.0)
    out = clean(alpha + drift + mains, FS)
    # What survives should be almost exactly the 10 Hz component.
    residual = middle(out - alpha)
    assert rms(residual) < 0.02 * rms(middle(alpha))


def test_notch_removes_60hz_but_not_10hz():
    out_mains = notch(tone(60.0), FS)
    out_alpha = notch(tone(10.0), FS)
    assert rms(middle(out_mains)) < 0.05 * rms(tone(60.0))
    assert rms(middle(out_alpha)) == pytest.approx(rms(tone(10.0)), rel=0.02)


def test_notch_skipped_above_nyquist():
    x = tone(10.0, fs=100)
    assert np.array_equal(notch(x, 100, freq=60.0), x)


def test_clean_accepts_multichannel_input():
    data = np.vstack([tone(10.0)] * 4)
    assert clean(data, FS).shape == data.shape


def test_epoch_count_and_shape_with_overlap():
    data = np.zeros((4, 10 * FS))
    epochs, starts = epoch(data, FS, epoch_seconds=2.0, overlap=0.5)
    # 2 s windows every 1 s across 10 s: starts at 0..8 -> 9 epochs.
    assert epochs.shape == (9, 4, 2 * FS)
    assert starts[0] == 0.0 and starts[-1] == 8.0


def test_epoch_contents_line_up_with_source():
    data = np.arange(4 * 6 * FS, dtype=float).reshape(4, 6 * FS)
    epochs, _ = epoch(data, FS, epoch_seconds=2.0, overlap=0.5)
    assert np.array_equal(epochs[1], data[:, FS : 3 * FS])


def test_epoch_too_short_returns_empty():
    epochs, starts = epoch(np.zeros((4, FS)), FS, epoch_seconds=2.0)
    assert epochs.shape[0] == 0 and starts.size == 0


def test_amplitude_rejection_flags_blink_like_spike():
    rng = np.random.default_rng(1)
    epochs = rng.normal(0, 5.0, size=(5, 4, 2 * FS))  # ~30 µV peak-to-peak
    epochs[2, 1, 100:130] += 200.0  # a blink-sized deflection on AF7 only
    keep = reject_by_amplitude(epochs, threshold_uv=150.0)
    assert keep.tolist() == [True, True, False, True, True]


def test_contamination_report_counts():
    rng = np.random.default_rng(2)
    epochs = rng.normal(0, 5.0, size=(4, 4, 2 * FS))
    epochs[:2, 3, :50] += 300.0  # clench-like burst on TP10 in half the epochs
    report = contamination_report(epochs, threshold_uv=150.0)
    assert report["n_rejected"] == 2
    assert report["fraction_rejected"] == 0.5
    assert report["fraction_over_by_channel"]["TP10"] == 0.5
    assert report["fraction_over_by_channel"]["AF7"] == 0.0


def test_preprocess_recording_end_to_end():
    data = np.vstack([tone(10.0, seconds=20.0)] * 4) + 500.0  # large DC offset, like a Muse
    epochs, starts, keep = preprocess_recording(data, FS)
    assert len(epochs) == len(starts) == len(keep) == 19
    # The DC offset must be gone, or every epoch would fail the threshold.
    assert keep[1:-1].all()
