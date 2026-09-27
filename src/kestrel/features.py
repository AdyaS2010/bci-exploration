"""Spectral features per epoch: band powers, ratios, upper alpha, frontal alpha asymmetry.

The output is a tidy table: a list of dicts, one row per epoch, each carrying its label.
That's easy to read, easy to write to CSV, and needs no extra dependency (no pandas).
"""

import csv
from pathlib import Path

import numpy as np
from scipy.signal import welch

from kestrel import config

# Stops log() and ratios from blowing up on a flat or disconnected channel. Far below any
# real EEG power (µV²/Hz), so it never changes a real value in a meaningful way.
_EPS = 1e-12


def welch_psd(epoch, fs, segment_seconds=config.WELCH_SEGMENT_SECONDS):
    """Power spectral density of one epoch.

    epoch: array (n_channels, n_samples) in µV.
    Returns (freqs [Hz], psd [µV²/Hz] with shape (n_channels, n_freqs)).
    """
    nperseg = min(int(round(segment_seconds * fs)), epoch.shape[-1])
    # Hann window with 50% overlap is Welch's standard choice. It trades a little
    # frequency resolution for much lower variance than a single periodogram.
    freqs, psd = welch(epoch, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    return freqs, psd


def band_power(freqs, psd, band):
    """Absolute power in a band (µV²): integrate the PSD between the band edges.

    Edges are inclusive, so neighbouring bands share their boundary bin. At 1 Hz
    resolution this makes an alpha band of 8-13 Hz span exactly 5 Hz, as named.
    """
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if mask.sum() < 2:
        raise ValueError(f"Band {band} has fewer than 2 frequency bins; epoch too short?")
    return np.trapezoid(psd[..., mask], freqs[mask], axis=-1)


def band_powers(freqs, psd, bands=None):
    """Absolute power for every band. Returns {band_name: array (n_channels,)}."""
    bands = config.BANDS if bands is None else bands
    return {name: band_power(freqs, psd, edges) for name, edges in bands.items()}


def relative_band_powers(freqs, psd, bands=None, total_range=config.TOTAL_POWER_RANGE):
    """Each band's share of total 1-40 Hz power.

    Relative power cancels out overall gain differences (like a looser electrode on
    one day), which makes it fairer to compare across sessions than absolute power.
    """
    total = band_power(freqs, psd, total_range) + _EPS
    return {name: power / total for name, power in band_powers(freqs, psd, bands).items()}


def frontal_alpha_asymmetry(alpha_left, alpha_right):
    """Frontal alpha asymmetry: ln(right alpha) - ln(left alpha), here ln(AF8) - ln(AF7).

    Positive means more alpha on the right. Because alpha is read as reduced cortical
    activity, positive FAA is conventionally read as relatively greater left-frontal
    activity. AF7/AF8 sit near the eyes, so this is sensitive to lateral eye movements.
    """
    return np.log(alpha_right + _EPS) - np.log(alpha_left + _EPS)


def epoch_features(epoch, fs, channels=None):
    """All features for one epoch, as a flat dict of floats.

    Column names look like "alpha_abs_AF7" or "theta_rel_TP9". Channel-level features
    are kept separate on purpose: averaging frontal and temporal sites would blend eye
    artifacts (frontal) with jaw artifacts (temporal) and hide both.
    """
    channels = config.CHANNELS if channels is None else channels
    freqs, psd = welch_psd(epoch, fs)

    absolute = band_powers(freqs, psd)
    absolute["upper_alpha"] = band_power(freqs, psd, config.UPPER_ALPHA)
    total = band_power(freqs, psd, config.TOTAL_POWER_RANGE) + _EPS

    row = {}
    for i, ch in enumerate(channels):
        for name, power in absolute.items():
            row[f"{name}_abs_{ch}"] = float(power[i])
            row[f"{name}_rel_{ch}"] = float(power[i] / total[i])
        # Theta/beta is the classic attention ratio; alpha/theta tracks relaxed
        # wakefulness vs drowsiness. Both are exploratory here, not validated markers.
        row[f"theta_beta_ratio_{ch}"] = float(absolute["theta"][i] / (absolute["beta"][i] + _EPS))
        row[f"alpha_theta_ratio_{ch}"] = float(absolute["alpha"][i] / (absolute["theta"][i] + _EPS))

    left = channels.index(config.FRONTAL_LEFT)
    right = channels.index(config.FRONTAL_RIGHT)
    row["frontal_alpha_asymmetry"] = float(
        frontal_alpha_asymmetry(absolute["alpha"][left], absolute["alpha"][right])
    )
    return row


def feature_table(epochs, fs, label, keep_mask=None, channels=None, epoch_starts=None, **meta):
    """Build the tidy per-epoch table for one recording.

    epochs: array (n_epochs, n_channels, n_samples).
    keep_mask: boolean array from preprocess.reject_by_amplitude. Rejected epochs stay in
        the table, flagged, rather than silently vanishing, so we can always report how
        many were dropped and why.
    meta: extra columns copied onto every row (e.g. recording="...", session=3).
    """
    rows = []
    for i, epoch in enumerate(epochs):
        row = {"label": label, "epoch": i}
        if epoch_starts is not None:
            row["start_s"] = float(epoch_starts[i])
        row["rejected"] = bool(keep_mask is not None and not keep_mask[i])
        row.update(meta)
        row.update(epoch_features(epoch, fs, channels))
        rows.append(row)
    return rows


def column(rows, name, include_rejected=False):
    """Pull one feature out of a table as a NumPy array (clean epochs only by default)."""
    return np.array([r[name] for r in rows if include_rejected or not r["rejected"]])


def write_csv(rows, path):
    """Save a feature table to CSV so it can be opened in a spreadsheet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
