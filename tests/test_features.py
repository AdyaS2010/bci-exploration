"""Signal-math tests: inject sines at known frequencies and check the features follow.

If any of these fail, every downstream number is suspect.
"""

import numpy as np
import pytest

from kestrel import config
from kestrel.features import (
    band_power,
    band_powers,
    epoch_features,
    feature_table,
    frontal_alpha_asymmetry,
    relative_band_powers,
    welch_psd,
)

FS = 256
SECONDS = 2.0


def sine(freq_hz, amplitude_uv=10.0, seconds=SECONDS, fs=FS, n_channels=4):
    t = np.arange(int(seconds * fs)) / fs
    wave = amplitude_uv * np.sin(2 * np.pi * freq_hz * t)
    return np.tile(wave, (n_channels, 1))


@pytest.mark.parametrize(
    "freq, expected_band",
    [(2.5, "delta"), (6.0, "theta"), (10.0, "alpha"), (20.0, "beta"), (35.0, "gamma")],
)
def test_known_frequency_lands_in_its_band(freq, expected_band):
    freqs, psd = welch_psd(sine(freq), FS)
    powers = band_powers(freqs, psd)
    for ch in range(4):
        winner = max(powers, key=lambda band: powers[band][ch])
        assert winner == expected_band


def test_psd_peak_at_injected_frequency():
    freqs, psd = welch_psd(sine(11.0), FS)
    assert freqs[np.argmax(psd[0])] == pytest.approx(11.0)


def test_band_power_matches_sine_variance():
    # A sine of amplitude A has variance A²/2, and all of it sits near its frequency,
    # so the alpha band should hold about that much power. This checks units and
    # scaling, which is exactly what a NumPy integration change could silently break.
    amplitude = 10.0
    freqs, psd = welch_psd(sine(10.0, amplitude, seconds=8.0), FS)
    alpha = band_power(freqs, psd, config.BANDS["alpha"])
    assert alpha[0] == pytest.approx(amplitude**2 / 2, rel=0.1)


def test_power_scales_with_amplitude_squared():
    freqs, psd_small = welch_psd(sine(10.0, 5.0), FS)
    _, psd_big = welch_psd(sine(10.0, 15.0), FS)
    ratio = band_power(freqs, psd_big, (8, 13)) / band_power(freqs, psd_small, (8, 13))
    assert ratio[0] == pytest.approx(9.0, rel=1e-6)


def test_relative_powers_near_one_and_alpha_dominates():
    rng = np.random.default_rng(0)
    signal = sine(10.0) + rng.normal(0, 1.0, size=(4, int(SECONDS * FS)))
    freqs, psd = welch_psd(signal, FS)
    rel = relative_band_powers(freqs, psd)
    assert rel["alpha"][0] > 0.8
    # Adjacent bands share edge bins, so the sum is slightly above 1, never far off.
    assert sum(r[0] for r in rel.values()) == pytest.approx(1.0, abs=0.15)


def test_upper_alpha_separates_from_lower_alpha():
    upper = epoch_features(sine(11.0), FS)
    lower = epoch_features(sine(8.5), FS)
    assert upper["upper_alpha_abs_AF7"] > 10 * lower["upper_alpha_abs_AF7"]


def test_frontal_alpha_asymmetry_sign():
    assert frontal_alpha_asymmetry(alpha_left=1.0, alpha_right=2.0) > 0
    assert frontal_alpha_asymmetry(alpha_left=2.0, alpha_right=1.0) < 0
    assert frontal_alpha_asymmetry(alpha_left=3.0, alpha_right=3.0) == pytest.approx(0.0)


def test_epoch_features_faa_uses_af7_and_af8():
    signal = sine(10.0)
    af8 = config.CHANNELS.index("AF8")
    signal[af8] *= 2  # four times the power on the right
    row = epoch_features(signal, FS)
    assert row["frontal_alpha_asymmetry"] == pytest.approx(np.log(4.0), rel=1e-6)


def test_feature_table_flags_rejected_epochs_without_dropping_them():
    epochs = np.stack([sine(10.0), sine(10.0), sine(10.0)])
    keep = np.array([True, False, True])
    rows = feature_table(epochs, FS, label="eyes_closed", keep_mask=keep, session=1)
    assert len(rows) == 3
    assert [r["rejected"] for r in rows] == [False, True, False]
    assert all(r["label"] == "eyes_closed" and r["session"] == 1 for r in rows)
