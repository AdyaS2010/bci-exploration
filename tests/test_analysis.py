"""Tests for effect sizes, the Berger check, and state comparison, using fake feature rows."""

import numpy as np
import pytest

from kestrel import config
from kestrel.analysis import (
    balance_by_label,
    berger_check,
    cohens_d,
    compare_states,
    compare_two,
    linear_trend,
    recording_means,
)


def fake_rows(label, n, alpha_level, rng, recording="r1"):
    """Feature rows whose alpha power is lognormal around `alpha_level` (µV²)."""
    rows = []
    for i in range(n):
        row = {"label": label, "epoch": i, "rejected": False, "recording": recording}
        for ch in config.CHANNELS:
            row[f"alpha_abs_{ch}"] = float(alpha_level * np.exp(rng.normal(0, 0.3)))
        rows.append(row)
    return rows


def test_cohens_d_known_value():
    a = np.array([2.0, 4.0, 6.0])
    b = np.array([0.0, 2.0, 4.0])
    assert cohens_d(a, b) == pytest.approx(1.0)  # means differ by 2, pooled SD is 2


def test_compare_two_ci_contains_d_and_detects_shift():
    rng = np.random.default_rng(0)
    result = compare_two(rng.normal(1, 1, 200), rng.normal(0, 1, 200))
    assert result["d_ci_low"] < result["d"] < result["d_ci_high"]
    assert result["d"] == pytest.approx(1.0, abs=0.25)
    assert result["p"] < 1e-6


def test_berger_check_passes_when_eyes_closed_alpha_is_higher():
    rng = np.random.default_rng(1)
    result = berger_check(
        fake_rows("eyes_open", 50, 10, rng), fake_rows("eyes_closed", 50, 30, rng)
    )
    assert result["passed"]
    assert result["all_channels"]["d"] > 1


def test_berger_check_fails_honestly_when_there_is_no_effect():
    rng = np.random.default_rng(2)
    result = berger_check(
        fake_rows("eyes_open", 50, 10, rng), fake_rows("eyes_closed", 50, 10, rng)
    )
    assert not result["passed"]


def test_berger_check_ignores_rejected_epochs():
    rng = np.random.default_rng(3)
    closed = fake_rows("eyes_closed", 50, 30, rng)
    opened = fake_rows("eyes_open", 50, 10, rng)
    for r in opened[:10]:
        r["rejected"] = True
    assert berger_check(opened, closed)["all_channels"]["n_b"] == 40


def test_balance_by_label_equalises_counts():
    rng = np.random.default_rng(4)
    rows = fake_rows("reading", 80, 10, rng) + fake_rows("math", 30, 10, rng)
    balanced = balance_by_label(rows)
    assert sum(r["label"] == "reading" for r in balanced) == 30
    assert sum(r["label"] == "math" for r in balanced) == 30


def test_compare_states_ranks_separating_feature_first():
    rng = np.random.default_rng(5)
    rows = fake_rows("rest", 60, 30, rng) + fake_rows("math", 60, 10, rng)
    for r in rows:
        r["noise_rel_X"] = float(rng.normal())  # a feature with no real difference
    results = compare_states(rows, ["noise_rel_X", "alpha_abs_TP9"])
    assert results[0]["feature"] == "alpha_abs_TP9"
    assert results[0]["epsilon_sq"] > results[1]["epsilon_sq"]
    assert results[0]["pairwise"]["math vs rest"]["d"] < 0  # math has less alpha


def test_recording_means_groups_by_label_and_recording():
    rng = np.random.default_rng(6)
    rows = fake_rows("rest", 10, 10, rng, "a") + fake_rows("rest", 10, 100, rng, "b")
    means = recording_means(rows, "alpha_abs_AF7")
    assert set(means["rest"]) == {"a", "b"}
    assert means["rest"]["b"] - means["rest"]["a"] == pytest.approx(1.0, abs=0.3)  # log10


def test_linear_trend_recovers_slope():
    x = np.arange(10)
    y = 0.5 * x + np.random.default_rng(7).normal(0, 0.1, 10)
    result = linear_trend(x, y)
    assert result["slope_ci_low"] < 0.5 < result["slope_ci_high"]
