"""Tests for the feedback logic, especially that sham is truly decoupled from the brain."""

import numpy as np
import pytest

from kestrel.neurofeedback import (
    FeedbackState,
    ShamFeedback,
    choose_blinded_mode,
    feedback_from_power,
    feedback_step,
)

FS = 256


def window(alpha_uv, spike=False, seed=0):
    t = np.arange(2 * FS) / FS
    rng = np.random.default_rng(seed)
    w = alpha_uv * np.sin(2 * np.pi * 11 * t) + rng.normal(0, 1.0, size=(4, 2 * FS))
    if spike:
        w[1, 100:120] += 300.0  # blink-sized
    return w


def state(mode, sham=None):
    return FeedbackState(mode, baseline_mean=0.0, baseline_sd=0.5, channel_idx=[0, 3], sham=sham)


def test_feedback_from_power_is_centred_and_monotonic():
    assert feedback_from_power(1.0, 1.0, 0.5) == pytest.approx(0.5)
    assert feedback_from_power(1.5, 1.0, 0.5) > feedback_from_power(1.0, 1.0, 0.5)


def test_real_mode_follows_alpha():
    s = state("real")
    low = feedback_step(window(1.0), FS, s)["shown"]
    high = feedback_step(window(10.0), FS, s)["shown"]
    assert high > low


def test_sham_ignores_the_brain_but_still_logs_it():
    # Two sham sessions with the same seed, fed very different EEG, must show exactly
    # the same trace, while the logged true power must differ.
    a = state("sham", ShamFeedback(seed=42))
    b = state("sham", ShamFeedback(seed=42))
    log_a = [feedback_step(window(1.0, seed=i), FS, a) for i in range(20)]
    log_b = [feedback_step(window(10.0, seed=i), FS, b) for i in range(20)]
    assert [e["shown"] for e in log_a] == [e["shown"] for e in log_b]
    assert log_b[0]["true_power"] > log_a[0]["true_power"]


def test_yoked_sham_replays_trace():
    sham = ShamFeedback(yoked_trace=[0.1, 0.9, 0.4])
    assert [sham.next() for _ in range(4)] == [0.1, 0.9, 0.4, 0.1]


def test_random_sham_stays_in_range():
    sham = ShamFeedback(seed=1)
    values = [sham.next() for _ in range(500)]
    assert 0 < min(values) and max(values) < 1


@pytest.mark.parametrize("mode", ["real", "sham"])
def test_artifact_freezes_display_in_both_modes(mode):
    s = state(mode, ShamFeedback(seed=3))
    before = feedback_step(window(5.0), FS, s)["shown"]
    entry = feedback_step(window(5.0, spike=True), FS, s)
    assert entry["artifact"] and entry["true_power"] is None
    assert entry["shown"] == before


def test_blinded_mode_is_valid_and_reproducible():
    assert choose_blinded_mode(seed=7) in ("real", "sham")
    assert choose_blinded_mode(seed=7) == choose_blinded_mode(seed=7)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        state("placebo")
