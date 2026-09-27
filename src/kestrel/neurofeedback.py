"""Real-time neurofeedback loop with a real-vs-sham switch built in.

How one update works:
    1. Take the last few seconds of EEG, filter it, and keep the newest window.
    2. If any channel exceeds the artifact threshold, freeze the display and flag it.
    3. Otherwise compute log10 upper-alpha power at the target channels.
    4. Real mode shows that power, z-scored against a resting baseline and squashed
       to 0..1. Sham mode shows a value from a yoked or random trace instead.

The true brain signal is computed and logged in both modes. That's essential: the
manipulation check ("did the target signal actually move?") and the real-vs-sham
comparison both need what the brain did, not just what the screen showed.

Artifacts freeze the display in both modes. If blinking froze the bar only in real
mode, the participant could tell the conditions apart, which would break the blind.
"""

import json
import time
from pathlib import Path

import numpy as np

from kestrel import config
from kestrel.features import band_power, welch_psd
from kestrel.preprocess import clean, reject_by_amplitude

MODES = ("real", "sham")


def target_power(window, fs, channel_idx, band=config.NF_TARGET_BAND):
    """log10 of mean target-band power across the target channels, for one window."""
    freqs, psd = welch_psd(window[channel_idx], fs)
    return float(np.log10(band_power(freqs, psd, band).mean()))


def window_is_artifact(window, threshold_uv=config.REJECT_PEAK_TO_PEAK_UV):
    return not reject_by_amplitude(window[np.newaxis], threshold_uv)[0]


def feedback_from_power(power, baseline_mean, baseline_sd):
    """Map target power to a 0..1 display value.

    z-scoring against the person's own resting baseline makes 0.5 mean "typical for
    you", whatever the day's electrode contact. The logistic squash keeps the display
    bounded without a hard clip, so big changes still register.
    """
    z = (power - baseline_mean) / max(baseline_sd, 1e-6)
    return float(1 / (1 + np.exp(-z)))


class ShamFeedback:
    """A feedback trace that looks like real feedback but isn't tied to this brain.

    Yoked (preferred): replays the display trace from an earlier real session, so the
    sham has realistic statistics. Otherwise randomized: a smooth AR(1) random walk,
    squashed like the real signal, which drifts about as slowly as real alpha feedback.
    """

    def __init__(self, yoked_trace=None, seed=None, smoothness=0.9):
        self.yoked = list(yoked_trace) if yoked_trace else None
        self.rng = np.random.default_rng(seed)
        self.smoothness = smoothness
        self.step = 0
        self.z = 0.0

    def next(self):
        if self.yoked:
            value = self.yoked[self.step % len(self.yoked)]
        else:
            noise = self.rng.normal(0, np.sqrt(1 - self.smoothness**2))
            self.z = self.smoothness * self.z + noise
            value = float(1 / (1 + np.exp(-self.z)))
        self.step += 1
        return value


def choose_blinded_mode(seed=None):
    """Pick real or sham at random, for single-blind self-experiments.

    The caller should write the mode into the session log and not print it, so the
    person training doesn't know the condition until analysis.
    """
    return str(np.random.default_rng(seed).choice(MODES))


class FeedbackState:
    """Everything one session carries between updates."""

    def __init__(self, mode, baseline_mean, baseline_sd, channel_idx, sham=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.baseline_mean = baseline_mean
        self.baseline_sd = baseline_sd
        self.channel_idx = channel_idx
        self.sham = sham if sham is not None else ShamFeedback()
        self.last_shown = 0.5


def feedback_step(window, fs, state):
    """One update: returns a log entry with the true signal and what was shown."""
    artifact = window_is_artifact(window)
    power = None if artifact else target_power(window, fs, state.channel_idx)
    true_value = (
        None if artifact else feedback_from_power(power, state.baseline_mean, state.baseline_sd)
    )

    if artifact:
        shown = state.last_shown
    elif state.mode == "real":
        shown = true_value
    else:
        shown = state.sham.next()
    state.last_shown = shown
    return {"true_power": power, "true_value": true_value, "shown": shown, "artifact": artifact}


def text_bar(value, width=40):
    """Minimal terminal display (a dashboard is a later sprint)."""
    filled = int(round(value * width))
    return "[" + "#" * filled + " " * (width - filled) + "]"


def _latest_window(board, rows, fs, window_s):
    # Filter a longer buffer than we analyse, so filter edge effects fall on samples
    # we then discard rather than on the window we measure.
    buffer_n = int(2 * window_s * fs)
    data = board.get_current_board_data(buffer_n)[rows].astype(float)
    if data.shape[1] < buffer_n:
        return None
    return clean(data, fs)[:, -int(window_s * fs) :]


def measure_baseline(board, rows, fs, channel_idx, seconds, window_s=None, update_s=None):
    """Resting baseline: mean and SD of target power over clean windows, no feedback shown."""
    window_s = config.NF_WINDOW_SECONDS if window_s is None else window_s
    update_s = config.NF_UPDATE_SECONDS if update_s is None else update_s
    powers = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        time.sleep(update_s)
        window = _latest_window(board, rows, fs, window_s)
        if window is not None and not window_is_artifact(window):
            powers.append(target_power(window, fs, channel_idx))
    if len(powers) < 5:
        raise RuntimeError("Too few clean baseline windows; check electrode contact.")
    return float(np.mean(powers)), float(np.std(powers, ddof=1))


def run_session(board, rows, fs, state, seconds, window_s=None, update_s=None, display=None):
    """Run feedback for `seconds`. Returns the per-update log.

    `display` receives the shown value (0..1). By default it draws a terminal bar that
    never reveals the mode.
    """
    window_s = config.NF_WINDOW_SECONDS if window_s is None else window_s
    update_s = config.NF_UPDATE_SECONDS if update_s is None else update_s
    if display is None:

        def display(value):
            print("\r" + text_bar(value), end="", flush=True)

    log = []
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        time.sleep(update_s)
        window = _latest_window(board, rows, fs, window_s)
        if window is None:
            continue
        entry = feedback_step(window, fs, state)
        entry["t"] = round(time.monotonic() - start, 3)
        log.append(entry)
        display(entry["shown"])
    return log


def save_session_log(log, meta, path):
    """Save the session: metadata (including the mode) plus every update."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "updates": log}, indent=2))
    return path
