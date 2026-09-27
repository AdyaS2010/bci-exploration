"""Single source of truth for hardware, signal-processing, and path settings.

Every module reads its constants from here so a choice (a band edge, a filter cutoff,
a rejection threshold) is made once, documented once, and changed in one place.
"""

from pathlib import Path

from brainflow.board_shim import BoardIds

# --- Hardware ---------------------------------------------------------------

# Flip to False once the Muse is paired. Everything downstream is board-agnostic.
USE_SYNTHETIC = True

MUSE_BOARD_ID = BoardIds.MUSE_2_BOARD.value  # native Bluetooth, no BLED112 dongle
SYNTHETIC_BOARD_ID = BoardIds.SYNTHETIC_BOARD.value

# Muse 2 electrode order as BrainFlow returns it. The synthetic board has 16 generic
# channels at 250 Hz; acquire.py takes its first four and relabels them with these
# names so downstream code runs unchanged (the sidecar records the original names).
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
FRONTAL_LEFT = "AF7"
FRONTAL_RIGHT = "AF8"

# Nominal Muse rate. Code always uses the rate stored with each recording instead,
# because the synthetic board runs at 250 Hz.
MUSE_SAMPLING_RATE = 256

# Seconds to let the stream settle before keeping samples. The Muse's first second
# or so after connecting often contains dropouts and a large DC transient.
SETTLE_SECONDS = 2.0

# --- Frequency bands (Hz, inclusive edges) ----------------------------------

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    # At AF7/AF8/TP9/TP10, "gamma" is dominated by eye and jaw muscle (EMG). It's kept
    # as a feature largely as an artifact indicator, not as a claim about cortical gamma.
    "gamma": (30.0, 40.0),
}

# Upper alpha is the neurofeedback target (see docs/preregistration.md). The band is
# fixed here, before any data, rather than tuned to the individual alpha peak after
# seeing results. That rules out one route to fooling ourselves.
UPPER_ALPHA = (10.0, 12.0)

# Denominator for relative power: the whole analysed range after bandpass filtering.
TOTAL_POWER_RANGE = (1.0, 40.0)

# --- Preprocessing ----------------------------------------------------------

# 1 Hz high-pass removes slow electrode drift and sweat artifacts. 40 Hz low-pass
# keeps everything in BANDS while cutting most EMG and all mains harmonics.
BANDPASS_HZ = (1.0, 40.0)
FILTER_ORDER = 4

# US mains is 60 Hz. The 40 Hz low-pass already attenuates it, but dry electrodes with
# poor contact can pick up mains strong enough to leak through, so notch it explicitly.
NOTCH_HZ = 60.0
NOTCH_QUALITY = 30.0

# 2 s epochs give 0.5 Hz resolution if needed and enough cycles of slow theta. The 50%
# overlap doubles the epoch count, but overlapping epochs aren't independent, and
# analysis.py says so wherever it reports a p-value.
EPOCH_SECONDS = 2.0
EPOCH_OVERLAP = 0.5

# Peak-to-peak threshold (µV) on the filtered signal. Clean resting EEG at these sites
# is usually well under 100 µV peak-to-peak. Blinks at AF7/AF8 and jaw clenches at
# TP9/TP10 routinely exceed it. With only four channels, ICA can't cleanly separate
# eye and muscle sources, so this threshold plus the deliberate blink/clench catch
# blocks are the primary artifact defense. Tune it only on catch-block data, never on
# experimental data.
REJECT_PEAK_TO_PEAK_UV = 150.0

# Welch segment length. 1 s segments give 1 Hz resolution, and a 2 s epoch then
# averages three half-overlapping segments, which smooths the PSD estimate.
WELCH_SEGMENT_SECONDS = 1.0

# --- Neurofeedback --------------------------------------------------------------

NF_TARGET_BAND = UPPER_ALPHA
# Temporal sites are the closest the Muse gets to the posterior alpha generators, and
# they're further from the eyes than AF7/AF8, whose low frequencies are dominated by
# blinks and eye movement. Jaw clenches do hit TP9/TP10, which is why the artifact
# freeze and the clench catch block matter.
NF_CHANNELS = ["TP9", "TP10"]
# 2 s windows match the offline epochs, so live and offline power are comparable.
# Updating every 0.25 s feels responsive without the bar jittering on noise.
NF_WINDOW_SECONDS = 2.0
NF_UPDATE_SECONDS = 0.25

# --- Recording labels ---------------------------------------------------------

# Known block labels. Not enforced (new activities are welcome), but record_session.py
# warns on anything else, which catches typos like "eyes_closd" that would silently
# split a condition in two.
KNOWN_LABELS = [
    "eyes_open",
    "eyes_closed",
    "blink",
    "clench",
    "rest",
    "reading",
    "math",
    "coding",
    "music",
    "conversation",
    "baseline",
    "neurofeedback",
]

# Pseudonymous subject ID. Never a real name, so data stays de-identified by default.
DEFAULT_SUBJECT = "S01"

# --- Paths --------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"  # git-ignored
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = DATA_DIR / "results"
