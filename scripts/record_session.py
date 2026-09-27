"""Record one labelled block and save it to data/raw/.

    python scripts/record_session.py --seconds 60 --label eyes_closed
    python scripts/record_session.py --seconds 300 --label reading --note "novel, chapter 3"

Uses the Muse unless config.USE_SYNTHETIC is True (or --synthetic is passed). Prints a
quick contact-quality summary afterwards, so a bad recording is caught immediately.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainflow.board_shim import BoardShim  # noqa: E402

from kestrel import config  # noqa: E402
from kestrel.acquire import load_recording, record_block  # noqa: E402
from kestrel.features import recording_features  # noqa: E402
from kestrel.preprocess import contamination_report  # noqa: E402

# Catch blocks are supposed to be full of artifacts, so don't nag about contact there.
CATCH_LABELS = ("blink", "clench")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--label", required=True, help="block label, e.g. eyes_open, reading, clench")
    p.add_argument("--seconds", type=float, default=60, help="recording length (default 60)")
    p.add_argument("--note", default="", help="free-text context: what, how you felt, etc.")
    p.add_argument("--subject", default=config.DEFAULT_SUBJECT, help="pseudonymous ID, not a name")
    p.add_argument("--synthetic", action="store_true", help="force the synthetic board")
    p.add_argument("--countdown", type=int, default=3, help="seconds before recording starts")
    return p.parse_args()


def main():
    args = parse_args()
    BoardShim.disable_board_logger()
    if args.label not in config.KNOWN_LABELS:
        print(
            f"Note: '{args.label}' is not in config.KNOWN_LABELS. Check for typos, since "
            "a misspelt label silently splits a condition in two."
        )
    use_synthetic = True if args.synthetic else None
    if args.synthetic or config.USE_SYNTHETIC:
        print("Using the SYNTHETIC board, not the Muse.")

    for i in range(args.countdown, 0, -1):
        print(f"Starting '{args.label}' in {i} ...", flush=True)
        time.sleep(1)
    print(f"Recording {args.seconds:g} s (plus {config.SETTLE_SECONDS:g} s settling) ...")
    path = record_block(args.label, args.seconds, args.note, args.subject, use_synthetic)

    eeg, meta = load_recording(path)
    lost = max((meta["expected_samples"] or 0) - meta["n_samples"], 0)
    print(f"Saved {path}")
    print(
        f"  {meta['n_samples']} samples at {meta['sampling_rate']} Hz "
        f"({meta['duration_s']} s); {lost} fewer than expected"
    )
    _, epochs, _ = recording_features(eeg, meta)
    if len(epochs) == 0:
        print("  Too short to epoch; no quality summary.")
        return
    report = contamination_report(epochs, meta["channels"])
    print(
        f"  {report['n_rejected']}/{report['n_epochs']} epochs over the "
        f"{config.REJECT_PEAK_TO_PEAK_UV:g} uV artifact threshold"
    )
    for ch, frac in report["fraction_over_by_channel"].items():
        warn = frac > 0.3 and args.label not in CATCH_LABELS
        flag = "  <- check contact" if warn else ""
        median = report["median_ptp_uv"][ch]
        print(f"    {ch}: median {median:.0f} uV p-p, {frac:.0%} over{flag}")


if __name__ == "__main__":
    main()
