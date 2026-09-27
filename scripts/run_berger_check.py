"""Validity gate: does alpha rise when the eyes close (the Berger effect)?

Guided recording (sit still, relax your jaw, look at one spot when eyes are open):
    python scripts/run_berger_check.py --seconds 60

Or analyse recordings you already made:
    python scripts/run_berger_check.py --open data/raw/S01_..._eyes_open.npy \
                                       --closed data/raw/S01_..._eyes_closed.npy

PASS means the channel-averaged Cohen's d (closed minus open, log10 alpha) is positive
and its 95% CI excludes zero. If this fails on the real Muse, fix the rig (fit, contact,
artifacts) before trusting anything downstream.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from brainflow.board_shim import BoardShim  # noqa: E402

from kestrel import config  # noqa: E402
from kestrel.acquire import load_recording, record_block  # noqa: E402
from kestrel.analysis import berger_check  # noqa: E402
from kestrel.features import recording_features, welch_psd  # noqa: E402
from kestrel.preprocess import contamination_report  # noqa: E402

OPEN_COLOR, CLOSED_COLOR = "#2a78d6", "#eb6834"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--seconds", type=float, default=60, help="length of each block (default 60)")
    p.add_argument("--open", type=Path, help="existing eyes-open recording (.npy)")
    p.add_argument("--closed", type=Path, help="existing eyes-closed recording (.npy)")
    p.add_argument("--synthetic", action="store_true", help="force the synthetic board")
    return p.parse_args()


def guided_recording(seconds, use_synthetic):
    paths = {}
    instructions = [
        ("eyes_open", "Eyes OPEN. Soft gaze on one spot, blink normally, jaw relaxed."),
        ("eyes_closed", "Eyes CLOSED. Stay awake and still, jaw relaxed."),
    ]
    for label, instruction in instructions:
        input(f"\n{instruction}\nPress Enter to start {seconds:g} s ...")
        paths[label] = record_block(
            label, seconds, note="berger check", use_synthetic=use_synthetic
        )
        print(f"Saved {paths[label].name}")
    return paths["eyes_open"], paths["eyes_closed"]


def mean_psd(epochs, keep, fs):
    usable = epochs[keep] if keep.any() else epochs
    freqs = welch_psd(usable[0], fs)[0]
    return freqs, np.mean([welch_psd(e, fs)[1] for e in usable], axis=0)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)


def plot(open_psd, closed_psd, channels, result, path):
    """One small panel per channel: mean PSD eyes open vs closed, alpha band shaded."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for i, (ax, ch) in enumerate(zip(axes.flat, channels, strict=True)):
        for (freqs, psd), color, name in [
            (open_psd, OPEN_COLOR, "eyes open"),
            (closed_psd, CLOSED_COLOR, "eyes closed"),
        ]:
            shown = (freqs >= config.BANDPASS_HZ[0]) & (freqs <= config.BANDPASS_HZ[1])
            ax.semilogy(freqs[shown], psd[i][shown], color=color, linewidth=2, label=name)
        ax.axvspan(*config.BANDS["alpha"], color="#888888", alpha=0.12, linewidth=0)
        r = result[ch]
        title = f"{ch}   d = {r['d']:.2f} [{r['d_ci_low']:.2f}, {r['d_ci_high']:.2f}]"
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlim(*config.BANDPASS_HZ)
        style(ax)
        if i >= 2:
            ax.set_xlabel("Frequency (Hz)")
        if i % 2 == 0:
            ax.set_ylabel("Power (µV²/Hz)")
    axes.flat[0].legend(frameon=False)
    verdict = "PASS" if result["passed"] else "FAIL"
    overall = result["all_channels"]
    fig.suptitle(
        f"Berger check: {verdict}   all-channel d = {overall['d']:.2f} "
        f"[{overall['d_ci_low']:.2f}, {overall['d_ci_high']:.2f}]   (shaded: alpha 8-13 Hz)",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def print_table(result, channels):
    print(f"\n{'channel':<14}{'open':>8}{'closed':>8}{'ratio':>7}{'t':>8}{'p':>10}{'d':>7}  95% CI")
    for key in [*channels, "all_channels"]:
        r = result[key]
        # mean_a is eyes closed, mean_b eyes open, both mean log10 alpha power.
        ratio = 10 ** (r["mean_a"] - r["mean_b"])
        print(
            f"{key:<14}{r['mean_b']:>8.2f}{r['mean_a']:>8.2f}{ratio:>7.2f}{r['t']:>8.2f}"
            f"{r['p']:>10.2g}{r['d']:>7.2f}  [{r['d_ci_low']:.2f}, {r['d_ci_high']:.2f}]"
        )
    print("open/closed: mean log10 alpha power; ratio: closed/open alpha power; p one-sided")


def main():
    args = parse_args()
    BoardShim.disable_board_logger()
    use_synthetic = True if args.synthetic else None
    if args.open and args.closed:
        open_path, closed_path = args.open, args.closed
    elif args.open or args.closed:
        sys.exit("Give both --open and --closed, or neither to record now.")
    else:
        open_path, closed_path = guided_recording(args.seconds, use_synthetic)

    loaded = {}
    for name, path in [("open", open_path), ("closed", closed_path)]:
        eeg, meta = load_recording(path)
        rows, epochs, keep = recording_features(eeg, meta, recording=Path(path).stem)
        loaded[name] = {"rows": rows, "epochs": epochs, "keep": keep, "meta": meta}
        report = contamination_report(epochs, meta["channels"])
        print(f"eyes {name}: {len(epochs)} epochs, {report['n_rejected']} rejected as artifacts")
        if meta["board"] == "synthetic":
            print("  (synthetic board: it has no eyes, so FAIL is the correct answer here)")

    meta = loaded["open"]["meta"]
    channels, fs = meta["channels"], meta["sampling_rate"]
    result = berger_check(loaded["open"]["rows"], loaded["closed"]["rows"], channels)
    print_table(result, channels)
    print(f"\nRESULT: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"Caveat: {result['note']}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    png = config.RESULTS_DIR / f"berger_{stamp}.png"
    psds = [mean_psd(loaded[k]["epochs"], loaded[k]["keep"], fs) for k in ("open", "closed")]
    plot(*psds, channels, result, png)
    out = {"open": str(open_path), "closed": str(closed_path), "result": result}
    (config.RESULTS_DIR / f"berger_{stamp}.json").write_text(json.dumps(out, indent=2))
    print(f"Plot and numbers saved to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
