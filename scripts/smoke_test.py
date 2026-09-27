"""End-to-end check on BrainFlow's synthetic board. No headset needed.

Runs every stage once: acquire -> save -> preprocess -> features -> neurofeedback
(real and sham), and saves a PSD plot. If this passes, the plumbing works. It says
nothing about whether real EEG is trustworthy; that's run_berger_check.py's job.

    python scripts/smoke_test.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from brainflow.board_shim import BoardShim  # noqa: E402

from kestrel import config  # noqa: E402
from kestrel.acquire import (  # noqa: E402
    connect,
    eeg_layout,
    load_recording,
    record,
    save_recording,
)
from kestrel.features import recording_features, welch_psd  # noqa: E402
from kestrel.neurofeedback import (  # noqa: E402
    FeedbackState,
    ShamFeedback,
    measure_baseline,
    run_session,
)
from kestrel.preprocess import contamination_report  # noqa: E402

# Kept apart from data/raw so synthetic data never mixes into real analyses.
SMOKE_DIR = config.SYNTHETIC_DIR / "smoke"
CHANNEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def plot_psd(epochs, keep, fs, channels, path):
    usable = epochs[keep] if keep.any() else epochs  # plot something even if all rejected
    freqs = welch_psd(usable[0], fs)[0]
    psds = np.mean([welch_psd(e, fs)[1] for e in usable], axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, ch in enumerate(channels):
        shown = (freqs >= config.BANDPASS_HZ[0]) & (freqs <= config.BANDPASS_HZ[1])
        ax.semilogy(freqs[shown], psds[i][shown], color=CHANNEL_COLORS[i], linewidth=2, label=ch)
    ax.axvspan(*config.BANDS["alpha"], color="#888888", alpha=0.12, linewidth=0)
    ax.text(10.5, ax.get_ylim()[1], "alpha", ha="center", va="top", color="#52514e")
    ax.set_xlim(*config.BANDPASS_HZ)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (µV²/Hz, log scale)")
    ax.set_title("Smoke test: mean PSD per channel (synthetic board)", loc="left")
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    BoardShim.disable_board_logger()
    board = connect(use_synthetic=True)
    bid = board.get_board_id()
    fs = BoardShim.get_sampling_rate(bid)
    rows_idx, channels, source_names = eeg_layout(bid)
    print(f"Board: synthetic (id {bid}), {fs} Hz")
    print(f"EEG rows {rows_idx}: {source_names} relabelled as {channels}")

    try:
        print("Recording 10 s ...")
        eeg, fs = record(board, 10)
        path = save_recording(
            eeg,
            fs,
            "smoke_test",
            note="synthetic smoke test",
            board_name="synthetic",
            source_channel_names=source_names,
            requested_seconds=10,
            out_dir=SMOKE_DIR,
        )
        eeg, meta = load_recording(path)
        print(
            f"Saved {path.name}: {meta['n_samples']} samples (expected {meta['expected_samples']})"
        )

        rows, epochs, keep = recording_features(eeg, meta, recording=path.stem)
        report = contamination_report(epochs, channels)
        print(f"Epochs: {report['n_epochs']}, rejected {report['n_rejected']}")
        first = next(r for r in rows if not r["rejected"])
        print(
            "First clean epoch, relative alpha: "
            + ", ".join(f"{ch} {first[f'alpha_rel_{ch}']:.2f}" for ch in channels)
        )
        n_features = sum(
            k not in ("label", "epoch", "start_s", "rejected", "recording", "subject")
            for k in first
        )
        assert len(rows) == report["n_epochs"] > 0
        assert all(np.isfinite(v) for v in first.values() if isinstance(v, float))

        png = SMOKE_DIR / "smoke_test_output.png"
        plot_psd(epochs, keep, fs, channels, png)
        print(f"Features per epoch: {n_features}. Plot: {png}")

        print("Neurofeedback: 3 s baseline, then 3 s real and 3 s sham ...")
        nf_idx = [channels.index(ch) for ch in config.NF_CHANNELS]
        board.start_stream()
        time.sleep(config.NF_WINDOW_SECONDS * 2 + 0.5)  # fill the rolling buffer
        mean, sd = measure_baseline(board, rows_idx, fs, nf_idx, seconds=3)
        for mode in ("real", "sham"):
            state = FeedbackState(mode, mean, sd, nf_idx, sham=ShamFeedback(seed=0))
            log = run_session(board, rows_idx, fs, state, seconds=3, display=lambda v: None)
            clean_n = sum(not e["artifact"] for e in log)
            assert log and all(0 <= e["shown"] <= 1 for e in log)
            print(f"  {mode}: {len(log)} updates, {clean_n} clean")
        board.stop_stream()
    finally:
        board.release_session()

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
