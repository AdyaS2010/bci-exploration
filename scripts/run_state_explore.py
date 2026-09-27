"""Compare EEG signatures across labelled activity recordings.

    python scripts/run_state_explore.py
    python scripts/run_state_explore.py --labels reading math rest --top 15

Every recording goes through identical preprocessing and per-epoch features. Labels are
balanced to equal epoch counts, then each feature is ranked by how well it separates the
states (Kruskal-Wallis epsilon-squared). For the top features, it also shows whether
each state's value repeats across separate recordings, because a "signature" seen in
only one session isn't a signature yet.
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

from kestrel import config  # noqa: E402
from kestrel.acquire import list_recordings, load_recording  # noqa: E402
from kestrel.analysis import (  # noqa: E402
    PSEUDOREPLICATION_NOTE,
    balance_by_label,
    compare_states,
    recording_means,
    transform,
)
from kestrel.features import recording_features, write_csv  # noqa: E402

ACTIVITY_LABELS = ["rest", "reading", "math", "coding", "music", "conversation"]
NON_FEATURE_COLUMNS = {"label", "epoch", "start_s", "rejected", "recording", "subject"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--labels", nargs="+", default=ACTIVITY_LABELS, help="states to compare")
    p.add_argument("--subject", default=config.DEFAULT_SUBJECT)
    p.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    p.add_argument("--top", type=int, default=10, help="how many top features to print")
    p.add_argument("--seed", type=int, default=0, help="seed for epoch balancing")
    return p.parse_args()


def load_all(paths):
    rows, quality = [], {}
    for path in paths:
        eeg, meta = load_recording(path)
        rec_rows, _, keep = recording_features(eeg, meta, recording=path.stem)
        rows.extend(rec_rows)
        q = quality.setdefault(meta["label"], {"recordings": 0, "epochs": 0, "rejected": 0})
        q["recordings"] += 1
        q["epochs"] += len(keep)
        q["rejected"] += int((~keep).sum())
    return rows, quality


def plot_top(rows, results, labels, path, n=6):
    """Distribution of each top feature per state. States sit on the x-axis, so color
    isn't needed for identity and the boxes stay neutral."""
    top = results[:n]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, res in zip(axes.flat, top, strict=False):
        feat = res["feature"]
        data = [transform(feat, [r[feat] for r in rows if r["label"] == lab]) for lab in labels]
        ax.boxplot(
            data,
            tick_labels=labels,
            showfliers=False,
            medianprops={"color": "#2a78d6", "linewidth": 2},
            boxprops={"color": "#52514e"},
            whiskerprops={"color": "#52514e"},
            capprops={"color": "#52514e"},
        )
        scale = "log10 " if "_abs_" in feat else ""
        ax.set_title(f"{scale}{feat}\nε² = {res['epsilon_sq']:.2f}", loc="left", fontsize=9)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    for ax in list(axes.flat)[len(top) :]:
        ax.set_visible(False)
    fig.suptitle("Top state-separating features (balanced clean epochs)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    paths = list_recordings(args.raw_dir, labels=args.labels, subject=args.subject)
    if not paths:
        sys.exit(f"No recordings for {args.labels} in {args.raw_dir}.")
    rows, quality = load_all(paths)
    labels = sorted(quality)
    if len(labels) < 2:
        sys.exit(f"Only found {labels}; need at least two states to compare.")

    print("Data per state:")
    for lab in labels:
        q = quality[lab]
        print(
            f"  {lab:<14}{q['recordings']} recording(s), {q['epochs']} epochs, "
            f"{q['rejected']} rejected ({q['rejected'] / max(q['epochs'], 1):.0%})"
        )
        if q["recordings"] < 2:
            print("                 only one recording: can't tell signature from session noise")

    balanced = balance_by_label(rows, seed=args.seed)
    n_each = len(balanced) // len(labels)
    print(f"\nBalanced to {n_each} clean epochs per state.")
    features = [k for k in rows[0] if k not in NON_FEATURE_COLUMNS]
    results = compare_states(balanced, features)

    print(f"\nTop {args.top} features by epsilon-squared (share of rank variance explained).")
    print(f"p_fdr is corrected for testing {len(features)} features; trust it over raw p.")
    for res in results[: args.top]:
        best_pair, best = max(res["pairwise"].items(), key=lambda kv: abs(kv[1]["d"]))
        print(
            f"  {res['feature']:<28} eps^2={res['epsilon_sq']:.2f}  p={res['p']:.2g}  "
            f"p_fdr={res['p_fdr']:.2g}  largest: {best_pair} d={best['d']:.2f} "
            f"[{best['d_ci_low']:.2f}, {best['d_ci_high']:.2f}]"
        )
        for lab, recs in recording_means(rows, res["feature"]).items():
            values = ", ".join(f"{v:.3g}" for v in recs.values())
            print(f"      {lab:<12} per-recording means: {values}")
    n_sig = sum(r["p_fdr"] < 0.05 for r in results)
    print(f"\n{n_sig} of {len(features)} features separate the states at p_fdr < .05.")
    print(f"Caveat: {PSEUDOREPLICATION_NOTE}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = write_csv(rows, config.RESULTS_DIR / f"state_features_{stamp}.csv")
    json_path = config.RESULTS_DIR / f"state_compare_{stamp}.json"
    json_path.write_text(json.dumps({"quality": quality, "results": results}, indent=2))
    png = config.RESULTS_DIR / f"state_compare_{stamp}.png"
    plot_top(balanced, results, labels, png)
    print(f"Saved {csv_path.name}, {json_path.name}, {png.name} in {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
