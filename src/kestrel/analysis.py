"""Statistics: effect sizes with confidence intervals, the Berger check, state comparison.

A caveat that applies to every epoch-level p-value in this module: epochs from one
recording overlap and are autocorrelated, so they are not independent samples. Treating
them as independent (pseudo-replication) makes p-values look smaller than they should.
Effect sizes are still meaningful descriptions of one recording. Claims about learning
or about a state in general should rest on session-level data (one value per session).
"""

from itertools import combinations

import numpy as np
from scipy import stats

from kestrel import config
from kestrel.features import column

PSEUDOREPLICATION_NOTE = (
    "Epoch-level p-values treat overlapping, autocorrelated epochs as independent, "
    "so they overstate certainty. Read effect sizes first; confirm across sessions."
)


def cohens_d(a, b):
    """Standardised mean difference (mean(a) - mean(b)) / pooled SD."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    return (a.mean() - b.mean()) / np.sqrt(pooled_var)


def cohens_d_ci(d, na, nb, level=0.95):
    """Approximate confidence interval for Cohen's d (Hedges & Olkin normal approximation)."""
    se = np.sqrt((na + nb) / (na * nb) + d**2 / (2 * (na + nb)))
    z = stats.norm.ppf(0.5 + level / 2)
    return d - z * se, d + z * se


def compare_two(a, b, alternative="two-sided"):
    """Welch's t-test plus Cohen's d with a 95% CI, for condition a vs condition b.

    Welch's version doesn't assume equal variances, which rarely hold between, say,
    eyes-closed (big alpha swings) and eyes-open recordings.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)
    d = cohens_d(a, b)
    low, high = cohens_d_ci(d, len(a), len(b))
    return {
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "t": float(t),
        "p": float(p),
        "d": float(d),
        "d_ci_low": float(low),
        "d_ci_high": float(high),
    }


def transform(feature_name, values):
    """Put a feature on the scale we run statistics on.

    Absolute band power is strongly right-skewed (occasional huge epochs), so it's
    compared as log10. Ratios, relative power and asymmetry are left as they are.
    """
    values = np.asarray(values, float)
    if "_abs_" in feature_name:
        return np.log10(values)
    return values


def berger_check(open_rows, closed_rows, channels=None):
    """Validity gate: is alpha higher with eyes closed than eyes open?

    Compares log10 absolute alpha on clean epochs, per channel and averaged over all
    channels. The rig passes if the channel-average effect is in the expected direction
    and the 95% CI of d excludes zero (a one-sided test, because the direction is
    predicted in advance). The Muse has no occipital electrodes, where the Berger
    effect is strongest, so expect the temporal sites (TP9/TP10) to show it best.
    """
    channels = config.CHANNELS if channels is None else channels
    results = {}
    for ch in channels:
        name = f"alpha_abs_{ch}"
        closed = transform(name, column(closed_rows, name))
        opened = transform(name, column(open_rows, name))
        results[ch] = compare_two(closed, opened, alternative="greater")

    def channel_mean(rows):
        # Average log alpha across channels for each epoch.
        return np.mean([np.log10(column(rows, f"alpha_abs_{ch}")) for ch in channels], axis=0)

    overall = compare_two(channel_mean(closed_rows), channel_mean(open_rows), alternative="greater")
    results["all_channels"] = overall
    results["passed"] = bool(overall["d"] > 0 and overall["d_ci_low"] > 0)
    results["note"] = PSEUDOREPLICATION_NOTE
    return results


def balance_by_label(rows, seed=0):
    """Subsample clean epochs so every label contributes the same number of epochs.

    Without this, a state recorded for 20 minutes would dominate one recorded for 5 in
    any pooled statistic. The subsample is random, not "the first N", so it doesn't
    favour the start of each recording, and it's seeded so reruns are identical.
    """
    rng = np.random.default_rng(seed)
    clean_rows = [r for r in rows if not r["rejected"]]
    labels = sorted({r["label"] for r in clean_rows})
    groups = {lab: [r for r in clean_rows if r["label"] == lab] for lab in labels}
    n_min = min(len(g) for g in groups.values())
    balanced = []
    for lab in labels:
        idx = np.sort(rng.choice(len(groups[lab]), size=n_min, replace=False))
        balanced.extend(groups[lab][i] for i in idx)
    return balanced


def recording_means(rows, feature, recording_key="recording"):
    """Mean of one (transformed) feature per recording, grouped by label.

    If recordings of the same activity disagree as much as different activities do,
    the activity doesn't have a stable signature, however good one session looks.
    """
    out = {}
    for r in rows:
        if r["rejected"]:
            continue
        out.setdefault(r["label"], {}).setdefault(r[recording_key], []).append(r[feature])
    return {
        lab: {rec: float(np.mean(transform(feature, vals))) for rec, vals in recs.items()}
        for lab, recs in out.items()
    }


def compare_states(rows, features):
    """For each feature, how well does it separate the labelled states?

    Kruskal-Wallis (rank-based, robust to the outliers EEG always has) tests whether any
    state differs. Epsilon-squared, H / (n - 1), is its effect size: the share of rank
    variance explained by state, from 0 to 1. Pairwise Cohen's d says which states
    differ and in which direction. Results are sorted by effect size, largest first.

    Testing ~57 features at once means some will reach p < .05 by chance alone. p_fdr is
    the Benjamini-Hochberg adjusted p-value across all features tested. Use it, not the
    raw p, to decide whether any feature separates states at all.
    """
    clean_rows = [r for r in rows if not r["rejected"]]
    labels = sorted({r["label"] for r in clean_rows})
    if len(labels) < 2:
        raise ValueError("Need at least two labelled states to compare.")

    results = []
    for feat in features:
        groups = {
            lab: transform(feat, [r[feat] for r in clean_rows if r["label"] == lab])
            for lab in labels
        }
        h, p = stats.kruskal(*groups.values())
        n = sum(len(g) for g in groups.values())
        pairwise = {
            f"{a} vs {b}": compare_two(groups[a], groups[b]) for a, b in combinations(labels, 2)
        }
        results.append(
            {
                "feature": feat,
                "H": float(h),
                "p": float(p),
                "epsilon_sq": float(h / (n - 1)),
                "means": {lab: float(g.mean()) for lab, g in groups.items()},
                "within_sd": {lab: float(g.std(ddof=1)) for lab, g in groups.items()},
                "pairwise": pairwise,
            }
        )
    adjusted = stats.false_discovery_control([r["p"] for r in results])
    for r, p_fdr in zip(results, adjusted, strict=True):
        r["p_fdr"] = float(p_fdr)
    results.sort(key=lambda r: r["epsilon_sq"], reverse=True)
    return results


def linear_trend(x, y, level=0.95):
    """Least-squares slope of y over x, with a confidence interval.

    Used for learning curves: y = mean target power per block or per session,
    x = block or session number. A slope CI that excludes zero is evidence of change.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    fit = stats.linregress(x, y)
    t_crit = stats.t.ppf(0.5 + level / 2, df=len(x) - 2)
    return {
        "slope": float(fit.slope),
        "slope_ci_low": float(fit.slope - t_crit * fit.stderr),
        "slope_ci_high": float(fit.slope + t_crit * fit.stderr),
        "p": float(fit.pvalue),
        "r": float(fit.rvalue),
        "n": len(x),
    }
