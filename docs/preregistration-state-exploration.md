# Preregistration: do everyday activities leave distinct EEG signatures?

| | |
|---|---|
| Project | Kestrel, state-exploration pillar (single subject, self-experiment) |
| Participant | S01, pseudonymous |
| Hardware | Muse 2 (TP9, AF7, AF8, TP10), BrainFlow; all settings from `src/kestrel/config.py` |
| Companion to | [preregistration.md](preregistration.md) (neurofeedback) |
| Written | 2026-09-27, before any activity data exist |
| Status | **Draft.** It freezes (tag `prereg-states-v1`) once the Berger validity gate passes on the real Muse. |

This pillar is exploratory by nature: the point is to mine my own recordings for patterns. The danger with mining is that with 57 features and a handful of activities, *something* will always look different. So this plan splits the data in two:

- **Discovery (days 1–5):** explore freely, then write down a small number of specific predictions.
- **Confirmation (days 6–10):** test only those predictions, on recordings that didn't exist when the predictions were made.

Only a pattern that holds up in confirmation gets called a signature.

---

## 1. Questions

1. **Do distinct activities leave distinct signatures?** Which features separate which activities, and by how much?
2. **Are the signatures consistent?** Does an activity look the same across days, or does day-to-day variation (electrode fit, sleep, mood) swamp the activity differences?
3. **Are the signatures brain or behavior?** Activities differ in eye movement, typing, and jaw tension. A signature that lives entirely in artifact-sensitive features describes behavior, not a brain state. That's still worth knowing, but it's a different finding.
4. *(Exploratory)* **How do states settle?** Within a block, how long until features stabilise after starting a task?

## 2. Activities

Five activities, each standardised so the comparison is fair. All are done seated in the same chair, at the same screen, with the headset refit once at the start of each day.

| Label | What exactly | Eyes and hands |
|---|---|---|
| `rest` | Look at a fixation cross on the screen, mind wandering allowed, silence | Eyes open, hands still |
| `reading` | Read a novel on screen (same book throughout, continuing where I left off), scroll with one key | Horizontal saccades, minimal typing |
| `math` | Mental arithmetic problems shown on screen (two-digit multiplication), answers typed | Fixation plus typing |
| `coding` | Write Python on a small self-contained exercise, new one each day, no copy-paste | Saccades plus typing |
| `music` | Listen to instrumental music through headphones (same pre-chosen playlist, next track each time) while looking at the fixation cross | Eyes open, hands still |

**Why "conversation" is excluded for now:** speaking drives strong jaw and face muscle activity straight into TP9/TP10. It would separate from everything else for purely muscular reasons. It can be added later as a deliberate artifact-contrast condition.

**Why eyes-open for every activity:** eye closure alone raises alpha massively (the Berger effect). If any activity allowed closed eyes, it would win on alpha for a trivial reason.

## 3. Recording schedule

- **10 recording days**, at most one per day, at a similar time of day (±2 h).
- **Each day:**
  1. Contact check (60 s eyes-open, ≤30% epochs over threshold per channel)
  2. Catch blocks: 20 s blink, 20 s clench
  3. All 5 activities, **exactly 300 s each** with 60 s rest between
- **Order is counterbalanced.** Days 1–5 follow a 5×5 Latin square, so each activity appears once in each position (1st…5th). Days 6–10 use a second, different Latin square. This stops "always first, while fresh" or "always last, while tired" from masquerading as an activity signature.
- **Result:** 10 recordings × 300 s per activity, 5 for discovery and 5 for confirmation.
- **Recording command:** `record_session.py --label <activity> --seconds 300 --note "day N, position P"`. After each block I also note sleepiness (KSS 1–9) and whether I was actually engaged.

## 4. Fair-comparison rules (fixed in advance)

1. **Identical preprocessing** for every recording: `recording_features` (60 Hz notch, 1–40 Hz bandpass, 2 s epochs, 50% overlap, 150 µV rejection). No per-activity tuning.
2. **Matched duration:** every block is 300 s. The first 20 s of each block are dropped as a transition period (epochs with `start_s` < 20). Pooled epoch-level summaries use `balance_by_label`, so every activity contributes the same number of clean epochs.
3. **Per-epoch features** as produced by `features.py`: absolute and relative band powers per channel, upper alpha, theta/beta and alpha/theta ratios, frontal alpha asymmetry. Absolute powers are analysed as log10.
4. **The recording is the unit for claims.** Epoch-level statistics are descriptive only, because overlapping epochs aren't independent. Every inferential test uses one value per recording (the mean over its clean epochs).
5. **Within-day comparisons first.** Electrode fit changes each day, so the primary comparisons are paired within a day (activity A vs activity B on the same day, same fit). Relative power is emphasised for the same reason.
6. **Artifact-sensitive features are flagged, not dropped:** anything at gamma, delta at AF7/AF8 (eye movement), and frontal alpha asymmetry (sensitive to lateral eye movement). A signature carried only by these features is reported as *behavioral* (see §7).
7. **Rejection rates are reported per activity.** If an activity loses much more data than others (for example, over 20 percentage points more), its surviving epochs are a biased sample, and that's stated next to every result involving it.

## 5. Discovery analysis (days 1–5)

Anything goes, as long as it's written down. Specifically:

- Run `run_state_explore.py` on days 1–5. Look at epoch-level epsilon-squared, FDR-adjusted p, pairwise d, and per-recording means.
- Recompute at the recording level. For each feature, take the 5 per-recording means per activity. Rank features by recording-level Kruskal-Wallis epsilon-squared, and check within-day consistency: in how many of the 5 days does the same ordering hold?
- **Then choose at most 3 predictions** (§6 explains the cap). Each is a specific directional claim with a feature, a pair, and a direction, for example "relative theta at AF8 is higher during math than during rest". A prediction qualifies only if the direction holds within-day on at least 4 of the 5 discovery days. Preference goes to features that aren't artifact-sensitive.
- **Commit the predictions** to §9 of this file, dated, **before day 6 is recorded.** Exploration continues afterwards, but nothing added after that commit counts as confirmatory.

## 6. Confirmation analysis (days 6–10)

Only the committed predictions are tested.

- **Test per prediction:** a within-day paired comparison. For each confirmation day, take the difference in recording means between the two activities (A − B). That gives 5 paired differences.
  - Primary statistic: a one-sided paired t-test on those 5 differences, in the predicted direction.
  - Also reported: mean paired difference, paired Cohen's d_z with 95% CI, and the count of days in the predicted direction.
- **Multiple testing:** Holm correction across the (at most 3) predictions at family-wise α = .05.
- **A prediction replicates** only if it survives Holm **and** the direction holds on at least 4 of the 5 days. The day count stops one extreme day from carrying the result.
- **Why a t-test and not a sign test:** with 5 days, a sign test can never go below p = 1/32 ≈ .031. That's already above Holm's first threshold (.025 or lower) whenever there are 2 or more predictions, so nothing could ever be confirmed.
- **Power, honestly:** with 5 pairs and 3 predictions, 80% power needs a very consistent within-day effect (d_z ≈ 1.9, or 2.2 at 5 predictions, which is why predictions are capped at 3). Pairing within a day cancels day-to-day electrode fit, so strong activity contrasts can plausibly reach this, but subtle ones can't. Near-misses (for example, 4/5 days in the predicted direction but not surviving Holm) are reported as "suggestive, needs replication", never as confirmed.

## 7. Consistency and brain-vs-behavior checks (all 10 days)

- **Activity vs day variance.** For each top feature: the share of variance in recording means explained by activity, compared with the share explained by day. If day dominates, the "signature" is mostly day-to-day noise.
- **Artifact profile per activity:** blink rate (AF7/AF8 deflections over 100 µV per minute), gamma at TP9/TP10, and rejection rate, reported side by side with the signature features. If a confirmed signature correlates strongly (|r| > 0.7 across recordings) with an artifact measure, it's reported as possibly behavioral.
- **Catch-block reference:** how much blinks and clenches move each feature. This shows which features artifacts can move at all.

## 8. What a null result looks like, and why it matters

| Pattern | Meaning |
|---|---|
| **No prediction replicates** | At this hardware level, day-to-day variation is as large as any activity difference. Four dry electrodes can't reliably tell these activities apart in one person. That's a useful bound on what consumer-EEG "state detection" can really do. |
| **Only artifact-sensitive features replicate** | The activities are distinguishable, but by how the eyes and muscles behave, not by a brain state. Any consumer product claiming to read "focus" from these sites faces the same issue. |
| **Discovery patterns vanish in confirmation** | The patterns were overfit noise. This is exactly what the split exists to catch, and it's the most common fate of exploratory findings. Reporting it is the honest outcome. |
| **Some brain-plausible features replicate** | A candidate signature, specific to one person and this setup, worth testing with more days and, later, more people. |

A null here is informative because the design rules out the usual excuses: durations are matched, preprocessing is identical, order is counterbalanced, artifacts are measured, and discovery is kept separate from confirmation. If nothing survives, it's because the signal isn't there at this resolution, not because the comparison was unfair.

## 9. Committed predictions and deviations log

Dated entries, newest last. The predictions from §5 go here before day 6 is recorded.

| Date | Entry |
|---|---|
| 2026-09-27 | Draft written. No activity data recorded. Freezes after the Berger gate passes. |
