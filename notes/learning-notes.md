# Learning notes

My running file. Not polished, just real. Newest entries at the bottom of each section.

## EEG basics

## The bands

| Band | Range in Kestrel (config.py) | What it's associated with |
|---|---|---|
| delta | 1–4 Hz | |
| theta | 4–8 Hz | |
| alpha | 8–13 Hz | |
| upper alpha | 10–12 Hz | the neurofeedback target |
| beta | 13–30 Hz | |
| gamma | 30–40 Hz | at Muse sites, mostly muscle |

## Hardware and artifacts (Muse 2)

- Four dry electrodes: TP9, AF7, AF8, TP10. No occipital coverage.

## The rigor mindset

## Breaking down the Muse app

## Build findings

Things the pipeline taught me while it was being built (details in the commit history).

- **60 Hz leaks through a 40 Hz low-pass.** 60 Hz is only 1.5x the cutoff, so about
  1.4% of a strong mains signal survives. That's why there's a separate notch filter.
- **Live filtering is not offline filtering.** Zero-phase filters look into the "future",
  which live data doesn't have, so the newest samples get distorted. The live loop uses a
  causal filter instead. The smoke test caught this.
- **Pseudo-replication is real.** Two recordings of the *same* synthetic signal gave
  d = -1.58 on one channel when every epoch was treated as independent. Big-looking
  effects inside one session can just be session-to-session drift. Sessions are the unit.
- **Testing many features finds noise.** With 57 features, some hit p < .05 by chance,
  so the state explorer reports FDR-corrected p-values.
