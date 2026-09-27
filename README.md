# Kestrel

*A kestrel holds itself still in mid-air to focus on one thing.*

Kestrel is a low-cost EEG research pipeline built on a consumer Muse 2 headset (four channels). It has two goals:

1. **Self-regulation, proven honestly.** Can I learn to voluntarily raise a specific brain rhythm (upper alpha), and can I show that the change is real? That means a sham-controlled neurofeedback design, artifact checks, and measured learning over sessions, not a placebo or a jaw clench.
2. **State exploration.** Do the things I actually do, like reading, math, coding, music, and rest, leave distinct and repeatable EEG signatures?

The Muse app already gives "calm" and "focus" scores. Kestrel doesn't rebuild those. It builds a transparent pipeline where every number can be traced back to raw data, then uses that pipeline for controlled experiments.

> **Not a medical device.** This is self-experimentation for a student research project. Nothing here makes a diagnostic or clinical claim.

## Status

Semester 1, early phase. What exists:

| Stage | Module | Status |
|---|---|---|
| Acquisition (Muse 2 or synthetic board) | `src/kestrel/acquire.py` | written, tested on synthetic board |
| Preprocessing (1–40 Hz bandpass, 60 Hz notch, epochs, amplitude rejection) | `src/kestrel/preprocess.py` | written, unit-tested |
| Features (band powers, ratios, upper alpha, frontal alpha asymmetry) | `src/kestrel/features.py` | written, unit-tested |
| Validity gate (Berger effect) | `scripts/run_berger_check.py` | **awaiting first real Muse recording** |
| State exploration | `scripts/run_state_explore.py` | written, awaiting data |
| Real-time loop with real/sham toggle | `src/kestrel/neurofeedback.py` | written, tested on synthetic board |

Hypotheses and analysis plans are written before any real data exist:
[docs/preregistration.md](docs/preregistration.md) (neurofeedback) and
[docs/preregistration-state-exploration.md](docs/preregistration-state-exploration.md) (activity signatures).

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate             # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m pytest                   # signal-math tests
python scripts/smoke_test.py       # end-to-end run on BrainFlow's synthetic board, no headset needed
```

With the headset paired, set `USE_SYNTHETIC = False` in `src/kestrel/config.py`, then:

```bash
python scripts/record_session.py --seconds 60 --label eyes_open
python scripts/run_berger_check.py              # guided eyes-open / eyes-closed validity check
```

## Layout

```
src/kestrel/     pipeline modules (config, acquire, preprocess, features, analysis, neurofeedback)
scripts/         command-line entry points
tests/           unit tests for the signal math
notes/           running learning notes
docs/            sprint plans, reflections, preregistration
data/            recordings (git-ignored; raw EEG is never committed)
```

## Limitations, stated up front

- **Four dry electrodes** (TP9, AF7, AF8, TP10). No occipital or central coverage, so alpha is measured far from where it is strongest.
- **Consumer hardware.** Contact quality varies, and the frontal and temporal sites pick up eye and jaw muscle activity. Beta and especially gamma at these sites are often mostly muscle (EMG), not brain.
- **Single subject.** Results describe one person. They don't generalize without replication.
- **Epochs aren't independent.** Overlapping windows from one recording are correlated, so within-session p-values overstate certainty. The session is the real unit of analysis for learning claims.

## Data handling

Recordings stay local in `data/`, which is git-ignored. Each recording carries a pseudonymous subject ID (e.g. `S01`), never a name. That makes consent, de-identification, and deletion requests simple if classmates join later.

## License

MIT, see [LICENSE](LICENSE).
