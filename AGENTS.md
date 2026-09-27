# Kestrel — project brief and build spec

*For the coding agent. Read this first, keep it open, and treat it as the source of truth for what we're building and how.*

A kestrel holds itself still in mid-air to focus on one thing. That's the project in a word: understanding a mind's states, and learning to hold one on purpose.

---

## What this project actually is

I'm building a low-cost EEG system that reads brain activity from a consumer headset (a Muse 2, four channels) and lets me both study my cognitive states and train myself to steer them. Reading a real, trustworthy signal is the floor, not the point. The Muse app already spits out tidy focus and calm scores, and we are not rebuilding that. What the app never does is the actual interesting work, and that's where Kestrel lives.

There are two things I'm really after, and they share the same honest engine:

**1. Self-regulation, proven honestly.** Can a person learn to voluntarily move a specific brain signal, and can I prove that's real rather than placebo or a muscle twitch? That means a sham control, an artifact defense, and measuring learning over time, none of which the app touches.

**2. State exploration, or cartography of my own mind.** Beyond "can I control it," the richer question is how my brain states actually correspond to and respond to the real things I do. Reading versus math versus coding versus music versus rest versus conversation. Do distinct activities leave distinct signatures? How do states shift, settle, and interact as I move between tasks? This is the part that opens up genuinely interesting deep studies and reflections, mining my own recordings for patterns instead of just asserting a number.

So the guiding principle for all the code is honesty over impressiveness. A result that survives scrutiny beats a flashy one every time, and a clean null, reported honestly, is still a real finding. Never fake, smooth over, or overclaim outputs.

## Why it's novel (so you don't accidentally rebuild something ordinary)

Plenty of projects slap a "focused vs relaxed" label on Muse data. Almost none do it rigorously: proving the effect is specific to the target band, proving it isn't an artifact, testing against a sham, and then genuinely exploring how states map onto real activity rather than stopping at a classifier. That rigor and that honest exploration at the cheap-hardware tier is the open gap. Our contribution is the trustworthy pipeline, the controlled result, and the state-mapping study, not a headset trick.

## Scope right now (don't get ahead of this)

This is Semester 1, early phase. Build in this order and don't jump ahead:

1. **Acquisition** — pull raw EEG off the Muse through BrainFlow and save it cleanly, with metadata.
2. **Validity gate** — reproduce the Berger effect (alpha rises when eyes close). If we can't show that, nothing downstream is trustworthy, so this gates everything.
3. **Preprocessing + artifact handling** — filtering, epoching, and quantifying how much blinks and clenches contaminate the signal.
4. **Features** — band powers, ratios, upper-alpha, frontal alpha asymmetry, as a tidy per-epoch table.
5. **State exploration** — record labeled activity sessions and compare their signatures. Which features separate which states, and how reliably.
6. **Real-time loop** — a live feedback signal, built from the start to support a real-vs-sham toggle.

Explicit non-goals for now: no machine-learning classifier yet, no dashboard app yet, no multi-person study yet, no heart-rate/multimodal work yet. Those are later sprints. Keep the codebase from sprawling toward them.

## Tech stack (all open source, keep it lean)

- Python 3.11+
- **BrainFlow** for acquisition (board-agnostic, so the same code runs on a synthetic board and the real Muse with a one-line change)
- **NumPy / SciPy** for the numerical and spectral work (Welch PSD, filtering, stats)
- **Matplotlib** for analysis plots, **Plotly** for anything interactive
- **pytest** for tests, **ruff** for linting/formatting
- MNE-Python gets added later, when we do ICA-style artifact work. Don't add it, or any other dependency, until the code actually needs it.

## Repo structure

```
kestrel/
  README.md
  AGENTS.md               # this brief, so context always travels with the repo
  pyproject.toml          # ruff + pytest config, package metadata
  requirements.txt
  .gitignore
  conftest.py             # puts src/ on the path for tests
  src/kestrel/
    __init__.py
    config.py             # single source of truth: channels, bands, filters, paths
    acquire.py            # record from Muse or synthetic board -> raw file + JSON sidecar
    preprocess.py         # bandpass, notch, epoch, amplitude rejection
    features.py           # Welch PSD, band powers, ratios, frontal alpha asymmetry
    analysis.py           # Berger test, state comparison, effect sizes
    neurofeedback.py      # real-time rolling-window loop; real vs sham feedback modes
  scripts/
    smoke_test.py         # synthetic-board end-to-end check, no hardware
    record_session.py     # CLI: record a labeled block
    run_berger_check.py   # eyes-open vs eyes-closed, report the effect
    run_state_explore.py  # compare signatures across labeled activity sessions
  tests/
    test_features.py      # inject a known-frequency sine, assert the right band dominates
    test_preprocess.py
  notes/learning-notes.md # my running research notes
  docs/
    plans/                # sprint plans
    reflections/          # sprint reflections
    preregistration.md    # hypotheses + analysis plan, written before the real experiment
  data/                   # recordings — GIT-IGNORED, never committed
```

## The pipeline, concretely

Build each stage as a small, testable function, not one giant script.

**Acquire.** BrainFlow `MUSE_2_BOARD`, 256 Hz, channels TP9/AF7/AF8/TP10. Record in labeled blocks (`eyes_open`, `eyes_closed`, `blink`, `clench`, `reading`, `math`, `coding`, `music`, `rest`, and so on). Every recording saves the raw array plus a small JSON sidecar: label, note, sampling rate, channels, sample count, timestamp. That metadata is what makes the state-exploration work possible later.

**Preprocess.** Bandpass roughly 1–40 Hz, notch at 60 Hz (US mains). Epoch into fixed windows with overlap. Reject or flag epochs over an amplitude threshold. With only four channels, don't lean on ICA as the whole answer, treat threshold rejection plus deliberate blink/clench catch blocks as the primary artifact defense, and say so plainly in the code.

**Features.** Welch PSD per epoch. Absolute and relative band power for delta, theta, alpha, beta, gamma. Upper-alpha specifically (most trainable). Frontal alpha asymmetry from AF7 vs AF8. Output a tidy per-epoch feature table, one row per epoch with the label attached.

**Validity gate.** `run_berger_check.py` records eyes-open and eyes-closed, computes alpha in each, and reports whether closed > open with a t-test and Cohen's d. This is the go/no-go for trusting the whole rig.

**State exploration.** `run_state_explore.py` loads several labeled activity recordings, turns each into per-epoch features, and compares their signatures, which features separate which states, how big the differences are, and how consistent they are within a state. This is where the reflective, mining-for-patterns work happens.

**Real-time loop.** A rolling window computes the target band power live and turns it into a feedback signal. Build the real-vs-sham switch in from day one: real mode feeds back the true signal, sham mode feeds back a yoked or randomized signal that looks identical but isn't tied to the brain. Retrofitting that later is painful.

**Analysis.** Effect sizes with confidence intervals, not just p-values. Honest plots. Report brain measures and, later, behavior together.

## How I want you to work

- Small, readable functions over clever one-liners. This is research code a high schooler and an advisor need to read and trust. Clarity wins.
- Comment the why, not the what. Explain a filter choice or a threshold, not that a loop is a loop.
- Write tests for the signal math. The clean way: generate a synthetic signal at a known frequency, run it through the feature code, and assert the right band dominates. If the math breaks, the test catches it.
- Commit small and often, with plain messages that say what changed and why. The history should read like a lab notebook, not "update files."
- Never commit raw data. `data/` stays git-ignored. Keeps the repo light and handles neural-data privacy from the start.
- Pin the environment once it installs clean, so it's reproducible elsewhere.
- Don't over-engineer. No frameworks we don't need, no abstraction for imagined futures. Build for the sprint we're in.
- Flag limitations out loud, in code and docs. Four channels, consumer hardware, single subject. Honesty about limits is a feature here.

## Guardrails

- Research and self-experimentation, not a medical device. No diagnostic or clinical claims anywhere.
- When classmates come in later, do informed consent, de-identified data, local storage, and honor deletion requests. Design the data handling now so that's easy later.
- Follow CRED-nf (the neurofeedback reporting standard): pre-register hypotheses before the real experiment, always include a control, and always include a manipulation check that the target signal actually moved.

## Setup steps

1. The GitHub repo is https://github.com/AdyaS2010/bci-exploration (the project inside it is Kestrel).
2. Clone it, or if starting fresh:

```bash
git clone https://github.com/AdyaS2010/bci-exploration.git
cd bci-exploration
```

3. Environment:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

4. Run the tests and the no-hardware smoke check:

```bash
python -m pytest
python scripts/smoke_test.py
```

5. Freeze versions whenever a dependency is added and it runs clean:

```bash
pip freeze > requirements.txt
```

6. First real milestone: get `run_berger_check.py` working on the actual Muse and confirm the eyes-closed alpha rise. That's when the rig is trustworthy and the real work starts.

## The one-line reminder to keep on the wall

We're not measuring brainwaves. The app already does that. We're studying how the mind's states actually behave, and proving, honestly, whether they can be steered.
