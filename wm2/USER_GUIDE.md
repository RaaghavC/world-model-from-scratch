# User guide: run it yourself (step by step)

This walks you through running the project from a blank machine. No prior
experience needed. If a step fails, see **Troubleshooting** at the bottom.

New to the *ideas*? Read [`UNDERSTANDING.md`](UNDERSTANDING.md) first.

---

## 1. What you need

- A **Mac with Apple Silicon** (M1/M2/M3/M4) is ideal — training uses the built-in
  GPU automatically. A Linux/Windows machine with an NVIDIA GPU also works. A
  plain CPU works too, just slower.
- **Python 3.10 or newer**. Check by opening a terminal and typing:
  ```bash
  python3 --version
  ```
  If that errors, install Python from <https://www.python.org/downloads/> (Mac users
  can also use `brew install python`).
- About **2 GB of free disk** and ~**1 hour** for a full from-scratch run.

You do **not** need a powerful computer, an internet connection during training, or
any paid services.

---

## 2. Get the code and install

```bash
# 1. get the code (replace URL with wherever you cloned it from)
git clone <REPOSITORY_URL>
cd worldModel

# 2. (recommended) make an isolated Python environment
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

# 3. install the handful of dependencies
pip install -r requirements.txt
```

That installs PyTorch (the math engine), NumPy, imageio (for GIFs), and matplotlib
(for plots). Nothing else.

---

## 3. Run it

The whole pipeline is one script. From the repo's top folder:

```bash
bash wm2/run_all.sh
```

That runs all stages in order and prints progress. It takes roughly 30–60 minutes on
a laptop. **Or** run stages one at a time to watch each piece (set the path once):

```bash
export PYTHONPATH=$PWD      # tells Python where to find the code; run once per terminal

python3 -m wm2.train collect   # ~5 sec   make practice data (random play)
python3 -m wm2.train wm        # ~12 min  train the world model (See + Remember)
python3 -m wm2.train ac        # ~10 min  train a controller in imagination (optional)
python3 -m wm2.train lam       # ~4 min   the "discover the actions" model (Genie-style)
python3 -m wm2.train jepa      # ~1 min   the decoder-free predictor (JEPA-style)
python3 -m wm2.train eval      # ~2 min   measure everything; writes a results file
python3 -m wm2.train viz       # ~2 min   render the dream GIFs and plots
```

Each stage saves its results to disk, so you can stop and resume. The trained models
land in `wm2_checkpoints/`, and all pictures/plots/numbers in `wm2_artifacts/`.

> Tip: just want to *see* the results without an hour of training? After a run, open
> the files in `wm2_artifacts/` (next section). If checkpoints already exist, you can
> re-run only `eval` and `viz`.

---

## 4. What you get, and how to read it

Everything lands in **`wm2_artifacts/`**. Open these:

| File | What it shows |
|---|---|
| `dream_compare.gif` | Three side-by-side columns: **real** game, the model's **reconstruction**, and its **dream** (imagined from one frame). If the orange dot in the right column tracks the left, the model learned the dynamics. |
| `free_dream.gif` | A *pure* hallucination: one real frame, then everything is imagined. |
| `drift_curve.png` | How dream accuracy degrades the longer it imagines (it rises = drifts). |
| `latent_map.png` | A 2-D map of the model's internal "summaries," coloured by the dot's true position — nearby summaries = nearby positions means the summary is meaningful. |
| `code_action_matrix.png` | The "discover the actions" result: which invented code lines up with which real control. |
| `eval_results.json` | All the numbers (see below). |

**Reading `eval_results.json`** (a text file of results):
- `faithfulness.agent_pos_R2_mean` ≈ **0.97** — how well position is readable from the
  summary (1.0 = perfect).
- `faithfulness.agent_vel_R2_mean` ≈ **0.32** — same for velocity (invisible in one
  frame, so this being above 0 is the interesting part).
- `controllability.wm2_planner.success` ≈ **0.97** vs `controllability.random.success`
  ≈ **0.25** — how often the trained controller vs. random play reaches the goal.
- `latent_action.purity` ≈ **0.37** — how cleanly the *unsupervised* discovered
  actions match the real ones (0.20 would be pure chance).
- `worldscore_composite.mean` ≈ **0.76** — an overall 0–1 score.

A plain-English tour of what these mean is in [`UNDERSTANDING.md`](UNDERSTANDING.md)
§6, and the full analysis is in [`README.md`](README.md) and [`FINDINGS.md`](FINDINGS.md).

---

## 5. Tinker

All settings live in one file, **`wm2/config.py`**, with comments. Some fun knobs:

- `n_bodies` — set to `2` or `3` for extra bouncing obstacles (a harder, "multi-body"
  world with collisions and occlusion).
- `wm_epochs` — train the world model longer (sharper dreams) or shorter (faster).
- `imag_horizon`, and the planner's horizon in `wm2/evaluate.py` (`WMPlanner(..., H=5)`)
  — how far the controller imagines ahead each step.
- `direct_control` — `True` makes the action set velocity directly (the regime where
  the "discover the actions" model works best).

After changing a setting, re-run the relevant stage (e.g. `python3 -m wm2.train wm`).

---

## 6. Compare to the original 2018 model

The repo also contains the 2018 baseline in `wm/`. Run it the same way:

```bash
bash run_all.sh        # the baseline's own pipeline (top-level script)
```

Its results land in `artifacts/`. The whole point of this project is the side-by-side
comparison — see the results table in [`wm2/README.md`](README.md).

---

## 7. Troubleshooting

- **`ModuleNotFoundError: No module named 'wm2'`** — you forgot `export PYTHONPATH=$PWD`
  (run it from the repo's top folder), or you're not in the repo folder.
- **`command not found: python3`** — install Python (§1). On some systems it's just
  `python`.
- **It's using the CPU and feels slow** — that's fine, it still works; just leave it
  longer. On Apple Silicon it should auto-pick the GPU ("device=mps" in the logs).
- **Out of memory** — lower `wm_batch` and `wm_seq_len` in `wm2/config.py`.
- **A GIF looks noisy in the background** — known cosmetic quirk (the model
  concentrates on the moving dot); the dot's *motion* is what's learned and measured.
- **Want to start clean** — delete `wm2_data/`, `wm2_checkpoints/`, and
  `wm2_artifacts/`, then re-run.

That's everything. Have fun poking at a world model you trained yourself.
