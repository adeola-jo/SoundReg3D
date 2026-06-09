# SoundReg3D (exploration branch)

SoundReg casts acoustic object detection as autoregressive sequence
generation: from one multichannel STFT window of the car's 32-microphone
array, emit the set of sounding road users as (range, azimuth) tokens,
one object at a time, **loudest first** (dominance order), each object
conditioned on the ones already emitted. `[EOS]` ends the scene, so the
number of objects needs no slot count and no confidence threshold.

This branch is **notebooks only**. Every idea lives in one self-contained
end-to-end notebook (settings, data, model, training, evaluation, result)
that can be read top to bottom and rerun with Run All. The structured
package lives on `main`; nothing here imports from it.

| Notebook | What it does |
| --- | --- |
| `notebooks/01_soundreg.ipynb` | SoundReg trained from scratch on the Acoustic-BEV `nn_data` pickles. Reuses the MoG-slot front end (SpeedFiLM + conv trunk from `Acoustic-BEV/src/evidential`), swaps the K-slot MoG head for a dominance-ordered causal token decoder, exports benchmark-format prediction CSVs, and scores with the benchmark matcher. |

Full method and implementation documentation: **`docs/01_soundreg.md`**
(data contract, geometry conventions and the axis swap, dominance math,
tokenization, model shapes, training, evaluation, known gaps relative to
the working brief).


# =====================================================================
# Quick Start
# =====================================================================

On the machine that has the data (defaults point at
`/home/joadeola/Datasets/nn_data`):

```bash
git checkout exploration
jupyter lab notebooks/01_soundreg.ipynb     # then Run All
```

If the data lives elsewhere, point at it before starting Jupyter:

```bash
export SOUNDREG_DATA=/path/to/nn_data
```

Fast end-to-end check (a few minutes, no full training): edit the smoke
knobs at the bottom of the Settings cell and Run All:

```python
LIMIT_TRAIN, LIMIT_VAL, LIMIT_TEST = 256, 64, 64
EPOCHS = 3
```

Dependencies: torch, numpy, scipy, matplotlib, jupyter. No package
install is needed on this branch.


# =====================================================================
# Settings Workflow
# =====================================================================

Each notebook has exactly ONE Settings cell, right after the title. All
paths, hyperparameters, and experiment knobs live there; nothing below it
should need editing for a normal run. The cell resolves the repo root by
walking up from the notebook's working directory, so the notebook runs
from anywhere.

The knobs that change between experiments:

| Knob | Meaning | Default |
| --- | --- | --- |
| `RUN_NAME` | run folder under `notebooks/runs/` | `soundreg_v1` |
| `ORDERING` | emission order: `dominance`, `random`, `near_to_far` | `dominance` |
| `N_R`, `N_THETA` | polar vocabulary resolution (range, azimuth bins) | 24, 72 |
| `MAX_OBJECTS` | decoder capacity cap | 6 |
| `EPOCHS`, `BATCH_SIZE`, `LR` | training budget | 60, 32, 3e-4 |
| `ANGLE_GATE`, `RANGE_GATE` | benchmark matching gates | 5 deg, 5 m |
| `LIMIT_*` | smoke knobs, `None` for full runs | `None` |
| `SEED` | fixed benchmark-wide seed | 42069 |

One experiment = one change in Settings, rerun, and a new `RUN_NAME` so
the previous run's artifacts survive.


# =====================================================================
# Data And Geometry
# =====================================================================

The data is exactly what the acoustic-bev-benchmark uses: `train.pkl`,
`val.pkl`, and four test splits (`static_easy`, `static_difficult`,
`easy_together`, `difficult_together`). Each sample carries a 32-mic STFT
stored as `(64, F=84, T)` with interleaved real/imag channels (mic i:
real `2i`, imag `2i+1`), the ego velocity, and the annotated sounding
objects. Targets follow the benchmark convention:
`theta = atan2(y, x) % 360`, `r = hypot(x, y)`, objects beyond 30 m
dropped.

`metadata/mic_on_the_car.xml` is the 32-microphone array geometry (two
mounting heights, four clusters around the car). The dominance beamformer
parses it and swaps the file's `(x_fwd, y_up, z_lat)` axes into the
annotation ground plane; see `docs/01_soundreg.md` section 3 for why.


# =====================================================================
# Outputs
# =====================================================================

Run artifacts are gitignored and land next to the notebooks:

    notebooks/
    ├── runs/<RUN_NAME>/
    │   ├── history.csv            per-epoch train/val loss and token accuracy
    │   ├── best.pt                lowest val loss checkpoint
    │   └── last.pt
    └── results/
        ├── pred_soundreg_<split>.csv    benchmark-format predictions
        └── metrics_soundreg.csv         one row per (split, gate)

`metrics_soundreg.csv` has the same columns as the benchmark's
`benchmark_table_full.csv` (tp, fp, fn, precision, recall, f1, mAAE_deg,
mADE_m), so SoundReg rows can be compared directly against the existing
models. Reference row: evidential MoG-slot reaches F1 0.5515 at the 5 deg
gate on `static_easy` and 0.4706 at 5 deg + 5 m.

How to read a run: check the `*_together` splits first (multi-source
scenes are where dominance-ordered conditioning is supposed to pay), and
read the printed EOS calibration line before blaming weak recall on the
idea. Early `[EOS]` silently drops quiet sources and is a calibration
problem, not a formulation problem.


# =====================================================================
# Branch Rules
# =====================================================================

- One self-contained notebook per idea. No imports from the `main`
  package; if an idea graduates, it gets promoted to `main` as
  structured code.
- All knobs in the Settings cell, smoke knobs included.
- Run artifacts stay out of git (`notebooks/runs/`, `notebooks/results/`).
- Plain commit messages.

Sources: the working brief (`soundreg_brief (1).pdf`) defines the
formulation; AutoReg3D (`2603.07985v1 (1).pdf`) is the sequence-detection
recipe it adapts; `Acoustic-BEV/src/evidential` is the origin of the
reused front end and the reference for code style.
