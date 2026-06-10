# Premise Study — does the SoundReg hypothesis hold on this data?

Branch: `premise-study`. This directory is self-contained: it measures whether the
core SoundReg premise is true **before** any more architecture work, following the
pre-registered protocol in [`PROTOCOL.md`](PROTOCOL.md). No training. The decision
(build v2 / ego-noise pivot / few-mic pivot / rescope) is **computed** by
`decide.py` against frozen thresholds in `config.yaml` — not narrated.

## TL;DR — how to run

```bash
git clone <repo> && cd SoundReg && git checkout premise-study
pip install -r premise_study/requirements.txt
# optional but recommended on your GPU box (auto-detected, big SRP speedup):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 0) verify the pipeline end-to-end on synthetic stand-ins (~5 min, no data needed)
python -m premise_study.run_all --config premise_study/config.yaml --smoke

# 1) wire ONE function to your dataset: iter_samples() in data_adapter.py
#    (schema documented in its docstring; matches the items[token] dict)
#    and set mic_xml + data paths in config.yaml

# 2) build the clip bank + noise beds + stats from the real data
python -m premise_study.data_adapter extract --config premise_study/config.yaml
#    -> prints how many one-box windows exist; WARNS if < 80 (Set A realism risk)

# 3) run all measurements (resumable: re-run the same command if interrupted;
#    completed stages are skipped via their output files)
python -m premise_study.run_all --config premise_study/config.yaml

# 4) read the verdict
python -m premise_study.decide --workdir premise_out --config premise_study/config.yaml
```

Outputs land in `premise_out/`: `m1..m6_*.json` (raw measurements),
`plot1_hiding.png`, `plot2_headroom.png`, `plot3_crb.png` (the three decision
plots), and `decision.json` (P1–P5 verdicts + branch).

## What gets measured (one line each — details in PROTOCOL.md)

| ID | Question | Module |
|---|---|---|
| M1 | How much of the spectrogram does each source own (glimpse measure)? | `glimpse.py` |
| M2 | Does *perfectly* explaining away louder sources help a classical detector find hidden ones? (plain / oracle-SIC / oracle-null) | `detector.py` |
| M3 | Does loudest-first order beat reverse order? | `detector.py` via `run_all` |
| M4 | What can the array physically resolve (CRB for θ and r)? | `crb.py` |
| M6 | At how few microphones does sequencing start to matter? | `run_all.py` |

## Decision rules (frozen)

`config.yaml → decision:` holds the P1–P5 thresholds from PROTOCOL.md §4.
**Edit them only before the first real run**, in a dedicated commit explaining why.
After that they are frozen; `decide.py` reads them, prints PASS/GRAY/FAIL per rule,
and selects the branch by the precedence order P2a-fail → P2b → P4 → P1.

## Honest caveats (also in PROTOCOL.md §6)

- Free-field rendering: no reflections, no car-body shadowing.
- The CRB implemented is the **known-waveform, coherent** bound — deliberately
  optimistic. Logic: P4 **FAIL is decisive** (if even the optimistic bound says
  range is unresolvable, it is); P4 **PASS is necessary, not sufficient**.
- The SRP-PHAT detector measures *information availability*, not every learned
  model's behavior.
- Conclusions are only as strong as the clip bank; the extractor warns when the
  one-box window count is low.

## Where things plug into the existing repo

The study only needs the dataset through `data_adapter.iter_samples()` — wire it
to the existing loader (the function docstring shows the expected per-sample
schema). `geometry.load_mic_positions()` parses the mic XML; if the XML layout
differs from the common patterns it tries, adapt only that function.

## File map

```
premise_study/
  PROTOCOL.md       the pre-registered protocol (read this first)
  config.yaml       paths, grid, STFT, and the FROZEN decision thresholds
  geometry.py       mic XML -> positions; polar grid; near-field steering
  render.py         free-field multichannel rendering; multichannel STFT
  scenes.py         Set A (naturalistic) / Set B (controlled sweeps) / B3 (order)
  glimpse.py        G_j(C) glimpse measures and headroom
  detector.py       SRP-PHAT polar scan; plain / oracle-SIC / oracle-null; matching
  crb.py            Cramér–Rao maps and r_max for range feasibility
  data_adapter.py   the ONLY file touching the real dataset (extract + anchor)
  run_all.py        resumable M1–M6 runner (auto-GPU via torch if present)
  plots.py          the three decision plots
  decide.py         evaluates P1–P5, prints + saves the branch
```

## Runtime expectations

Smoke: ~5 min CPU. Real run at protocol scale (10k Set A + sweeps): the SRP grid
scan dominates; with torch on a single modern GPU expect well under an hour;
CPU-only is feasible overnight. The runner checkpoints after every measurement
(and per-array-size inside M6), so interruptions cost nothing.
