# tasks/lessons.md

## 2026-06-09 — documentation style
**Correction:** "Your documentation of the code is really horrible, and not my
style." Terse one-line docstrings are NOT acceptable here.
**Rule:** Before writing code in any of Joseph's projects, read his existing
code first (`Acoustic-BEV/src/evidential` is the reference). Docstrings are
didactic and self-contained: PUBLIC SURFACE / FLOW / DATA CONTRACT sections,
indented ASCII math, aligned shape tables, "Why" rationale paragraphs,
`# ====` section banners, per-arg comments in grouped `__init__` signatures,
commented YAML configs ordered by edit frequency.

## 2026-06-09 — exploration workflow
**Correction:** A 25-file package is not walkable during exploratory research.
"Usually whenever I am iterating or starting exploratory research I prefer to
just leave everything as notebooks. A single end-to-end notebook, to keep
things simple and short."
**Rule:** Exploratory work happens on the `exploration` branch, strictly in
single self-contained notebooks (Settings cell with all knobs first, sections
Dataset -> Model -> Losses -> Training -> Curves -> Inference -> Evaluation ->
Result). Structured packages are for `main`, after an idea has earned it.

## 2026-06-10, diagnosis before intervention
**Correction:** "Just saying this is what we would change up front doesn't do
justice to our suspicion on what could have caused a particular training to
fail." A pre-planned campaign of changes is not research.
**Rule:** Every experiment cycle starts from a SYMPTOM, enumerates competing
HYPOTHESES for its cause, runs cheap DISCRIMINATING diagnostics (decode-only,
re-scoring, per-token analyses) before any retraining, then picks the
intervention implied by the surviving hypothesis, with a stated predicted
effect. An intervention whose result contradicts its predicted mechanism is a
failed explanation even if the metric improves. The ledger entry format is:
Symptom, Hypotheses, Diagnostics, Surviving explanation, Intervention,
Predicted effect, Outcome.
