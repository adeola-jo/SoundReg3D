# CLAUDE.md

## Workflow Orchestration

### Rule 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### Rule 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### Rule 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the mistake category
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### Rule 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### Rule 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### Rule 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go for failing CI tests without being told how

## Task Management

1. **Plan First:** Write plan to `tasks/todo.md` with checkable items
2. **Verify Plans:** Check in before implementing implementation
3. **Track Progress:** Mark items complete as you go
4. **Explain Changes:** High-level summary at each step
5. **Document Results:** Add review section to `tasks/todo.md`
6. **Capture Lessons:** Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First:** Make every change as simple as possible. Impact minimal code.
- **No Laziness:** Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact:** Changes should only touch what's necessary. Avoid introducing bugs.

---

## Project-Specific Rules (SoundReg3D)

### What this project is
Autoregressive acoustic object detection (SoundReg): emit sounding road users
`(class, range, azimuth)` from a mic-array STFT window, loudest first (dominance
order), each object conditioned on the ones already emitted. PLAN.md is the
research roadmap; the brief (`soundreg_brief.pdf`) and the AutoReg3D paper
(arXiv:2603.07985) are the source documents.

### Branch workflow
- `main` — the structured `soundreg/` package (modular, tested).
- `exploration` — **notebooks only, strictly.** During exploratory research,
  everything lives in a single end-to-end notebook per idea: short and simple,
  walkable top to bottom. Model the notebooks on
  `adeola-jo/acoustic-bev-benchmark` (cleanup branch, `benchmark/notebooks/`):
  one intro markdown (Method + Structure), then
  `Settings -> Dataset -> Model -> Losses -> Training -> Curves -> Inference -> Evaluation -> Result`.
  ALL paths/hyperparameters/smoke knobs go in ONE Settings cell (repo-root
  walk-up, `LIMIT_*` smoke knobs, fixed `SEED`).

### Code and documentation style
Match `Lahm21111/Acoustic-BEV` `src/evidential` (feature/mog-slot branch):
- Module docstrings are didactic and self-contained: `PUBLIC SURFACE` /
  `FLOW` / `DATA CONTRACT` sections, indented ASCII math, aligned shape
  tables, "Why X:" rationale paragraphs.
- `# ====...====` section banners between logical blocks.
- `__init__` args grouped with `# ---- group ----` dividers and per-arg comments.
- Google-style `Args:` / `Returns:` with explicit shapes like `(B, K, M)`.
- Named module constants with a comment explaining each (`_LOG_EPS`-style).
- Explicit `raise ValueError` with got-values in the message.
- Heavily commented YAML configs, sections ordered by edit frequency.

### Project conventions (fixed — do not change casually)
- theta in degrees in [-180, 180); ego frame x forward, y left; r in meters.
- Circular math everywhere an angle is subtracted: `((a - b + 180) mod 360) - 180`.
- Matching = Hungarian on BEV polar distance, gate 2 m; F1 / cardinality MAE /
  range MAE / circular azimuth MAE; recall stratified by per-source SNR and by
  number of concurrent sources (the headline test).
- Thresholds for score-based baselines are tuned on val ONLY, never test.
- Watch EOS calibration from epoch 1 (early-EOS silently drops masked sources).
- Run artifacts: `runs/<exp_name>/seed<k>/`; headline numbers = mean ± std over seeds.
