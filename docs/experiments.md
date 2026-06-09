# SoundReg experiment ledger (exploration branch)

**GOAL (set by Joseph, 2026-06-10): the best single configuration by F1 across
ALL four test sets, target 0.65 to 0.70 F1 on every split, beating the
evidential reference (0.5515 @5deg static_easy). Everything is negotiable:
architecture components (FiLM included), backbone formulation, MAX_OBJECTS,
training recipe, data augmentation (must respect the rig: fixed rooftop array,
driving scenario; superposition-based scene mixing is valid, rotations and
channel permutations are not). Process: diagnose, hypothesize, intervene;
research literature when the evidence points outside the current design.**

One entry per experiment. Single-change discipline: every run differs from its
stated control by exactly one thing, so improvements stay attributable.
Conventions: benchmark matcher, gates 5deg and 5deg+5m; reference rows are
evidential MoG-slot (F1 0.5515 @5deg / 0.4706 @5deg+5m on static_easy).
Artifacts: notebooks/runs/<run>/ (checkpoints, history.csv, curves.png) and
notebooks/results/ (pred + metrics CSVs), both local/gitignored; this ledger is
the committed record. Train data: 4193 scenes (1035 with 0 objects, 2465 with
1, 608 with 2, 85 with 3+); val 280; tests 139/158/277/235 scenes.

---

## E1: v1 baseline (control)

Run `soundreg_v1`, 2026-06-10 00:36. Notebook as committed: reused MoG-slot
front end (random init), mean over T, 360 angular tokens, decoder d=192 x4,
vocab 24 r x 72 az, greedy decode, 60 epochs.

| split | gate | P | R | F1 | mAAE | mADE |
| --- | --- | --- | --- | --- | --- | --- |
| static_easy | 5deg | 0.531 | 0.410 | 0.462 | 2.42 | |
| static_easy | 5deg+5m | 0.494 | 0.381 | 0.430 | 2.45 | 1.79 |
| static_difficult | 5deg | 0.465 | 0.251 | 0.326 | 2.32 | |
| static_difficult | 5deg+5m | 0.343 | 0.186 | 0.241 | 2.36 | 1.76 |
| easy_together | 5deg | 0.517 | 0.413 | 0.459 | 2.30 | |
| easy_together | 5deg+5m | 0.461 | 0.368 | 0.409 | 2.28 | 1.82 |
| difficult_together | 5deg | 0.437 | 0.248 | 0.316 | 2.33 | |
| difficult_together | 5deg+5m | 0.338 | 0.192 | 0.245 | 2.30 | 1.69 |

EOS Brier 0.127 (P(EOS) 0.22 when should continue, 0.80 when should stop).
Training: best val CE at epoch 7 (1.35), divergence to 2.99 by epoch 60.

**Interpretation.** Precision and range error already beat the evidential
reference (P 0.53 vs 0.45; mADE 1.8 vs 2.3); the entire F1 gap is recall
(0.41 vs 0.71). Two confirmed causes: conservative cardinality (the decoder
emits 81 detections for 105 GT on static_easy) and severe overfitting (the
usable model existed at epoch 7 of 60). Both were predicted by the audit
(brief Section 8, findings 4 and 5).

**Decisions.** (a) attack recall first with length-controlled decoding on the
existing checkpoint, zero training cost (E2); (b) fix the training regime
before further architecture ablations so they compare clean models (E3);
(c) then the v2 component ablations one at a time.

---

## D1: diagnostic battery on the trained v1 checkpoints (no training)

Script: decode-only, both checkpoints, gate sweep, cardinality confusion,
recall by dominance rank, per-token-type val CE, length-controlled arm.
Raw output: notebooks/results/diag_v1.json.

**Symptom A: recall 0.41 vs reference 0.71.** Hypotheses and verdicts:

| hypothesis | discriminating evidence | verdict |
| --- | --- | --- |
| A1 cardinality prior, early EOS | confusion matrix: the decoder NEVER emits a 2nd object on any split (pred N <= 1 in 100% of scenes; 2-GT scenes get 1 or 0 predictions) | CONFIRMED, strong form |
| A2 localization just outside the 5 deg gate | recall 0.41 -> 0.61 when the gate widens 5 -> 10 deg (static_easy); +0.15 on difficult splits | CONFIRMED, second factor |
| A3 masked/2nd-rank objects under-recovered | recall by dominance rank: rank0 0.27-0.43 vs rank1+ 0.16-0.20 | CONFIRMED (relevant to the core hypothesis later) |
| A4 token-CE is a bad model-selection proxy | last.pt (epoch 60) F1 0.602 static_easy vs best.pt (epoch 7) 0.462; 0.457 vs 0.316 on difficult_together | CONFIRMED, dominant factor |

**Symptom B: val CE divergence after epoch 7.** Per-type CE roughly doubles
(range 2.60 -> 5.63, azimuth 1.63 -> 3.09, EOS 0.30 -> 1.13) while detection
F1 RISES sharply. So the divergence is an overconfidence/calibration artifact
of teacher-forced CE, not detection overfitting. The earlier "severe
overfitting" reading of E1 was wrong as stated; corrected here.

**Length-controlled arm (best.pt, min 1 object):** F1 0.462 -> 0.533
static_easy, +0.07 to +0.08 everywhere; confirms A1 mechanically. But plain
greedy on last.pt (0.602) beats even that, so checkpoint selection dominates.

**Surviving explanation.** v1's reported numbers were limited by (1) selecting
the checkpoint by val CE, (2) a hard learned cardinality cap at one object
(train prior: 84% of scenes have <= 1 object), (3) angular error mass in the
5-10 deg band.

**Decisions.** D2 (no training): score last.pt fully (both gates, mADE, EOS,
cardinality confusion, with and without min-1 decode) for the corrected v1
baseline. E3 (training): retrain identically but select checkpoints by val F1
(decoded), 120 epochs, since F1 was still climbing at 60. Predicted effect:
E3 best >= last.pt everywhere; if val F1 plateaus then falls, true detection
overfitting exists after all and we will see where.
