# SoundReg

Autoregressive acoustic object detection: emit the set of sounding road users
`(class, range, azimuth)` from a multichannel mic-array window, **loudest first**
(dominance order), each object conditioned on those already emitted.

See **PLAN.md** for the full research plan, conventions, milestones, and the
experiment matrix. The formulation follows the working brief
(`soundreg_brief.pdf`) and adapts AutoReg3D (arXiv:2603.07985) from LiDAR
near-to-far ordering to acoustic dominance ordering.

## Setup

```bash
pip install -e .[dev]
pytest tests/            # fast end-to-end checks (~1 min CPU)
```

## Run

```bash
# SoundReg on synthetic scenes (end-to-end today, real data lands in M1)
python scripts/train.py --config experiments/configs/soundreg_synth.yaml

# Baselines through the exact same trainer/metrics
python scripts/train.py --config experiments/configs/srp_peaks.yaml
python scripts/train.py --config experiments/configs/heatmap_synth.yaml

# Ablations: one config flag each
python scripts/train.py --config experiments/configs/ablation_order_random.yaml
python scripts/train.py --config experiments/configs/ablation_context_masked.yaml
python scripts/train.py --config experiments/configs/soundreg_synth.yaml \
    --set data.ordering=near_to_far --set exp_name=ablation_order_n2f

# Re-decode a trained run (beam, length-controlled, sampling)
python scripts/eval.py --run runs/soundreg_synth/seed0 --mode beam
python scripts/eval.py --run runs/soundreg_synth/seed0 --min-objects 3
```

Run artifacts: `runs/<exp_name>/seed<k>/{config.yaml, log.jsonl, best.pt,
metrics.json, stratified.npz}`. Headline numbers are mean ± std over seeds.

## Extending (the research axes)

| Axis | Where | How |
|---|---|---|
| Encoder/backbone | `soundreg/models/encoders/` | register in `ENCODERS`, name it in `model.encoder.name` |
| Emission ordering | `soundreg/data/ordering.py` | register in `ORDERINGS`, set `data.ordering` |
| Detector/baseline | `soundreg/models/` | implement `DetectorBase`, register in `MODELS`, wire in `factory.py` |
| Features | `soundreg/data/features.py` | add a stack, list it in `data.features` |
| Decoding | `soundreg/models/decoder.py` | greedy / sample / beam / `min_objects` already in |
