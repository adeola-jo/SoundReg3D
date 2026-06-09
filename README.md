# SoundReg3D — exploration branch

Notebook-only branch for exploratory research. One self-contained end-to-end
notebook per idea; nothing here imports from the structured package (that
lives on `main`).

| Notebook | What it does |
|---|---|
| `notebooks/01_soundreg.ipynb` | SoundReg on the Acoustic-BEV `nn_data` pickles: reuses the MoG-slot front end (SpeedFiLM + conv trunk), swaps the K-slot head for a dominance-ordered causal token decoder, scores with the benchmark matcher (5deg / 5deg+5m gates) |

`metadata/mic_on_the_car.xml` is the 32-mic array geometry the dominance
beamformer uses. Data paths default to `/home/joadeola/Datasets/nn_data`
(override with the `SOUNDREG_DATA` env var). Run outputs land in
`notebooks/runs/` and `notebooks/results/` (gitignored).
