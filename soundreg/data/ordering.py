"""Object emission orderings — the central ablation axis of the project.

The SoundReg hypothesis lives or dies here: dominance order (loudest
first) should beat content-free orders (random, azimuth sweep) and the
LiDAR-native order (near-to-far) precisely on low-SNR masked sources.
The oracle orders bound what a perfect dominance estimator could buy.

Each ordering is a plain function

    (gt: SceneGT, ctx: dict, rng) -> permutation over range(gt.n)

registered in ORDERINGS, so a new ordering is one decorated function and
one config value (`data.ordering: <name>`). `ctx` carries the side
information the dataset computed:

    ctx["dominance"]  (N,)  steered-response power at each GT position.
                            Always available (computed from the mixture).
    ctx["energy"]     (N,)  TRUE received energy per source.
                            Simulation only — this is the oracle.

The ordering decides what the decoder may condition on: when object i is
emitted, objects pi(1) ... pi(i-1) are visible context. Loudest-first
means the easy, masking sources come out before the hard, masked ones.
"""

from __future__ import annotations

import numpy as np

from ..registry import ORDERINGS
from ..types import SceneGT


@ORDERINGS.register("dominance")
def dominance(gt: SceneGT, ctx: dict, rng) -> np.ndarray:
    """Loudest-first by beamformed steered-response power.

    The SoundReg default (brief Eq. 3). Estimated from the mixture, so it
    is what the model can actually have at inference-relevant fidelity.
    """
    return np.argsort(-np.asarray(ctx["dominance"]))


@ORDERINGS.register("oracle_energy")
def oracle_energy(gt: SceneGT, ctx: dict, rng) -> np.ndarray:
    """Loudest-first by true received energy. Upper bound for the
    dominance-estimator ablation; simulation only."""
    return np.argsort(-np.asarray(ctx["energy"]))


@ORDERINGS.register("oracle_snr")
def oracle_snr(gt: SceneGT, ctx: dict, rng) -> np.ndarray:
    """Easiest-first by true per-source mixture SNR. Simulation only.

    Subtly different from oracle_energy: a quiet source alone in its
    band can have high SNR. Comparing the two tells whether "loud" or
    "detectable" is the better curriculum.
    """
    return np.argsort(-np.asarray(gt.snr_db))


@ORDERINGS.register("near_to_far")
def near_to_far(gt: SceneGT, ctx: dict, rng) -> np.ndarray:
    """AutoReg3D's LiDAR ordering. In LiDAR, near occludes far; in audio
    the analogous structure is loud masks quiet — if dominance does not
    beat this, the acoustic analogy is doing no work."""
    return np.argsort(gt.r)


@ORDERINGS.register("azimuth")
def azimuth(gt: SceneGT, ctx: dict, rng) -> np.ndarray:
    """Sweep from -180 deg upward. Deterministic but content-free: a
    structure-without-signal control."""
    return np.argsort(gt.theta)


@ORDERINGS.register("random")
def random_order(gt: SceneGT, ctx: dict, rng) -> np.ndarray:
    """Fresh permutation per sample (per-index RNG, so still reproducible).
    The no-structure control every learned ordering must beat."""
    return rng.permutation(gt.n)


def order_objects(name: str, gt: SceneGT, ctx: dict, rng) -> np.ndarray:
    """Registry dispatch used by the dataset."""
    return ORDERINGS.get(name)(gt, ctx, rng)
