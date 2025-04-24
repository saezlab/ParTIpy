import numpy as np
import pytest
from partipy.arch import AA
from partipy.const import OPTIM_ALGS, WEIGHT_ALGS
from partipy.simulate import simulate_archetypes

from .utils import align_archetypes, compute_relative_rowwise_l2_distance


@pytest.mark.parametrize(
    "n_archetypes, n_dimensions",
    [(n_a, n_d) for n_a in range(3, 8, 2) for n_d in range(4, 9, 2) if n_a < n_d],
)
@pytest.mark.parametrize("optim_str", OPTIM_ALGS)
@pytest.mark.parametrize("weight_str", [w for w in WEIGHT_ALGS if w is not None])
def test_that_all_algorithms_can_identify_archetypes_in_robust_mode(
    n_archetypes: int,
    n_dimensions: int,
    optim_str: str,
    weight_str: str,
) -> None:
    N_SAMPLES = 1_000
    MAX_REL_DIST = 0.8 if n_archetypes < n_dimensions else 0.14
    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES,
        n_archetypes=n_archetypes,
        n_dimensions=n_dimensions,
        noise_std=0.0,
        seed=0,
    )

    # add outliers
    n_outliers = 1
    n_points_per_outlier = 3
    rng = np.random.default_rng(seed=0)
    outlier_indices = rng.choice(N_SAMPLES, size=n_outliers, replace=False)
    outlier_mtx = np.zeros((n_outliers * n_points_per_outlier, n_dimensions))
    for i, outlier_idx in enumerate(outlier_indices):
        start_idx = i * n_points_per_outlier
        end_idx = (i + 1) * n_points_per_outlier
        outlier_mtx[start_idx:end_idx, :] = rng.normal(
            loc=X[outlier_idx, :] * 3, scale=0.2, size=(n_points_per_outlier, n_dimensions)
        )
    X = np.vstack([X, outlier_mtx])

    AA_object = AA(
        n_archetypes=n_archetypes, init="uniform", optim=optim_str, weight=weight_str, early_stopping=False, max_iter=10
    )
    AA_object.fit(X)
    Z_hat = AA_object.Z

    Z_hat = align_archetypes(Z, Z_hat)

    rel_dist_between_archetypes = compute_relative_rowwise_l2_distance(Z, Z_hat)

    assert np.all(rel_dist_between_archetypes < MAX_REL_DIST)
