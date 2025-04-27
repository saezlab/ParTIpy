import numpy as np
import pytest
from partipy.arch import AA
from partipy.simulate import simulate_archetypes
from partipy.utils import align_archetypes, compute_relative_rowwise_l2_distance


@pytest.mark.parametrize("n_archetypes", [3, 7])
@pytest.mark.parametrize("n_dimensions", [5, 10])
@pytest.mark.parametrize("seed", [123, 456, 789])
def test_that_archetypes_can_be_identified_using_coresets_and_uniform_initialization(
    n_archetypes: int,
    n_dimensions: int,
    seed: int,
) -> None:
    N_SAMPLES = 100_000
    if n_dimensions == 5:
        MAX_REL_DIST = 0.10
    elif n_dimensions == 10:
        MAX_REL_DIST = 0.20
    else:
        raise NotImplementedError()

    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES,
        n_archetypes=n_archetypes,
        n_dimensions=n_dimensions,
        noise_std=0.05,
        seed=seed,
    )

    AA_object = AA(
        n_archetypes=n_archetypes,
        use_coreset=True,
        coreset_fraction=0.10,
        init="uniform",
        seed=seed,
    )

    AA_object.fit(X)

    # check that all constraints are satisfied and that A and B have the correct shape
    assert AA_object.A.shape[0] == N_SAMPLES
    assert AA_object.A.shape[1] == n_archetypes
    assert AA_object.B.shape[1] == N_SAMPLES
    assert AA_object.B.shape[0] == n_archetypes
    assert np.all(np.isclose(AA_object.A.sum(axis=1), 1))
    assert np.all(AA_object.A >= 0)
    assert np.all(np.isclose(AA_object.B.sum(axis=1), 1))
    assert np.all(AA_object.B >= 0)

    # now we check how accurately we identify the archetypes
    Z_hat = AA_object.Z

    Z_hat = align_archetypes(Z, Z_hat)

    rel_dist_between_archetypes = compute_relative_rowwise_l2_distance(Z, Z_hat)

    assert np.any(rel_dist_between_archetypes < MAX_REL_DIST)
