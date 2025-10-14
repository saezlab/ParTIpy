import numpy as np
import pytest
from partipy.arch import AA
from partipy.schema import OPTIM_ALGS
from partipy.simulate import simulate_archetypes
from partipy.utils import align_archetypes, compute_relative_rowwise_l2_distance

# for regularized_nnls the tests take much longer, and this algorithm is not recommended
FAST_OPTIM_ALGS = tuple(alg for alg in OPTIM_ALGS if alg != "regularized_nnls")


@pytest.mark.github_actions
@pytest.mark.parametrize("optim_str", FAST_OPTIM_ALGS)
def test_that_archetypes_can_be_identified_using_coresets_and_uniform_initialization_gh(
    optim_str: str,
) -> None:
    seed = 123
    N_SAMPLES = 1_000
    MAX_REL_DIST = 0.10
    n_archetypes = 3
    n_dimensions = 2

    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES,
        n_archetypes=n_archetypes,
        n_dimensions=n_dimensions,
        noise_std=0.05,
        seed=seed,
    )

    X_in = X.copy()

    AA_object = AA(
        n_archetypes=n_archetypes,
        optim=optim_str,
        coreset_algorithm="standard",
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
    assert np.all(np.isclose(AA_object.A.sum(axis=1), 1, atol=1e-3))
    assert np.all(AA_object.A >= 0)
    assert np.all(np.isclose(AA_object.B.sum(axis=1), 1, atol=1e-3))
    assert np.all(AA_object.B >= 0)

    # now we check how accurately we identify the archetypes
    Z_hat = AA_object.Z

    Z_hat = align_archetypes(Z, Z_hat)

    rel_dist_between_archetypes = compute_relative_rowwise_l2_distance(Z, Z_hat)

    assert np.any(rel_dist_between_archetypes < MAX_REL_DIST)

    # finally test that the input is not modified
    assert np.all(np.isclose(X_in, X))


@pytest.mark.parametrize("n_archetypes", [3, 7])
@pytest.mark.parametrize("n_dimensions", [7, 14])
@pytest.mark.parametrize("optim_str", FAST_OPTIM_ALGS)
@pytest.mark.parametrize("seed", [123, 456, 789])
def test_that_archetypes_can_be_identified_using_coresets_and_uniform_initialization(
    n_archetypes: int,
    n_dimensions: int,
    optim_str: str,
    seed: int,
) -> None:
    N_SAMPLES = 100_000
    if n_dimensions == 7:
        MAX_REL_DIST = 0.10
    elif n_dimensions == 14:
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

    X_in = X.copy()

    AA_object = AA(
        n_archetypes=n_archetypes,
        optim=optim_str,
        coreset_algorithm="standard",
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
    assert np.all(np.isclose(AA_object.A.sum(axis=1), 1, atol=1e-3))
    assert np.all(AA_object.A >= 0)
    assert np.all(np.isclose(AA_object.B.sum(axis=1), 1, atol=1e-3))
    assert np.all(AA_object.B >= 0)

    # now we check how accurately we identify the archetypes
    Z_hat = AA_object.Z

    Z_hat = align_archetypes(Z, Z_hat)

    rel_dist_between_archetypes = compute_relative_rowwise_l2_distance(Z, Z_hat)

    assert np.any(rel_dist_between_archetypes < MAX_REL_DIST)

    # finally test that the input is not modified
    assert np.all(np.isclose(X_in, X))
