import numpy as np
import pytest
from partipy.arch import AA
from partipy.const import INIT_ALGS, OPTIM_ALGS, WEIGHT_ALGS
from partipy.simulate import simulate_archetypes
from partipy.utils import align_archetypes, compute_relative_rowwise_l2_distance

# for regularized_nnls the tests take much longer, and this algorithm is not recommended
FAST_OPTIM_ALGS = tuple(alg for alg in OPTIM_ALGS if alg != "regularized_nnls")


@pytest.mark.parametrize(
    "n_archetypes, n_dimensions",
    [(n_a, n_d) for n_a in range(2, 9) for n_d in range(2, 19, 4) if n_a <= n_d],
)
@pytest.mark.parametrize("optim_str", OPTIM_ALGS)
def test_that_archetypes_can_be_identified_fail_if_we_dont_optimize(
    n_archetypes: int,
    n_dimensions: int,
    optim_str: str,
) -> None:
    N_SAMPLES = 1_000
    MAX_REL_DIST = 0.10 if n_archetypes < n_dimensions else 0.15
    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES,
        n_archetypes=n_archetypes,
        n_dimensions=n_dimensions,
        noise_std=0.0,
        seed=0,
    )

    AA_object = AA(n_archetypes=n_archetypes, init="uniform", optim=optim_str, max_iter=0)
    AA_object.fit(X)
    Z_hat = AA_object.Z

    Z_hat = align_archetypes(Z, Z_hat)

    rel_dist_between_archetypes = compute_relative_rowwise_l2_distance(Z, Z_hat)

    assert np.any(rel_dist_between_archetypes > MAX_REL_DIST)


@pytest.mark.parametrize(
    "n_archetypes, n_dimensions",
    [(n_a, n_d) for n_a in range(3, 5) for n_d in range(2, 6) if n_a <= n_d],
)
@pytest.mark.parametrize("optim_str", OPTIM_ALGS)
@pytest.mark.parametrize("init_str", INIT_ALGS)
def test_that_all_algorithms_can_identify_archetypes(
    n_archetypes: int,
    n_dimensions: int,
    optim_str: str,
    init_str: str,
) -> None:
    N_SAMPLES = 2_000
    MAX_REL_DIST = 0.8 if n_archetypes < n_dimensions else 0.14
    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES,
        n_archetypes=n_archetypes,
        n_dimensions=n_dimensions,
        noise_std=0.0,
        seed=0,
    )

    AA_object = AA(n_archetypes=n_archetypes, init=init_str, optim=optim_str)
    AA_object.fit(X)
    Z_hat = AA_object.Z

    Z_hat = align_archetypes(Z, Z_hat)

    rel_dist_between_archetypes = compute_relative_rowwise_l2_distance(Z, Z_hat)

    assert np.all(rel_dist_between_archetypes < MAX_REL_DIST)


@pytest.mark.parametrize(
    "n_archetypes, n_dimensions",
    [(n_a, n_d) for n_a in range(3, 8, 2) for n_d in range(2, 13, 2) if n_a <= n_d],
)
# @pytest.mark.parametrize("optim_str", FAST_OPTIM_ALGS)
@pytest.mark.parametrize("optim_str", ["projected_gradients"])
def test_that_fast_algorithms_can_identify_archetypes(
    n_archetypes: int,
    n_dimensions: int,
    optim_str: str,
) -> None:
    if n_dimensions < 8:
        N_SAMPLES = 10_000
    else:
        N_SAMPLES = 100_000
    MAX_REL_DIST = 0.15
    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES,
        n_archetypes=n_archetypes,
        n_dimensions=n_dimensions,
        noise_std=0.0,
        seed=0,
    )

    AA_object = AA(n_archetypes=n_archetypes, optim=optim_str)
    AA_object.fit(X)
    Z_hat = AA_object.Z

    Z_hat = align_archetypes(Z, Z_hat)

    rel_dist_between_archetypes = compute_relative_rowwise_l2_distance(Z, Z_hat)

    assert np.all(rel_dist_between_archetypes < MAX_REL_DIST)


@pytest.mark.parametrize("optim_str", OPTIM_ALGS)
@pytest.mark.parametrize("weight_str", WEIGHT_ALGS)
@pytest.mark.parametrize("init_str", INIT_ALGS)
def test_that_input_to_AA_is_not_modfied(optim_str, weight_str, init_str) -> None:
    N_SAMPLES = 100
    N_ARCHETYPES = 3
    N_DIMENSIONS = 4
    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES,
        n_archetypes=N_ARCHETYPES,
        n_dimensions=N_DIMENSIONS,
        noise_std=0.0,
        seed=0,
    )
    X_in = X.copy()

    AA_object = AA(
        n_archetypes=N_ARCHETYPES,
        optim=optim_str,
        weight=weight_str,
        init=init_str,
        early_stopping=True if weight_str is None else False,
    )
    AA_object.fit(X)

    assert np.all(np.isclose(X_in, X))
