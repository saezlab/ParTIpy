import numpy as np
import pytest
from partipy.arch import AA
from partipy.schema import OPTIM_ALGS
from partipy.selection import compute_IC
from partipy.simulate import simulate_archetypes

# for regularized_nnls the tests take much longer, and this algorithm is not recommended
FAST_OPTIM_ALGS = tuple(alg for alg in OPTIM_ALGS if alg != "regularized_nnls")


@pytest.mark.github_actions
def test_that_IC_works_gh() -> None:
    N_SAMPLES = 200
    N_DIMENSION = 2
    n_archetypes = 3
    X_true, A_true, Z_true = simulate_archetypes(
        n_samples=N_SAMPLES,
        n_archetypes=n_archetypes,
        n_dimensions=N_DIMENSION,
        noise_std=0.05,
        seed=0,
    )

    k_test_range = np.array(list(range(2, 7)))
    IC_vec = np.zeros_like(k_test_range, dtype=np.float64)
    for k_idx, k in enumerate(k_test_range):
        AA_object = AA(n_archetypes=k)
        AA_object.fit(X_true)
        X_hat = AA_object.A @ AA_object.Z
        IC_vec[k_idx] = compute_IC(X=X_true, X_tilde=X_hat, n_archetypes=k)

    assert k_test_range[np.argmin(IC_vec)] == n_archetypes


# @pytest.mark.parametrize("n_archetypes", [3, 5, 7])
@pytest.mark.parametrize("n_archetypes", [3, 4, 5])
def test_that_IC_works(
    n_archetypes: int,
) -> None:
    N_SAMPLES = 1_000
    N_DIMENSION = 10
    X_true, A_true, Z_true = simulate_archetypes(
        n_samples=N_SAMPLES,
        n_archetypes=n_archetypes,
        n_dimensions=N_DIMENSION,
        noise_std=0.05,
        seed=0,
    )

    k_test_range = np.array(list(range(2, 10)))
    IC_vec = np.zeros_like(k_test_range, dtype=np.float64)
    for k_idx, k in enumerate(k_test_range):
        AA_object = AA(n_archetypes=k)
        AA_object.fit(X_true)
        X_hat = AA_object.A @ AA_object.Z
        IC_vec[k_idx] = compute_IC(X=X_true, X_tilde=X_hat, n_archetypes=k)

    assert k_test_range[np.argmin(IC_vec)] == n_archetypes
