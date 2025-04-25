import numpy as np
import pytest
from partipy.optim import _compute_A_frank_wolfe, _compute_A_projected_gradients, _compute_A_regularized_nnls
from partipy.simulate import simulate_archetypes
from scipy.optimize import linear_sum_assignment

compute_a_functions = [_compute_A_regularized_nnls, _compute_A_projected_gradients, _compute_A_frank_wolfe]


def compute_dist_mtx(mtx_1, mtx_2):
    AB = np.dot(mtx_1, mtx_2.T)
    AA = np.sum(np.square(mtx_1), axis=1)
    BB = np.sum(np.square(mtx_2), axis=1)
    dist_mtx = (BB - 2 * AB).T + AA
    dist_mtx[np.isclose(dist_mtx, 0)] = (
        0  # avoid problems if we get small negative numbers due to numerical inaccuracies
    )
    dist_mtx = np.sqrt(dist_mtx)
    return dist_mtx


def align_archetypes(ref_arch, query_arch):
    # not sure if copy here is needed, compute_dist_mtx should not modify the matrices
    euclidean_d = compute_dist_mtx(ref_arch, query_arch.copy()).T
    ref_idx, query_idx = linear_sum_assignment(euclidean_d)
    return query_arch[query_idx, :]


def compute_rowwise_correlation(mtx_1, mtx_2):
    assert np.all(mtx_1.shape == mtx_2.shape)
    mtx_1 = mtx_1 - mtx_1.mean(axis=1, keepdims=True)
    mtx_1 /= mtx_1.std(axis=1, keepdims=True)
    mtx_2 = mtx_2 - mtx_2.mean(axis=1, keepdims=True)
    mtx_2 /= mtx_2.std(axis=1, keepdims=True)
    corr_vec = np.mean(mtx_1 * mtx_2, axis=1)
    return corr_vec


@pytest.mark.parametrize(
    "n_archetypes, n_dimensions",
    [(n_a, n_d) for n_a in range(3, 8) for n_d in range(4, 20, 2) if n_a <= n_d],
)
@pytest.mark.parametrize("n_samples", [100, 1_000, 10_000])
@pytest.mark.parametrize("compute_a_algorithm", compute_a_functions)
def test_that_archetypal_weights_can_be_identified(
    n_archetypes: int,
    n_dimensions: int,
    n_samples: int,
    compute_a_algorithm,
) -> None:
    MEAN_CORR = 0.90
    X, A, Z = simulate_archetypes(
        n_samples=n_samples,
        n_archetypes=n_archetypes,
        n_dimensions=n_dimensions,
        noise_std=0.0,
        seed=42,
    )

    # important as the optimization expects this datatypes, otherwise get a numba.core.errors.TypingError
    X, A, Z = X.astype(np.float32), A.astype(np.float32), Z.astype(np.float32)

    rng = np.random.default_rng(42)
    A_init = -np.log(rng.random((n_samples, n_archetypes), dtype=np.float32))
    A_init /= np.sum(A_init, axis=1, keepdims=True)

    A_approx = compute_a_algorithm(X=X, Z=Z, A=A_init)
    row_cors = compute_rowwise_correlation(A, A_approx)

    assert np.mean(row_cors) >= MEAN_CORR
