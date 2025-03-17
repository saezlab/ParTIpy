import numpy as np
import pytest
from partipy.arch import AA
from partipy.const import INIT_ALGS, OPTIM_ALGS, WEIGHT_ALGS
from partipy.generate_test_data import simulate
from scipy.optimize import linear_sum_assignment


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


@pytest.mark.parametrize("n_archetypes", list(range(2, 8)))
@pytest.mark.parametrize("optim_str", OPTIM_ALGS)
def test_that_archetypes_can_be_identified(
    n_archetypes: int,
    optim_str: str,
) -> None:
    N_SAMPLES = 1_000
    N_DIMENSIONS = 10
    MIN_CORR = 0.9
    X, A, Z = simulate(
        n_samples=N_SAMPLES,
        n_archetypes=n_archetypes,
        n_dimensions=N_DIMENSIONS,
        noise_std=0.0,
        seed=111,
    )

    A_hat, B_hat, Z_hat, RSS, varexpl = AA(n_archetypes=n_archetypes, optim=optim_str).fit(X).return_all()

    Z_hat = align_archetypes(Z, Z_hat)

    corr_between_archetypes = compute_rowwise_correlation(Z, Z_hat)
    assert np.all(corr_between_archetypes > MIN_CORR)


@pytest.mark.parametrize("optim_str", OPTIM_ALGS)
@pytest.mark.parametrize("weight_str", WEIGHT_ALGS)
@pytest.mark.parametrize("init_str", INIT_ALGS)
def test_that_input_to_AA_is_not_modfied(optim_str, weight_str, init_str) -> None:
    N_SAMPLES = 200
    N_DIMENSIONS = 3
    N_ARCHETYPES = 5
    X, A, Z = simulate(
        n_samples=N_SAMPLES,
        n_archetypes=N_ARCHETYPES,
        n_dimensions=N_DIMENSIONS,
        noise_std=0.0,
        seed=111,
    )
    X_in = X.copy()

    A_hat, B_hat, Z_hat, RSS, varexpl = (
        AA(n_archetypes=N_ARCHETYPES, optim=optim_str, weight=weight_str, init=init_str).fit(X).return_all()
    )

    assert np.all(np.isclose(X_in, X))
