"""
Optimize the archetypal analysis objective by block coordiante descent.

a) Regularized Nonnegative Least Squares
- Paper: A. Cutler and L. Breiman, “Archetypal analysis,” Technometrics, vol. 36, no. 4, pp. 338-347, 1994, doi: 10.1080/00401706.1994.10485840.


b) Projected Gradients (PCHA)
- Paper: M. Mørup and L. K. Hansen, “Archetypal analysis for machine learning and data mining,” Neurocomputing, vol. 80, pp. 54-63, Mar. 2012, doi: 10.1016/j.neucom.2011.06.033.


c) Adapted Frank-Wolfe algorithm
- Paper: C. Bauckhage, K. Kersting, F. Hoppe, and C. Thurau, “Archetypal analysis as an autoencoder,” presented at the Workshop “New Challenges in Neural Computation” (NC2) 2015, 2015. Accessed: Feb. 10, 2025. [Online]. Available: https://publica.fraunhofer.de/handle/publica/393337


Code adapted from
a) https://github.com/nichohelmut/football_results/blob/master/clustering/clustering.py
b) https://github.com/atmguille/archetypal-analysis (by Guillermo García Cobo)
"""

import numpy as np
from numba import float32, int32, jit, prange
from scipy.optimize import nnls

from .const import LAMBDA


def _inspect_array(arr: np.ndarray) -> dict:
    """
    Return key properties of a NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        The array to inspect.

    Returns
    -------
    dict
        Dictionary containing dtype, ndim, layout, writeability, and alignment.
    """
    layout = "C" if arr.flags["C_CONTIGUOUS"] else "F" if arr.flags["F_CONTIGUOUS"] else "A"
    return {
        "dtype": str(arr.dtype),
        "ndim": arr.ndim,
        "layout": layout,
        "read_only": not arr.flags["WRITEABLE"],
        "aligned": arr.flags["ALIGNED"],
    }


@jit(
    [float32(float32[:, :], float32[:, :], float32[:, :])], nopython=True, cache=True, fastmath=True
)  # pragma: no cover
def _compute_RSS_AZ(X, A, Z):
    X = np.ascontiguousarray(X)
    Z = np.ascontiguousarray(Z)
    A = np.ascontiguousarray(A)
    diff = X - np.dot(A, Z)
    return np.sum(diff * diff)


@jit(
    [float32(float32[:, :], float32[:, :], float32[:, :], float32[:, :])], nopython=True, cache=True, fastmath=True
)  # pragma: no cover
def _compute_RSS_ABX(X, A, B, WX):
    X = np.ascontiguousarray(X)
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    WX = np.ascontiguousarray(WX)
    diff = WX - np.dot(A, np.dot(B, X))
    return np.sum(diff * diff)


def _compute_A_regularized_nnls(
    X: np.ndarray,
    Z: np.ndarray,
    A: np.ndarray | None = None,
) -> np.ndarray:
    # huge_constant is added as a new column to account for w norm constraint
    X_padded = np.hstack([X, (LAMBDA * np.ones(X.shape[0]))[:, None]])
    Zt_padded = np.vstack([Z.T, LAMBDA * np.ones(Z.shape[0])])

    # Use non-negative least squares to solve the optimization problem
    A = np.array([nnls(A=Zt_padded, b=X_padded[n, :], maxiter=5 * Zt_padded.shape[1])[0] for n in range(X.shape[0])])
    A = A.astype(np.float32)
    return A


def _compute_B_regularized_nnls(
    X: np.ndarray,
    A: np.ndarray,
    B: np.ndarray | None = None,
    WX: np.ndarray = None,  # type: ignore[assignment]
) -> np.ndarray:
    # works for weighted and unweighted version
    if WX is None:
        WX = X
    else:
        assert (WX.shape[0] == X.shape[0]) and (WX.shape[1] == X.shape[1])
    Z = np.linalg.lstsq(a=A, b=WX, rcond=None)[0]
    Z_padded = np.hstack([Z, (LAMBDA * np.ones(Z.shape[0]))[:, None]])
    Xt_padded = np.vstack([X.T, LAMBDA * np.ones(X.shape[0])])  # this should actually be precomputed once
    B = np.array([nnls(A=Xt_padded, b=Z_padded[k, :], maxiter=5 * Xt_padded.shape[1])[0] for k in range(Z.shape[0])])
    B = B.astype(np.float32)
    return B


def _compute_A_projected_gradients(
    X: np.ndarray,
    Z: np.ndarray,
    A: np.ndarray,
    derivative_max_iter: int | np.int32 = 40,
    rel_tol_ls: float | np.float32 = 1e-3,
    rel_tol_conv: float | np.float32 = 1e-4,
) -> np.ndarray:
    """Updates the A matrix given the data matrix X and the archetypes Z.

    A is the matrix that provides the best convex approximation of X by Z.

    Parameters
    ----------
    X : numpy 2d-array
        Data matrix with shape (n_samples, n_features).

    Z : numpy 2d-array
        Archetypes matrix with shape (n_archetypes, n_features).

    A : numpy 2d-array
        A matrix with shape (n_samples, n_archetypes).

    derivative_max_iter: int
        Maximum number of steps for optimization

    Returns
    -------
    A : numpy 2d-array
        Updated A matrix with shape (n_samples, n_archetypes).
    """
    # check and other things
    assert (rel_tol_ls >= 0) and (rel_tol_conv >= 0)

    # ensure correct data type for parameters
    derivative_max_iter = np.int32(derivative_max_iter)
    rel_tol_ls = np.float32(rel_tol_ls)
    rel_tol_conv = np.float32(rel_tol_conv)

    # optimize A
    A = _compute_A_projected_gradients_jit(
        X=X, Z=Z, A=A, derivative_max_iter=derivative_max_iter, rel_tol_ls=rel_tol_ls, rel_tol_conv=rel_tol_conv
    )
    return A


@jit(
    [float32[:, :](float32[:, :], float32[:, :], float32[:, :], int32, float32, float32)],
    nopython=True,
    cache=True,
    fastmath=True,
)  # pragma: no cover
def _compute_A_projected_gradients_jit(
    X: np.ndarray,
    Z: np.ndarray,
    A: np.ndarray,
    derivative_max_iter: np.int32,
    rel_tol_ls: np.float32,
    rel_tol_conv: np.float32,
) -> np.ndarray:
    # initialize learning rate
    mu = np.float32(1.0)
    min_mu = np.float32(1e-6)

    # explicitly making sure everything is contiguous (C-order)
    X = np.ascontiguousarray(X)
    Z = np.ascontiguousarray(Z)
    A = np.ascontiguousarray(A)

    # terms that can be pre-computed
    XZT = np.dot(X, Z.T)
    ZZT = np.dot(Z, Z.T)

    for _ in range(derivative_max_iter):
        # make sure to multiply things in the right order to keep matrix sizes minimal
        G = np.float32(2.0) * (np.dot(A, ZZT) - XZT)  # G has shape N x K
        G = G - np.sum(A * G, axis=1)[:, None]  # chain rule of projection

        # line search for optimal step size
        prev_RSS = _compute_RSS_AZ(X=X, A=A, Z=Z)
        prev_A = A
        for _ in range(20):
            A = (prev_A - mu * G).clip(0)
            A = A / (np.sum(A, axis=1)[:, None] + np.finfo(np.float32).eps)  # Avoid division by zero
            RSS = _compute_RSS_AZ(X=X, A=A, Z=Z)
            if RSS <= (prev_RSS * (1 + rel_tol_ls)):
                mu *= np.float32(1.2)
                break
            else:
                mu *= np.float32(0.5)

                if mu < min_mu:
                    # Use the current A (even if not optimal) or revert to prev_A
                    # depending on which gives better RSS
                    if RSS > prev_RSS:
                        A = prev_A
                    break

        # check for convergence
        if (np.abs(prev_RSS - RSS) / (prev_RSS + 1e-9)) < rel_tol_conv:
            break

    A = np.ascontiguousarray(A)  # TODO: check if we need this line
    return A


def _compute_B_projected_gradients(
    X: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    WX: np.ndarray = None,  # type: ignore[assignment]
    derivative_max_iter: int | np.int32 = 40,
    rel_tol_ls: float | np.float32 = 1e-3,
    rel_tol_conv: float | np.float32 = 1e-4,
) -> np.ndarray:
    """Updates the B matrix given the data matrix X and the A matrix.

    Parameters
    ----------
    X : numpy 2d-array
        Data matrix with shape (n_samples, n_features).

    A : numpy 2d-array
        A matrix with shape (n_samples, n_archetypes).

    B : numpy 2d-array
        B matrix with shape (n_archetypes, n_samples).

    WX : numpy 2d-array, default None
        Weighted data matrix with shape (n_samples, n_features).

    derivative_max_iter: int
        Maximum number of steps for optimization

    Returns
    -------
    B : numpy 2d-array
        Updated B matrix with shape (n_archetypes, n_samples).
    """
    # checks
    assert (rel_tol_ls >= 0) and (rel_tol_conv >= 0)

    # ensure correct data type for parameters
    derivative_max_iter = np.int32(derivative_max_iter)
    rel_tol_ls = np.float32(rel_tol_ls)
    rel_tol_conv = np.float32(rel_tol_conv)

    # works for weighted and unweighted version
    if WX is None:
        WX = X
    else:
        assert (WX.shape[0] == X.shape[0]) and (WX.shape[1] == X.shape[1])

    # optimize B
    B = _compute_B_projected_gradients_jit(
        X=X, A=A, B=B, WX=WX, derivative_max_iter=derivative_max_iter, rel_tol_ls=rel_tol_ls, rel_tol_conv=rel_tol_conv
    )
    return B


@jit(
    [float32[:, :](float32[:, :], float32[:, :], float32[:, :], float32[:, :], int32, float32, float32)],
    nopython=True,
    cache=True,
    fastmath=True,
)  # pragma: no cover
def _compute_B_projected_gradients_jit(
    X: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    WX: np.ndarray,
    derivative_max_iter: np.int32,
    rel_tol_ls: np.float32,
    rel_tol_conv: np.float32,
) -> np.ndarray:
    # initialize learning rates
    mu = np.float32(1.0)
    min_mu = np.float32(1e-6)

    # explicitly making sure everything is contiguous (C-order)
    X = np.ascontiguousarray(X)
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    WX = np.ascontiguousarray(WX)

    # terms that can be pre-computed
    AT_WX_XT = np.dot(np.dot(A.T, WX), X.T)
    AT_A = np.dot(A.T, A)

    for _ in range(derivative_max_iter):
        # make sure to multiply things in the right order to keep matrix sizes minimal
        G = np.float32(2.0) * (np.dot(np.dot(AT_A, np.dot(B, X)), X.T) - AT_WX_XT)  # G has shape K x N
        G = G - np.sum(B * G, axis=1)[:, None]  # chain rule of projection

        # line search for optimal step size
        prev_RSS = _compute_RSS_ABX(X=X, A=A, B=B, WX=WX)
        prev_B = B
        for _ in range(20):
            B = (prev_B - mu * G).clip(0)
            B = B / (np.sum(B, axis=1)[:, None] + np.finfo(np.float32).eps)  # avoid division by zero
            RSS = _compute_RSS_ABX(X=X, A=A, B=B, WX=WX)
            if RSS <= (prev_RSS * (1 + rel_tol_ls)):
                mu *= np.float32(1.2)
                break
            else:
                mu *= np.float32(0.5)

                if mu < min_mu:
                    # Use the current B (even if not optimal) or revert to prev_B
                    # depending on which gives better RSS
                    if RSS > prev_RSS:
                        B = prev_B
                    break

        # check for convergence
        if (np.abs(prev_RSS - RSS) / (prev_RSS + 1e-9)) < rel_tol_conv:
            break

    B = np.ascontiguousarray(B)  # TODO: check if we need this line
    return B


@jit(nopython=True, cache=True, fastmath=True)  # pragma: no cover
def _add_argmins_per_row(mtx, argmins, mu):
    for idx in range(len(argmins)):
        mtx[idx, argmins[idx]] += mu
    return mtx


# NOTE: can lead to kernel crashed in Jupyter notebooks...
@jit(nopython=True, cache=True, parallel=True)  # pragma: no cover
def _add_argmins_per_row_p(mtx, argmins, mu):
    for idx in prange(len(argmins)):
        mtx[idx, argmins[idx]] += mu
    return mtx


def _compute_A_frank_wolfe(
    X: np.ndarray,
    Z: np.ndarray,
    A: np.ndarray,
    derivative_max_iter: int | np.int32 = 80,
    rel_tol_ls: float | np.float32 = 1e-3,
    rel_tol_conv: float | np.float32 = 1e-4,
) -> np.ndarray:
    # check and other things
    assert (rel_tol_ls >= 0) and (rel_tol_conv >= 0)

    # ensure correct data type for parameters
    derivative_max_iter = np.int32(derivative_max_iter)
    rel_tol_ls = np.float32(rel_tol_ls)
    rel_tol_conv = np.float32(rel_tol_conv)

    # optimize A
    A = _compute_A_frank_wolfe_jit(
        X=X, Z=Z, A=A, derivative_max_iter=derivative_max_iter, rel_tol_ls=rel_tol_ls, rel_tol_conv=rel_tol_conv
    )
    return A


@jit(
    [float32[:, :](float32[:, :], float32[:, :], float32[:, :], int32, float32, float32)],
    nopython=True,
    cache=True,
    fastmath=True,
)  # pragma: no cover
def _compute_A_frank_wolfe_jit(
    X: np.ndarray,
    Z: np.ndarray,
    A: np.ndarray,
    derivative_max_iter: np.int32,
    rel_tol_ls: np.float32,
    rel_tol_conv: np.float32,
) -> np.ndarray:
    # explicitly making sure everything is contiguous (C-roder)
    X = np.ascontiguousarray(X)
    Z = np.ascontiguousarray(Z)
    A = np.ascontiguousarray(A)

    # terms that can be pre-computed
    XZT = np.dot(X, Z.T)
    ZZT = np.dot(Z, Z.T)

    for iter in range(derivative_max_iter):
        # frank wolfe step size
        mu = np.float32(2 / (iter + 2))

        # compute the gradient
        G = np.float32(2.0) * (np.dot(A, ZZT) - XZT)  # G has shape N x K

        # for each sample, get the archetype column with the most negative gradient
        argmins = np.argmin(G, axis=1)

        # line search
        prev_RSS = _compute_RSS_AZ(X=X, A=A, Z=Z)
        prev_A = A
        for _ in range(20):
            A = (np.float32(1.0) - mu) * prev_A
            A = _add_argmins_per_row(mtx=A, argmins=argmins, mu=mu)
            RSS = _compute_RSS_AZ(X=X, A=A, Z=Z)
            if RSS <= (prev_RSS * (1 + rel_tol_ls)):
                mu *= np.float32(1.2)
                break
            else:
                mu *= np.float32(0.5)

        # check for convergence
        if (np.abs(prev_RSS - RSS) / (prev_RSS + 1e-9)) < rel_tol_conv:
            break

    return A


def _compute_B_frank_wolfe(
    X: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    WX: np.ndarray = None,  # type: ignore[assignment]
    derivative_max_iter: int | np.int32 = 80,
    rel_tol_ls: float | np.float32 = 1e-3,
    rel_tol_conv: float | np.float32 = 1e-4,
) -> np.ndarray:
    # checks
    assert (rel_tol_ls >= 0) and (rel_tol_conv >= 0)

    # ensure correct data type for parameters
    derivative_max_iter = np.int32(derivative_max_iter)
    rel_tol_ls = np.float32(rel_tol_ls)
    rel_tol_conv = np.float32(rel_tol_conv)

    # works for weighted and unweighted version
    if WX is None:
        WX = X
    else:
        assert (WX.shape[0] == X.shape[0]) and (WX.shape[1] == X.shape[1])

    # optimize B
    B = _compute_B_frank_wolfe_jit(
        X=X, A=A, B=B, WX=WX, derivative_max_iter=derivative_max_iter, rel_tol_ls=rel_tol_ls, rel_tol_conv=rel_tol_conv
    )
    return B


@jit(
    [float32[:, :](float32[:, :], float32[:, :], float32[:, :], float32[:, :], int32, float32, float32)],
    nopython=True,
    cache=True,
    fastmath=True,
)  # pragma: no cover
def _compute_B_frank_wolfe_jit(
    X: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    WX: np.ndarray,
    derivative_max_iter: np.int32,
    rel_tol_ls: np.float32,
    rel_tol_conv: np.float32,
) -> np.ndarray:
    # explicitly making sure everything is contiguous (C-order)
    X = np.ascontiguousarray(X)
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    WX = np.ascontiguousarray(WX)

    # terms that can be pre-computed
    AT_WX_XT = np.dot(np.dot(A.T, WX), X.T)
    AT_A = np.dot(A.T, A)

    for iter in range(derivative_max_iter):
        # frank wolfe step size
        mu = np.float32(2 / (iter + 2))

        # compute the gradient
        G = np.float32(2.0) * (np.dot(np.dot(AT_A, np.dot(B, X)), X.T) - AT_WX_XT)  # G has shape K x N

        # for each archetype, get the sample column with the most negative gradient
        argmins = np.argmin(G, axis=1)

        # line search for optimal step size
        prev_RSS = _compute_RSS_ABX(X=X, A=A, B=B, WX=WX)
        prev_B = B
        for _ in range(20):
            B = (np.float32(1.0) - mu) * prev_B
            B = _add_argmins_per_row(mtx=B, argmins=argmins, mu=mu)
            RSS = _compute_RSS_ABX(X=X, A=A, B=B, WX=WX)
            if RSS <= (prev_RSS * (1 + rel_tol_ls)):
                mu *= np.float32(1.2)
                break
            else:
                mu *= np.float32(0.5)

        # check for convergence
        if (np.abs(prev_RSS - RSS) / (prev_RSS + 1e-9)) < rel_tol_conv:
            break

    return B
