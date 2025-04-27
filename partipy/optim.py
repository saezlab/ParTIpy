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
from numba import float32, njit, prange
from scipy.optimize import nnls

from .const import LAMBDA


@njit(cache=True, fastmath=True)
def _compute_RSS(X, A, Z):
    diff = X - (A @ Z)
    return np.sum(diff * diff)  # Faster than norm squared


def _compute_A_regularized_nnls(
    X: np.ndarray,
    Z: np.ndarray,
    A: np.ndarray | None = None,
) -> np.ndarray:
    # huge_constant is added as a new column to account for w norm constraint
    X_padded = np.hstack([X, (LAMBDA * np.ones(X.shape[0]))[:, None]])
    Zt_padded = np.vstack([Z.T, LAMBDA * np.ones(Z.shape[0])])

    # Use non-negative least squares to solve the optimization problem
    A = np.array([nnls(A=Zt_padded, b=X_padded[n, :])[0] for n in range(X.shape[0])])
    A = A.astype(np.float32)
    return A


def _compute_B_regularized_nnls(
    X: np.ndarray,
    A: np.ndarray,
    B: np.ndarray | None = None,
) -> np.ndarray:
    Z = np.linalg.lstsq(a=A, b=X, rcond=None)[0]
    Z_padded = np.hstack([Z, (LAMBDA * np.ones(Z.shape[0]))[:, None]])
    Xt_padded = np.vstack([X.T, LAMBDA * np.ones(X.shape[0])])
    B = np.array([nnls(A=Xt_padded, b=Z_padded[k, :])[0] for k in range(Z.shape[0])])
    B = B.astype(np.float32)
    return B


@njit(cache=True)
def _compute_A_projected_gradients(
    X: float32[:, :],
    Z: float32[:, :],
    A: float32[:, :],
    derivative_max_iter: int = 40,
    rel_tol_ls: float | np.float32 = 1e-3,
    rel_tol_conv: float | np.float32 = 1e-4,
) -> float32[:, :]:
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
    assert (rel_tol_ls >= 0) and (rel_tol_conv >= 0)
    rel_tol_ls = np.float32(rel_tol_ls)
    rel_tol_conv = np.float32(rel_tol_conv)
    mu = np.float32(1.0)
    min_mu = np.float32(1e-6)

    # terms that can be pre-computed
    XZT = np.dot(X, Z.T)
    ZZT = np.dot(Z, Z.T)

    for _ in range(derivative_max_iter):
        # brackets are VERY important to save time
        G = np.float32(2.0) * (np.dot(A, ZZT) - XZT)  # G has shape N x K
        G = G - np.sum(A * G, axis=1)[:, None]  # chain rule of projection

        # line search for optimal step size
        prev_RSS = _compute_RSS(X=X, A=A, Z=Z)
        prev_A = A
        for _ in range(20):
            A = (prev_A - mu * G).clip(0)
            A = A / (np.sum(A, axis=1)[:, None] + np.finfo(np.float32).eps)  # Avoid division by zero
            RSS = _compute_RSS(X=X, A=A, Z=Z)
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
        if (np.abs(prev_RSS - RSS) / prev_RSS) < rel_tol_conv:
            break

    return A


@njit(cache=True)
def _compute_B_projected_gradients(
    X: float32[:, :],
    A: float32[:, :],
    B: float32[:, :],
    derivative_max_iter: int = 40,
    rel_tol_ls: float | np.float32 = 1e-3,
    rel_tol_conv: float | np.float32 = 1e-4,
) -> float32[:, :]:
    """Updates the B matrix given the data matrix X and the A matrix.

    Parameters
    ----------
    X : numpy 2d-array
        Data matrix with shape (n_samples, n_features).

    A : numpy 2d-array
        A matrix with shape (n_samples, n_archetypes).

    B : numpy 2d-array
        B matrix with shape (n_archetypes, n_samples).

    derivative_max_iter: int
        Maximum number of steps for optimization

    Returns
    -------
    B : numpy 2d-array
        Updated B matrix with shape (n_archetypes, n_samples).
    """
    assert (rel_tol_ls >= 0) and (rel_tol_conv >= 0)
    rel_tol_ls = np.float32(rel_tol_ls)
    rel_tol_conv = np.float32(rel_tol_conv)
    mu = np.float32(1.0)
    min_mu = np.float32(1e-6)

    # terms that can be pre-computed
    ATA = np.dot(A.T, A)
    ATXXT = np.dot(np.dot(A.T, X), X.T)

    for _ in range(derivative_max_iter):
        # brackets are VERY important to save time
        G = np.float32(2.0) * (np.dot(np.dot(ATA, np.dot(B, X)), X.T) - ATXXT)  # G has shape K x N
        G = G - np.sum(B * G, axis=1)[:, None]  # chain rule of projection

        # line search for optimal step size
        prev_RSS = np.linalg.norm(X - A @ (B @ X)) ** 2
        prev_B = B
        for _ in range(20):
            B = (prev_B - mu * G).clip(0)
            B = B / (np.sum(B, axis=1)[:, None] + np.finfo(np.float32).eps)  # Avoid division by zero
            RSS = np.linalg.norm(X - A @ (B @ X)) ** 2
            if RSS <= (prev_RSS * (1 + rel_tol_ls)):
                mu *= np.float32(1.2)
                break
            else:
                mu *= np.float32(0.5)

                if mu < min_mu:
                    # Use the current A (even if not optimal) or revert to prev_A
                    # depending on which gives better RSS
                    if RSS > prev_RSS:
                        B = prev_B
                    break

        # check for convergence
        if (np.abs(prev_RSS - RSS) / prev_RSS) < rel_tol_conv:
            break

    return B


@njit(cache=True)
def _add_argmins_per_row(mtx, argmins, mu):
    for idx in range(len(argmins)):
        mtx[idx, argmins[idx]] += mu
    return mtx


# NOTE: can lead to kernel crashed in Jupyter notebooks...
@njit(cache=True, parallel=True)
def _add_argmins_per_row_p(mtx, argmins, mu):
    for idx in prange(len(argmins)):
        mtx[idx, argmins[idx]] += mu
    return mtx


@njit(cache=True)
def _compute_A_frank_wolfe(
    X: float32[:, :],
    Z: float32[:, :],
    A: float32[:, :],
    derivative_max_iter: int = 80,
    rel_tol_ls: float | np.float32 = 1e-3,
    rel_tol_conv: float | np.float32 = 1e-4,
) -> float32[:, :]:
    assert (rel_tol_ls >= 0) and (rel_tol_conv >= 0)
    rel_tol_ls = np.float32(rel_tol_ls)
    rel_tol_conv = np.float32(rel_tol_conv)

    # terms that can be pre-computed
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
        prev_RSS = _compute_RSS(X=X, A=A, Z=Z)
        prev_A = A
        for _ in range(20):
            A = (np.float32(1.0) - mu) * prev_A
            A = _add_argmins_per_row(mtx=A, argmins=argmins, mu=mu)
            RSS = _compute_RSS(X=X, A=A, Z=Z)
            if RSS <= (prev_RSS * (1 + rel_tol_ls)):
                mu *= np.float32(1.2)
                break
            else:
                mu *= np.float32(0.5)

        # check for convergence
        if (np.abs(prev_RSS - RSS) / prev_RSS) < rel_tol_conv:
            break

    return A


@njit(cache=True)
def _compute_B_frank_wolfe(
    X: float32[:, :],
    A: float32[:, :],
    B: float32[:, :],
    derivative_max_iter: int = 80,
    rel_tol_ls: float | np.float32 = 1e-3,
    rel_tol_conv: float | np.float32 = 1e-4,
) -> float32[:, :]:
    assert (rel_tol_ls >= 0) and (rel_tol_conv >= 0)
    rel_tol_ls = np.float32(rel_tol_ls)
    rel_tol_conv = np.float32(rel_tol_conv)

    # terms that can be pre-computed
    ATA = np.dot(A.T, A)
    ATXXT = np.dot(np.dot(A.T, X), X.T)

    for iter in range(derivative_max_iter):
        # frank wolfe step size
        mu = np.float32(2 / (iter + 2))

        # compute the gradient
        G = np.float32(2.0) * (np.dot(np.dot(ATA, np.dot(B, X)), X.T) - ATXXT)  # G has shape K x N

        # for each archetype, get the sample column with the most negative gradient
        argmins = np.argmin(G, axis=1)

        # line search for optimal step size
        prev_RSS = np.linalg.norm(X - A @ (B @ X)) ** 2
        prev_B = B
        for _ in range(20):
            B = (np.float32(1.0) - mu) * prev_B
            B = _add_argmins_per_row(mtx=B, argmins=argmins, mu=mu)
            RSS = np.linalg.norm(X - A @ (B @ X)) ** 2
            if RSS <= (prev_RSS * (1 + rel_tol_ls)):
                mu *= np.float32(1.2)
                break
            else:
                mu *= np.float32(0.5)

        # check for convergence
        if (np.abs(prev_RSS - RSS) / prev_RSS) < rel_tol_conv:
            break

    return B


def _compute_A_frank_wolfe_depr(
    X: np.ndarray,
    Z: np.ndarray,
    A: np.ndarray,
    derivative_max_iter: int = 80,
    rel_tol_ls: float | np.float32 = 1e-3,
    rel_tol_conv: float | np.float32 = 1e-4,
) -> np.ndarray:
    n_samples = X.shape[0]
    e = np.zeros(A.shape, dtype=np.float32)

    assert (rel_tol_ls >= 0) and (rel_tol_conv >= 0)
    rel_tol_ls = np.float32(rel_tol_ls)
    rel_tol_conv = np.float32(rel_tol_conv)

    for iter in range(derivative_max_iter):
        # frank wolfe step size
        mu = np.float32(2 / (iter + 2))

        # compute the gradient
        G = np.float32(2.0) * (A @ (Z @ Z.T) - X @ Z.T)

        # for each sample, get the archetype column with the most negative gradient
        argmins = np.argmin(G, axis=1)

        # set the indicator matrix e
        e[range(n_samples), argmins] = 1.0

        # line search
        prev_RSS = np.linalg.norm(X - (A @ Z)) ** 2
        prev_A = A
        for _ in range(1_000):
            A = prev_A + mu * (e - prev_A)
            RSS = np.linalg.norm(X - (A @ Z)) ** 2
            if RSS <= (prev_RSS * (1 + rel_tol_ls)):
                mu *= np.float32(1.2)
                break
            else:
                mu *= np.float32(0.5)

        # check for convergence
        if (np.abs(prev_RSS - RSS) / prev_RSS) < rel_tol_conv:
            break

        # Reset e
        e[range(n_samples), argmins] = 0.0

    return A


def _compute_B_frank_wolfe_depr(
    X: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    derivative_max_iter: int = 80,
    rel_tol_ls: float | np.float32 = 1e-3,
    rel_tol_conv: float | np.float32 = 1e-4,
) -> np.ndarray:
    n_archetypes = A.shape[1]
    e = np.zeros(B.shape, dtype=np.float32)

    assert (rel_tol_ls >= 0) and (rel_tol_conv >= 0)
    rel_tol_ls = np.float32(rel_tol_ls)
    rel_tol_conv = np.float32(rel_tol_conv)

    for iter in range(derivative_max_iter):
        # frank wolfe step size
        mu = np.float32(2 / (iter + 2))

        # compute the gradient
        G = np.float32(2.0) * (((A.T @ A) @ (B @ X) @ X.T) - ((A.T @ X) @ X.T))

        # for each archetype, get the sample column with the most negative gradient
        argmins = np.argmin(G, axis=1)

        # set the indicator matrix e
        e[range(n_archetypes), argmins] = 1.0

        # line search for optimal step size
        prev_RSS = np.linalg.norm(X - A @ (B @ X)) ** 2
        prev_B = B
        for _ in range(1_000):
            B = prev_B + mu * (e - prev_B)
            RSS = np.linalg.norm(X - A @ (B @ X)) ** 2
            if RSS <= (prev_RSS * (1 + rel_tol_ls)):
                mu *= np.float32(1.2)
                break
            else:
                mu *= np.float32(0.5)

        # check for convergence
        if (np.abs(prev_RSS - RSS) / prev_RSS) < rel_tol_conv:
            break

        # reset e
        e[range(n_archetypes), argmins] = 0.0

    return B
