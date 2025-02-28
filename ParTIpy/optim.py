"""
Optimize the archetypal analysis objective by block coordiante descent.

a) Regularized Nonnegative Least Squares
- Paper: A. Cutler and L. Breiman, “Archetypal analysis,” Technometrics, vol. 36, no. 4, pp. 338–347, 1994, doi: 10.1080/00401706.1994.10485840.


b) Projected Gradients (PCHA)
- Paper: M. Mørup and L. K. Hansen, “Archetypal analysis for machine learning and data mining,” Neurocomputing, vol. 80, pp. 54–63, Mar. 2012, doi: 10.1016/j.neucom.2011.06.033.


c) Adapted Frank-Wolfe algorithm
- Paper: C. Bauckhage, K. Kersting, F. Hoppe, and C. Thurau, “Archetypal analysis as an autoencoder,” presented at the Workshop “New Challenges in Neural Computation” (NC2) 2015, 2015. Accessed: Feb. 10, 2025. [Online]. Available: https://publica.fraunhofer.de/handle/publica/393337


Code adapted from
a) https://github.com/nichohelmut/football_results/blob/master/clustering/clustering.py
b) https://github.com/atmguille/archetypal-analysis (by Guillermo García Cobo)
"""

import scipy.optimize
import numpy as np
from numba import njit

from .const import LAMBDA


def compute_A_regularized_nnls(
    X: np.ndarray,
    Z: np.ndarray,
    A: np.ndarray = None,
    derivative_max_iter=None,
) -> np.ndarray:
    # huge_constant is added as a new column to account for w norm constraint
    X_padded = np.hstack([X, (LAMBDA * np.ones(X.shape[0]))[:, None]])
    Zt_padded = np.vstack([Z.T, LAMBDA * np.ones(Z.shape[0])])

    # Use non-negative least squares to solve the optimization problem
    A = np.array(
        [
            scipy.optimize.nnls(A=Zt_padded, b=X_padded[n, :])[0]
            for n in range(X.shape[0])
        ]
    )
    return A


def compute_B_regularized_nnls(
    X: np.ndarray,
    A: np.ndarray,
    B: np.ndarray = None,
    derivative_max_iter=None,
) -> np.ndarray:
    Z = np.linalg.lstsq(a=A, b=X, rcond=None)[0]
    Z_padded = np.hstack([Z, (LAMBDA * np.ones(Z.shape[0]))[:, None]])
    Xt_padded = np.vstack([X.T, LAMBDA * np.ones(X.shape[0])])
    B = np.array(
        [
            scipy.optimize.nnls(A=Xt_padded, b=Z_padded[k, :])[0]
            for k in range(Z.shape[0])
        ]
    )
    return B


# @njit(cache=True)
def compute_A_projected_gradients(
    X: np.ndarray,
    Z: np.ndarray,
    A: np.ndarray,
    derivative_max_iter: int = 10,
    muA: float = 1.0,
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
    rel_tol = 1e-6
    prev_RSS = np.linalg.norm(X - A @ Z) ** 2
    for _ in range(derivative_max_iter):
        # brackets are VERY important to save time
        # [G] ~ N x K
        G = 2.0 * (A @ (Z @ Z.T) - X @ Z.T)
        G = (
            G - np.sum(A * G, axis=1)[:, None]
        )  # chain rule of projection, check broadcasting

        prev_A = A
        # NOTE: original implementation has a while True
        for _ in range(derivative_max_iter * 100):
            A = (prev_A - muA * G).clip(0)
            A = A / (
                np.sum(A, axis=1)[:, None] + np.finfo(np.float64).eps
            )  # Avoid division by zero
            RSS = np.linalg.norm(X - A @ Z) ** 2
            if RSS <= (prev_RSS * (1 + rel_tol)):
                muA *= 1.2
                break
            else:
                muA /= 2.0
    return A


# @njit(cache=True)
def compute_B_projected_gradients(
    X: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    derivative_max_iter: int = 10,
    muB: float = 1.0,
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

    derivative_max_iter: int
        Maximum number of steps for optimization

    Returns
    -------
    B : numpy 2d-array
        Updated B matrix with shape (n_archetypes, n_samples).
    """
    rel_tol = 1e-6
    prev_RSS = np.linalg.norm(X - A @ (B @ X)) ** 2
    for _ in range(derivative_max_iter):
        # brackets are VERY important to save time
        # [G] ~ K x N
        G = 2.0 * (((A.T @ A) @ (B @ X) @ X.T) - ((A.T @ X) @ X.T))
        G = (
            G - np.sum(B * G, axis=1)[:, None]
        )  # chain rule of projection, check broadcasting

        prev_B = B
        # NOTE: original implementation has a while True
        for _ in range(derivative_max_iter * 100):
            B = (prev_B - muB * G).clip(0)
            B = B / (
                np.sum(B, axis=1)[:, None] + np.finfo(np.float64).eps
            )  # Avoid division by zero
            RSS = np.linalg.norm(X - A @ (B @ X)) ** 2
            if RSS <= (prev_RSS * (1 + rel_tol)):
                muB *= 1.2
                break
            else:
                muB /= 2.0
    return B


# @njit(cache=True)
def compute_A_frank_wolfe(
    X: np.ndarray,
    Z: np.ndarray,
    A: np.ndarray = None,
    derivative_max_iter: int = 10,
) -> np.ndarray:
    n_samples, n_archetypes = X.shape[0], Z.shape[0]
    # TODO: why do we initalize A here all new?
    A = np.zeros((n_samples, n_archetypes))
    # TODO: why do we set here the first column of A to 1.0?
    # is this just a simplest way to create a matrix A that satisfies the constraints?
    A[:, 0] = 1.0
    e = np.zeros(A.shape)
    for t in range(derivative_max_iter):
        # compute the gradient wrt A (brackets are VERY important to save time)
        G = 2.0 * (A @ (Z @ Z.T) - X @ Z.T)

        # For each sample, get the archetype column with the most negative gradient
        argmins = np.argmin(G, axis=1)

        # set our indicator matrix e
        e[range(n_samples), argmins] = 1.0
        # for idx in prange(n_samples):
        #    e[idx, argmins[idx]] = 1.0

        # now where does this update rule come from?
        A += (2.0 / (t + 2.0)) * (e - A)

        # reset e
        e[range(n_samples), argmins] = 0.0
        # for idx in prange(n_samples):
        #    e[idx, argmins[idx]] = 0.0
    return A


# @njit(cache=True)
def compute_B_frank_wolfe(
    X: np.ndarray,
    A: np.ndarray,
    B=None,
    derivative_max_iter: int = 10,
) -> np.ndarray:
    n_samples, n_archetypes = X.shape[0], A.shape[1]
    B = np.zeros((n_archetypes, n_samples))
    # TODO: why do we set here the first column of B to 1.0?
    # is this just a simplest way to create a matrix B that satisfies the constraints?
    B[:, 0] = 1.0
    e = np.zeros(B.shape)
    for t in range(derivative_max_iter):
        # compute the gradient wrt B (brackets are VERY important to save time)
        G = 2.0 * ((A.T @ A) @ (B @ X) @ X.T - (A.T @ X) @ X.T)

        # For each archetype, get the sample column with the most negative gradient
        argmins = np.argmin(G, axis=1)

        # set our indicator matrix e
        e[range(n_archetypes), argmins] = 1.0
        # for idx in prange(n_archetypes):
        #    e[idx, argmins[idx]] = 1.0

        # now where does this update rule come from?
        B += (2.0 / (t + 2.0)) * (e - B)

        # reset e
        e[range(n_archetypes), argmins] = 0.0
        # for idx in prange(n_archetypes):
        #    e[idx, argmins[idx]] = 0.0
    return B
