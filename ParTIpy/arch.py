"""
Class for archetypal analysis

Note: notation used X ≈ A · B · X = A · Z

Code adapted from https://github.com/atmguille/archetypal-analysis (by Guillermo García Cobo)
"""

from typing import Union

import numpy as np


class AA(object):

    def __init__(self, 
                 n_archetypes: int, 
                 init: str = "furthest_sum", 
                 optim: str = "projected_gradients",
                 weight: Union[None, str] = None,
                 max_iter: int = 100, 
                 derivative_max_iter: int = 100,
                 tol: float = 1e-6, 
                 verbose: bool = False):
        self.n_archetypes = n_archetypes
        self.init = init
        self.optim = optim
        self.weight = weight
        self.max_iter = max_iter
        self.deriv_max_iter = derivative_max_iter
        self.tol = tol
        self.verbose = verbose
        self.A = None
        self.B = None
        self.Z = None  # Archetypes
        self.n_samples, self.n_features = None, None
        self.RSS = None
        self.varexpl = None


    def fit(self, X: np.ndarray):
        """
        Computes the archetypes and the RSS from the data X, which are stored
        in the corresponding attributes
        :param X: data matrix, with shape (n_samples, n_features)
        :return: self
        """
        self.n_samples, self.n_features = X.shape

        # ensure C-contiguous format for numba
        X = np.ascontiguousarray(X)

        # set the initalization function
        if self.init == "random":
            from .init import random_init
            initialize_B = random_init
        elif self.init == "furthest_sum":
            from .init import furthest_sum_init
            initialize_B = furthest_sum_init
        else:
            raise NotImplementedError()

        # set the optimization functions
        if self.optim == "regularized_nnls":
            from .optim import compute_A_regularized_nnls, compute_B_regularized_nnls
            compute_A = compute_A_regularized_nnls
            compute_B = compute_B_regularized_nnls
        elif self.optim == "projected_gradients":
            from .optim import compute_A_projected_gradients, compute_B_projected_gradients
            compute_A = compute_A_projected_gradients
            compute_B = compute_B_projected_gradients
        elif self.optim == "frank_wolfe":
            from .optim import compute_A_frank_wolfe, compute_B_frank_wolfe
            compute_A = compute_A_frank_wolfe
            compute_B = compute_B_frank_wolfe
        else:
            raise NotImplementedError()

        # set the weight function
        if self.weight:
            if self.weight == "bisquare":
                from .weights import compute_bisquare_weights
                compute_weights = compute_bisquare_weights
            else:
                raise NotImplementedError()

        # initialize B and the archetypes Z
        B = initialize_B(X=X, n_archetypes=self.n_archetypes)
        Z = B @ X

        # randomly initialize A
        A = -np.log(np.random.random((self.n_samples, self.n_archetypes)))
        A /= np.sum(A, axis=1, keepdims=True)

        TSS = np.sum(X * X)
        prev_RSS = None

        W = np.ones(X.shape[0]) if self.weight else None

        for _ in range(self.max_iter):

            X_w = np.diag(W) @ X if self.weight else X
            A = compute_A(X_w, Z, A, self.deriv_max_iter)
            B = compute_B(X_w, A, B, self.deriv_max_iter)
            Z = B @ X_w

            # compute residuals using the original data
            A_0 = compute_A(X, Z, A, self.deriv_max_iter) if self.weight else A
            R = X - A_0 @ Z
            W = compute_weights(R) if self.weight else None

            RSS = np.linalg.norm(R)**2
            if (prev_RSS is not None) and ((abs(prev_RSS - RSS) / prev_RSS) < self.tol):
                break
            prev_RSS = RSS

        # Recalculate A and B using the unweighted data
        if self.weight:
            A = compute_A(X, Z, A, self.deriv_max_iter)
            B = compute_B(X, A, B, self.deriv_max_iter)
            Z = B @ X
            RSS = np.linalg.norm(X - A @ Z)**2

        self.Z = Z
        self.A = A
        self.B = B
        self.RSS = RSS
        self.varexpl = (TSS - RSS) / TSS
        return self


    def archetypes(self) -> np.ndarray:
        """
        Returns the archetypes' matrix
        :return: archetypes matrix, with shape (n_archetypes, n_features)
        """
        return self.Z


    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the best convex approximation A of X by the archetypes Z
        :param X: data matrix, with shape (n_samples, n_features)
        :return: A matrix, with shape (n_samples, n_archetypes)
        """
        return self._computeA(X, self.Z)


    def return_all(self):
        return self.A, self.B, self.Z, self.RSS, self.varexpl

