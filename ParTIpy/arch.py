"""
Class for archetypal analysis

Note: notation used X ≈ A · B · X = A · Z

Code adapted from https://github.com/atmguille/archetypal-analysis (by Guillermo García Cobo)
"""

import numpy as np
import scanpy as sc

from .const import (
    DEFAULT_INIT,
    DEFAULT_OPTIM,
    DEFAULT_WEIGHT,
    INIT_ALGS,
    OPTIM_ALGS,
    WEIGHT_ALGS,
)
from .initialize import _furthest_sum_init, _random_init
from .optim import (
    _compute_A_frank_wolfe,
    _compute_A_projected_gradients,
    _compute_A_regularized_nnls,
    _compute_B_frank_wolfe,
    _compute_B_projected_gradients,
    _compute_B_regularized_nnls,
)
from .weights import compute_bisquare_weights


class AA:
    """
    TODO: Write docstring here
    ...
    """

    def __init__(
        self,
        n_archetypes: int,
        init: str = DEFAULT_INIT,
        optim: str = DEFAULT_OPTIM,
        weight: None | str = DEFAULT_WEIGHT,
        max_iter: int = 100,
        derivative_max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
    ):
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
        self.muA, self.muB = None, None
        self.n_samples, self.n_features = None, None
        self.RSS = None
        self.RSS_trace: list[float] = []
        self.varexpl = None
        self.adata = None

        # checks
        assert self.init in INIT_ALGS
        assert self.optim in OPTIM_ALGS
        assert self.weight in WEIGHT_ALGS

    def fit(self, X: np.ndarray):
        """
        Computes the archetypes and the RSS from the data X, which are stored
        in the corresponding attributes
        :param X: data matrix, with shape (n_samples, n_features)
        :return: self
        """
        if isinstance(X, sc.AnnData):
            if "X_pca" not in X.obsm:
                raise ValueError(
                    "X_pca not in AnnData object. Please use run PCA and set_dimension() to add both to the AnnData object."
                )
            self.adata = X
            X = X.obsm["X_pca"][:, : X.uns["PCs"]]

        self.n_samples, self.n_features = X.shape

        # ensure C-contiguous format for numba
        X = np.ascontiguousarray(X)

        # set the initalization function
        if self.init == "random":
            initialize_B = _random_init
        elif self.init == "furthest_sum":
            initialize_B = _furthest_sum_init
        else:
            raise NotImplementedError()

        # set the optimization functions
        if self.optim == "regularized_nnls":
            compute_A = _compute_A_regularized_nnls
            compute_B = _compute_B_regularized_nnls
        elif self.optim == "projected_gradients":
            compute_A = _compute_A_projected_gradients  # type: ignore[assignment]
            compute_B = _compute_B_projected_gradients  # type: ignore[assignment]
        elif self.optim == "frank_wolfe":
            compute_A = _compute_A_frank_wolfe  # type: ignore[assignment]
            compute_B = _compute_B_frank_wolfe  # type: ignore[assignment]
        else:
            raise NotImplementedError()

        # set the weight function
        if self.weight:
            if self.weight == "bisquare":
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

            RSS = np.linalg.norm(R) ** 2
            if (prev_RSS is not None) and ((abs(prev_RSS - RSS) / prev_RSS) < self.tol):
                break
            prev_RSS = RSS
            self.RSS_trace.append(float(RSS))

        # Recalculate A and B using the unweighted data
        if self.weight:
            A = compute_A(X, Z, A, self.deriv_max_iter)
            B = compute_B(X, A, B, self.deriv_max_iter)
            Z = B @ X
            RSS = np.linalg.norm(X - A @ Z) ** 2

        self.Z = Z
        self.A = A
        self.B = B
        self.RSS = RSS
        self.RSS_trace = np.array(self.RSS_trace)
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
        if self.optim == "regularized_nnls":
            return _compute_A_regularized_nnls(X, self.Z)
        elif self.optim == "projected_gradients":
            A_random = -np.log(np.random.random((self.n_samples, self.n_archetypes)))
            A_random /= np.sum(A_random, axis=1, keepdims=True)
            return _compute_A_projected_gradients(X=X, Z=self.Z, A=A_random)
        elif self.optim == "frank_wolfe":
            A_random = -np.log(np.random.random((self.n_samples, self.n_archetypes)))
            A_random /= np.sum(A_random, axis=1, keepdims=True)
            return _compute_A_frank_wolfe(X, self.Z, A=A_random)
        else:
            raise NotImplementedError()

    def return_all(self) -> tuple:
        """Return optimized matrices: A, B, Z, and fitting stats: RSS, varexpl."""
        return self.A, self.B, self.Z, self.RSS, self.varexpl

    def save_to_anndata(self, archetypes_only: bool = True):
        """
        Saves the results to the AnnData object provided in fit().

        Parameters:
        -----------
        archetypes_only: bool
          If True, only the archetypes (Z) are saved. Otherwise, all results (A, B, Z, RSS, varexpl) are saved.
        """
        if self.adata is None:
            raise ValueError("No AnnData object found. Please provide an AnnData object to fit().")

        if archetypes_only:
            self.adata.uns["archetypal_analysis"] = {
                "Z": self.Z,
            }
        else:
            self.adata.uns["archetypal_analysis"] = {
                "A": self.A,
                "B": self.B,
                "Z": self.Z,
                "RSS": self.RSS,
                "varexpl": self.varexpl,
            }
