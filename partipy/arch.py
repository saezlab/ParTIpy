"""
Class for archetypal analysis

Note: notation used X ≈ A B X = A Z

Code adapted from https://github.com/atmguille/archetypal-analysis (by Guillermo García Cobo)
"""

import numpy as np

from .const import (
    DEFAULT_INIT,
    DEFAULT_OPTIM,
    DEFAULT_WEIGHT,
    INIT_ALGS,
    MIN_ITERATIONS,
    OPTIM_ALGS,
    WEIGHT_ALGS,
    WEIGHTING_FLAVORS,
)
from .initialize import _init_furthest_sum, _init_plus_plus, _init_uniform
from .optim import (
    _compute_A_frank_wolfe,
    _compute_A_projected_gradients,
    _compute_A_regularized_nnls,
    _compute_B_frank_wolfe,
    _compute_B_projected_gradients,
    _compute_B_regularized_nnls,
)
from .weights import compute_bisquare_weights, compute_huber_weights


class AA:
    """
    Archetypal Analysis approximates data points as a convex combination of a set of archetypes, which are themselves convex combinations of the data points.
    The goal is to find the best approximation for a given number of archetypes, representing the structure of the data in a lower-dimensional space.

    The model is defined as follows:
        X ≈ A B X = A Z

    where:
        - X is the data point matrix.
        - A is the coefficient matrix mapping each data point to a convex combination of archetypes.
        - B is the coefficient matrix mapping each archetype to a convex combination of data points.
        - Z = B X is the matrix containing the archetypes coordinates.

    The optimization problem minimalizes the residual sum of squares (RSS)
        RSS = ||X - A Z||^2
    subject to the constraints that A and B are non-negative and their rows sum to 1, ensuring convex combinations.

    Parameters
    ----------
    n_archetypes : int
        Number of archetypes to compute.
    init : str, optional (default="furthest_sum)
        Initialization method for the archetypes. Options are:
        - "random": Random initialization.
        - "furthest_sum": Utilizes the furthest sum algorithm (recommended).
    optim: str, optional (default="projected_gradients")
        Optimization algorithm to use. Options are:
        - "regularized_nnls": Regularized non-negative least squares.
        - "projected_gradients": Projected gradient descent (PCHA).
        - "frank_wolfe": Frank-Wolfe algorithm.
    weight : str or None, optional (default: None)
        Weighting scheme for robust archetypal analysis. Options:
        - None: No weighting.
        - "bisquare": Bisquare weighting.
    max_iter : int, optional (default: 500)
        Maximum number of iterations for the optimization.
    tol : float, optional (default: 1e-6)
        Tolerance for convergence. The optimization stops if the relative change in RSS
        falls below this threshold.
    verbose : bool, optional (default: False)
        If True, print progress during optimization.
    seed : int, optional (default: 42)
        Random seed
    optim_kwargs : arguments that are passed to compute_A and compute_B
    """

    def __init__(
        self,
        n_archetypes: int,
        init: str = DEFAULT_INIT,
        optim: str = DEFAULT_OPTIM,
        weight: None | str = DEFAULT_WEIGHT,
        max_iter: int = 500,
        rel_tol: float = 1e-4,  # TODO: Which relative tolerance should we be using?
        early_stopping: bool = True,
        verbose: bool = False,
        seed: int = 42,
        **optim_kwargs,
    ):
        self.n_archetypes = n_archetypes
        self.init = init
        self.optim = optim
        self.weight = weight
        self.weighting_flavor = "mair_brefeld_2019"  # NOTE: harcoded for now
        self.max_iter = max_iter
        self.rel_tol = rel_tol
        self.early_stopping = early_stopping
        self.abs_tol: float = None  # type: ignore[assignment]
        self.verbose = verbose
        self.seed = seed
        self.optim_kwargs = optim_kwargs
        # NOTE: I don't want to use here type annotation np.ndarray: None | np.ndarray
        # because it makes little sense for downstream type checking
        self.A: np.ndarray = None  # type: ignore[assignment]
        self.B: np.ndarray = None  # type: ignore[assignment]
        self.Z: np.ndarray = None  # type: ignore[assignment]
        self.n_samples: int = None  # type: ignore[assignment]
        self.n_features: int = None  # type: ignore[assignment]
        self.RSS: float | None = None
        self.RSS_trace: np.ndarray = np.zeros(max_iter, dtype=np.float32)
        self.varexpl: float = None  # type: ignore[assignment]
        self.fitting_info: dict

        # checks
        if self.init not in INIT_ALGS:
            raise ValueError(f"Initialization method '{self.init}' is not supported. Must be one of {INIT_ALGS}.")

        if self.optim not in OPTIM_ALGS:
            raise ValueError(f"Optimization algorithm '{self.optim}' is not supported. Must be one of {OPTIM_ALGS}.")

        if self.weight not in WEIGHT_ALGS:
            raise ValueError(f"Weighting method '{self.weight}' is not supported. Must be one of {WEIGHT_ALGS}.")

        if self.weighting_flavor not in WEIGHTING_FLAVORS:
            raise ValueError(
                f"Weighting method '{self.weighting_flavor}' is not supported. Must be one of {WEIGHTING_FLAVORS}."
            )

        if self.max_iter < 0:
            raise ValueError(f"max_iter must be non-negative, got {self.max_iter}.")

        if self.weight is not None and early_stopping is not False:
            raise ValueError(
                "Early stopping must be disabled (early_stopping=False) when using weighted/robust"
                "archetypal analysis. This is because optimization with weights does not lead to RSS reduction"
            )

    def fit(self, X: np.ndarray):
        """
        Computes the archetypes and the RSS from the data X, which are stored
        in the corresponding attributes.

        Parameters
        ----------
        X : np.ndarray
            Data matrix with shape (n_samples, n_features).

        Returns
        -------
        self : AA
            The instance of the AA class, with computed archetypes and RSS stored as attributes.
        """
        self.n_samples, self.n_features = X.shape

        # ensure C-contiguous format for numba
        X = np.ascontiguousarray(X, dtype=np.float32)

        # center X by substracting the feature means, scale X by dividing by the feature stds
        # feature_means = X.mean(axis=0, keepdims=True)
        # feature_stds = X.std(axis=0, keepdims=True)
        # X -= feature_means
        # X /= feature_stds

        # set the initalization function
        if self.init == "uniform":
            initialize_B = _init_uniform
        elif self.init == "furthest_sum":
            initialize_B = _init_furthest_sum
        elif self.init == "plus_plus":
            initialize_B = _init_plus_plus
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
            elif self.weight == "huber":
                compute_weights = compute_huber_weights
            else:
                raise NotImplementedError()

        # initialize B and the archetypes Z
        B = initialize_B(X=X, n_archetypes=self.n_archetypes, seed=self.seed)
        Z = B @ X

        # randomly initialize A
        rng = np.random.default_rng(self.seed)  # Use a fixed seed
        A = -np.log(rng.random((self.n_samples, self.n_archetypes), dtype=np.float32))
        A /= np.sum(A, axis=1, keepdims=True)

        TSS = RSS = np.sum(X * X)

        W = np.ones(X.shape[0], dtype=np.float32) if self.weight else None

        convergence_flag = False
        for n_iter in range(self.max_iter):
            if self.weighting_flavor == "eugster_leisch_2011":
                # (W[:, None] * X) is the same as np.diag(W) @ X
                X_w = (W[:, None] * X) if self.weight else X  # type: ignore[arg-type, index]
                A = compute_A(X_w, Z, A, **self.optim_kwargs)
                B = compute_B(X_w, A, B, **self.optim_kwargs)
                Z = B @ X_w
            elif self.weighting_flavor == "mair_brefeld_2019":
                A = compute_A(X, Z, A, **self.optim_kwargs)
                A_w = (np.sqrt(W)[:, None] * A) if self.weight else A  # type: ignore[arg-type, index]
                X_w = (np.sqrt(W)[:, None] * X) if self.weight else X  # type: ignore[arg-type, index]
                B = compute_B(X_w, A_w, B, **self.optim_kwargs)
                Z = B @ X_w
            else:
                raise NotImplementedError()

            # compute residuals using the original data (to recompute the weights)
            A_0 = compute_A(X, Z, A, **self.optim_kwargs) if self.weight else A
            R = X - A_0 @ Z
            W = compute_weights(R) if self.weight else None

            # import matplotlib.pyplot as plt
            # plt.style.use("dark_background")
            # if n_iter in [0, 1, 3, 7, 15, 31]:
            #    # show residual norm
            #    plt.scatter(X[:, 0], X[:, 1], c=np.sum(np.abs(R), axis=1), alpha=1.0)
            #    plt.gca().set_aspect('equal')
            #    plt.title(n_iter)
            #    plt.colorbar()
            #    plt.show()
            #
            #    # show weights
            #    plt.scatter(X[:, 0], X[:, 1], c=W, alpha=1.0)
            #    plt.gca().set_aspect('equal')
            #    plt.title(n_iter)
            #    plt.colorbar()
            #    plt.show()

            # compute RSS and check for convergence
            RSS = np.linalg.norm(R) ** 2
            self.RSS_trace[n_iter] = float(RSS)
            max_window = min(n_iter, 20)
            rel_delta_RSS_mean_last_n = (
                np.mean(
                    (
                        self.RSS_trace[(n_iter - max_window + 1) : (n_iter + 1)]
                        - self.RSS_trace[(n_iter - max_window) : (n_iter)]
                    )
                    / self.RSS_trace[(n_iter - max_window) : (n_iter)]
                )
                if n_iter > 0
                else np.nan
            )
            if self.verbose:
                print(f"\rRSS: {RSS:.6f} | rel_delta_RSS: {rel_delta_RSS_mean_last_n:.6f}", end="", flush=True)
            if np.isnan(RSS) or np.isinf(RSS):
                print("\nWarning: RSS is NaN or Inf. Stopping optimization.")
                break

            if (n_iter >= MIN_ITERATIONS) and self.early_stopping:
                if (rel_delta_RSS_mean_last_n >= 0.0) or (np.abs(rel_delta_RSS_mean_last_n) < self.rel_tol):
                    convergence_flag = True
                    break
        if self.verbose:
            message = (
                f"\nAlgorithm converged after {n_iter} iterations."
                if convergence_flag
                else f"\nAlgorithm did not converge after {n_iter} iterations."
            )
            print(message)

        # re-scale and add back the feature means
        # X *= feature_stds
        # X += feature_means
        # Z = np.dot(B, X)

        # Recalculate A and B using the unweighted and data
        if self.weight:
            A = compute_A(X, Z, A, **self.optim_kwargs)
            B = compute_B(X, A, B, **self.optim_kwargs)
            Z = B @ X
            RSS = np.linalg.norm(X - A @ Z) ** 2

        self.Z = Z
        self.A = A
        self.B = B
        self.RSS = float(RSS)
        self.RSS_trace = self.RSS_trace[self.RSS_trace > 0.0]
        self.varexpl = (TSS - RSS) / TSS
        self.fitting_info = {
            "conv": convergence_flag if self.max_iter > 0 else None,
            "n_iter": n_iter if self.max_iter > 0 else None,
        }
        return self

    def archetypes(self) -> None | np.ndarray:
        """
        Returns the archetypes' matrix.

        Returns
        -------
        np.ndarray or None
            The archetypes matrix with shape (n_archetypes, n_features),
            or None if the archetypes have not been computed yet.
        """
        return self.Z

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the best convex approximation A of X by the archetypes Z.

        Parameters
        ----------
        X : np.ndarray
            Data matrix with shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The matrix A with shape (n_samples, n_archetypes).
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
        """
        Returns the optimized matrices and fitting statistics.

        Returns
        -------
        tuple
            A tuple containing:
            - A : np.ndarray
                Coefficient matrix with shape (n_samples, n_archetypes).
            - B : np.ndarray
                Coefficient matrix with shape (n_archetypes, n_samples).
            - Z : np.ndarray
                Archetype matrix with shape (n_archetypes, n_features).
            - RSS_trace : list[float]
                Residual sum of squares per iteration.
            - varexpl : float
                Variance explained by the model.
        """
        return self.A, self.B, self.Z, self.RSS_trace, self.varexpl
