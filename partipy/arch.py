"""Class for for fitting archetypal analysis models"""

import numpy as np

from ._docs import docs
from .const import (
    DEFAULT_INIT,
    DEFAULT_MAX_ITER,
    DEFAULT_OPTIM,
    DEFAULT_REL_TOL,
    DEFAULT_WEIGHT,
    INIT_ALGS,
    MIN_ITERATIONS,
    OPTIM_ALGS,
    WEIGHT_ALGS,
)
from .coreset import construct_coreset, construct_lightweight_coreset, construct_uniform_coreset
from .initialize import _init_A, _init_furthest_sum, _init_plus_plus, _init_uniform
from .optim import (
    _compute_A_frank_wolfe,
    _compute_A_projected_gradients,
    _compute_A_regularized_nnls,
    _compute_B_frank_wolfe,
    _compute_B_projected_gradients,
    _compute_B_regularized_nnls,
    _compute_RSS_AZ,
)
from .weights import compute_bisquare_weights, compute_huber_weights


@docs.dedent
class AA:
    r"""
    Archetypal Analysis approximates data points as a convex combination of a set of archetypes, which are themselves convex combinations of the data points.
    The goal is to find the best approximation for a given number of archetypes, representing the structure of the data in a lower-dimensional space.

    The model is defined as follows:

    .. math::

        X \approx A B X = A Z

    where:
        - X is the data point matrix.
        - A is the coefficient matrix mapping each data point to a convex combination of archetypes.
        - B is the coefficient matrix mapping each archetype to a convex combination of data points.
        - Z = B X is the matrix containing the archetypes coordinates.

    The optimization problem minimalizes the residual sum of squares (RSS) RSS = ||X - A Z||^2
    subject to the constraints that A and B are non-negative and their rows sum to 1, ensuring convex combinations.

    Parameters
    ----------
    n_archetypes : int
        Number of archetypes to compute.
    %(init)s
    %(optim)s
    %(weight)s
    %(max_iter)s
    %(rel_tol)s
    early_stopping : bool, default `True`
        Whether to stop the optimization early if the relative change in RSS is below a certain threshold.
    use_coreset : bool, default `False`
        Whether to use a coreset for the optimization. If True, a coreset is constructed from the data.
    coreset_flavor : {"default", "lightweight_kmeans", "uniform"}, default `default`
        The method used to construct the coreset. Options:

        - "default": Uses the default coreset construction method.
        - "lightweight_kmeans": Uses a lightweight k-means approach to construct the coreset.
        - "uniform": Constructs a uniform coreset.
    coreset_fraction : float, default `0.1`
        Fraction of the data to use for the coreset. Only used if `use_coreset` is True.
    coreset_size : int, default: `None`
        Size of the coreset to use. If None, it is set to `n_samples * coreset_fraction`.
    centering : bool, default `True`
        Whether to center the data by subtracting the feature means before optimization.
    scaling : bool, default `True`
        Whether to scale the data globally by dividing by the global norm before optimization.
    %(verbose)s
    %(seed)s
    optim_kwargs : dict
        Additional arguments that are passed to `compute_A` and `compute_B`.
    """

    def __init__(
        self,
        n_archetypes: int,
        init: str = DEFAULT_INIT,
        optim: str = DEFAULT_OPTIM,
        weight: None | str = DEFAULT_WEIGHT,
        max_iter: int = DEFAULT_MAX_ITER,
        rel_tol: float = DEFAULT_REL_TOL,
        early_stopping: bool = True,
        use_coreset: bool = False,
        coreset_flavor: str = "default",
        coreset_fraction: float = 0.1,
        coreset_size: None | int = None,
        centering: bool = True,
        scaling: bool = True,
        verbose: bool = False,
        seed: int = 42,
        **optim_kwargs,
    ):
        self.n_archetypes = n_archetypes
        self.init = init
        self.optim = optim
        self.weight = weight
        self.max_iter = max_iter
        self.rel_tol = rel_tol
        self.early_stopping = early_stopping
        self.use_coreset = use_coreset
        self.coreset_flavor = coreset_flavor
        self.coreset_fraction = coreset_fraction
        self.coreset_size = coreset_size
        self.centering = centering
        self.scaling = scaling
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

        if self.max_iter < 0:
            raise ValueError(f"max_iter must be non-negative, got {self.max_iter}.")

        if self.weight is not None and early_stopping is not False:
            raise ValueError(
                "Early stopping must be disabled (early_stopping=False) when using weighted/robust"
                "archetypal analysis. This is because optimization with weights does not lead to RSS reduction"
            )

        if self.use_coreset and self.weight:
            raise ValueError(
                "It is not yet implemented to use robust archetypal analysis and coresets at the same time"
            )

    def fit(self, X: np.ndarray):
        """
        Computes the archetypes and the RSS from the data X, which are stored
        in the corresponding attributes.

        Parameters
        ----------
        X : `np.ndarray`
            Data matrix with shape (n_samples, n_features).

        Returns
        -------
        self : AA
            The instance of the AA class, with computed archetypes and RSS stored as attributes.
        """
        self.n_samples, self.n_features = X.shape

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

        # ensure C-contiguous format for numba (plus using np.float32 datatype)
        X = np.ascontiguousarray(
            X, dtype=np.float32
        ).copy()  # if we don't explicitly copy here, multiprocessing can fail

        # center X by substracting the feature means
        if self.centering:
            feature_means = X.mean(axis=0, keepdims=True)
            X -= feature_means

        # scale X globally (needs to happen before we compute weights, otherwise the weights are off)
        # TODO: Test whether we can also just apply the same scaling to the weights
        if self.scaling:
            global_scale = np.linalg.norm(X) / np.sqrt(np.prod(X.shape))
            X /= global_scale

        # keep the raw X
        X_raw = X.copy()

        # construct the coreset and initialize A
        if self.use_coreset:
            if self.coreset_size is None:
                self.coreset_size = int(self.n_samples * self.coreset_fraction)

            if self.coreset_flavor == "default":
                coreset_indices, W = construct_coreset(X=X, coreset_size=self.coreset_size, seed=self.seed)
            elif self.coreset_flavor == "lightweight_kmeans":
                coreset_indices, W = construct_lightweight_coreset(X=X, coreset_size=self.coreset_size, seed=self.seed)
            elif self.coreset_flavor == "uniform":
                coreset_indices, W = construct_uniform_coreset(X=X, coreset_size=self.coreset_size, seed=self.seed)
            else:
                raise NotImplementedError()

            if self.verbose:
                print(f"coreset size = {self.coreset_size} | coreset flavor = {self.coreset_flavor}")

            X = X[coreset_indices, :].copy()  # TODO: probably no copy needed here!
            A = _init_A(n_samples=self.coreset_size, n_archetypes=self.n_archetypes, seed=self.seed)

        else:
            A = _init_A(n_samples=self.n_samples, n_archetypes=self.n_archetypes, seed=self.seed)

        # initialize B and the archetypes Z
        B, inital_indices = initialize_B(X=X, n_archetypes=self.n_archetypes, seed=self.seed, return_indices=True)
        Z = B @ X

        # initialize weights
        if self.weight:
            W = np.ones(X.shape[0], dtype=np.float32)
        elif self.use_coreset:
            # if we use coreset we only have to weight X a single time
            WX = W[:, None] * X  # same as np.diag(W) @ X

        TSS = RSS = np.sum(X * X)

        convergence_flag = False
        for n_iter in range(self.max_iter):
            if self.weight:
                WX = W[:, None] * X
                A = compute_A(WX, Z, A, **self.optim_kwargs)
                B = compute_B(WX, A, B, **self.optim_kwargs)
                Z = B @ WX

                # recompute weights based on the original, which are computed using the original data
                A_0 = compute_A(X, Z, A, **self.optim_kwargs)
                R = X - A_0 @ Z
                W = compute_weights(R)

            elif self.use_coreset:
                # compute A using the unweighted data X
                A = compute_A(X=X, Z=Z, A=A, **self.optim_kwargs)
                WA = W[:, None] * A
                B = compute_B(X=X, A=WA, B=B, WX=WX, **self.optim_kwargs)
                Z = B @ X

            else:
                A = compute_A(X, Z, A, **self.optim_kwargs)
                B = compute_B(X, A, B, **self.optim_kwargs)
                Z = B @ X

            # compute RSS and check for convergence
            RSS = _compute_RSS_AZ(X=X, A=A, Z=Z)
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
                print(
                    f"\riter: {n_iter} | RSS: {RSS:.3f} | rel_delta_RSS: {rel_delta_RSS_mean_last_n:.6f}",
                    end="",
                    flush=True,
                )
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

        if self.use_coreset:
            B_full = np.zeros((self.n_archetypes, self.n_samples))
            for B_col_idx, coreset_idx in enumerate(coreset_indices):
                B_full[:, coreset_idx] += B[:, B_col_idx]
            B = B_full
            Z = B @ X_raw
            # TODO: change to projected gradients or frank-wolfe here!
            A = _compute_A_regularized_nnls(X=X_raw, Z=Z, A=None)

        # If using weights, we need to recalculate A and B using the unweighted data
        if self.weight:
            A = compute_A(X, Z, A, **self.optim_kwargs)
            B = compute_B(X, A, B, **self.optim_kwargs)
            Z = B @ X

        if self.scaling:
            X *= global_scale
            X_raw *= global_scale
            Z *= global_scale

        if self.centering:
            X += feature_means
            X_raw += feature_means
            Z += feature_means

        # save the output
        TSS = np.sum(X_raw * X_raw)
        RSS = np.linalg.norm(X_raw - A @ Z) ** 2  # RSS on full TS
        self.RSS = RSS
        self.varexpl = (TSS - RSS) / TSS
        self.Z = Z
        self.A = A
        self.B = B
        self.RSS_trace = self.RSS_trace[self.RSS_trace > 0.0]
        self.fitting_info = {
            "conv": convergence_flag if self.max_iter > 0 else None,
            "n_iter": n_iter if self.max_iter > 0 else None,
            "coreset_indices": coreset_indices if self.use_coreset else None,
            "weights": W if (self.use_coreset or self.weight) else None,
            "inital_indices": inital_indices,
        }
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the best convex approximation A of X by the archetypes Z.

        Parameters
        ----------
        X : `np.ndarray`
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
