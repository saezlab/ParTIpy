import inspect

import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from tqdm import tqdm

from .arch import AA
from .const import DEFAULT_INIT, DEFAULT_OPTIM


def set_dimension_aa(adata: sc.AnnData, n_pcs: int) -> None:
    """
    Sets the number of PCs used for subsetting the PCA in `adata.obsm["X_pca"]`.
    If `adata.obsm["X_pca"]` does not exist, PCA is computed and stored in `adata.obsm["X_pca"]`.
    The number of PCs are stored in `adata.uns["n_pcs"]`

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing single-cell data.
    n_pcs : int
        The number of principal components (PCs) to retain. Must be less than or equal to the
        number of available PCs in `adata.obsm["X_pca"]`.

    Returns
    -------
    None
        The number of PCs are stored in `adata.uns["n_pcs"]`
    """
    # Validation input
    if "X_pca" not in adata.obsm:
        print("X_pca not found in adata.obsm. Computing PCA on highly variable genes...")
        sc.pp.pca(adata, mask_var="highly_variable")

    if n_pcs > adata.obsm["X_pca"].shape[1]:
        raise ValueError(f"Requested {n_pcs} PCs, but only {adata.obsm['X_pca'].shape[1]} PCs are available.")

    adata.uns["n_pcs"] = n_pcs


def var_explained_aa(
    adata: sc.AnnData,
    obsm_key: str = "X_pca",
    min_a: int = 2,
    max_a: int = 10,
    optim: str = DEFAULT_OPTIM,
    init: str = DEFAULT_INIT,
    n_jobs: int = -1,
    **kwargs,
) -> None:
    """
    Compute the variance explained by Archetypal Analysis (AA) for a range of archetypes.

    This function performs Archetypal Analysis (AA) for a range of archetypes (from `min_a` to `max_a`)
    on the PCA data stored in `adata.obsm["X_pca"]`. The results are
    stored in `adata.uns["AA_var"]`.

    Parameters
    ----------
    adata: sc.AnnData
        AnnData object containing adata.obsm["obsm_key"].
    obsm_key: str, optional (default="X_pca")
        opsm to use as representation to fit archetypes
    min_a : int, optional (default=2)
        Minimum number of archetypes to test.
    max_a : int, optional (default=10)
        Maximum number of archetypes to test.
    optim : str, optional (default=DEFAULT_OPTIM)
        The optimization function to use for Archetypal Analysis.
    init : str, optional (default=DEFAULT_INIT)
        The initialization function to use for Archetypal Analysis.
    n_jobs : int, optional (default=-1)
        Number of jobs for parallel computation. `-1` uses all available cores.
    kwargs:
        Additional keyword arguments passed to `AA` class.

    Returns
    -------
    None
        The results are stored in `adata.uns["AA_var"]` as a DataFrame with the following columns:
        - `k`: The number of archetypes.
        - `varexpl`: The variance explained by the model.
        - `varexpl_ontop`: The additional variance explained compared to the model with `k-1` archetypes.
        - `dist_to_projected`: The distance between the variance explained and its projection on the line
          connecting the variance explained of first and last k.
    """
    # Validation input
    if obsm_key not in adata.obsm.keys():
        raise ValueError("`obsm_key` must be in adata.obsm.keys()")
    if min_a < 2:
        raise ValueError("`min_a` must be at least 2.")
    if max_a < min_a:
        raise ValueError("`max_a` must be greater than or equal to `min_a`.")

    if "n_pcs" not in adata.uns:
        raise ValueError(
            "n_pcs not found in adata.uns. Please set the dimension for archetypal analysis with set_dimension()"
        )

    X = adata.obsm[obsm_key][:, : adata.uns["n_pcs"]]

    k_arr = np.arange(min_a, max_a + 1)

    # Parallel computation of AA
    def _compute_archeptyes(k):
        A, B, Z, RSS, varexpl = AA(n_archetypes=k, optim=optim, init=init, **kwargs).fit(X).return_all()
        return k, {"Z": Z, "A": A, "B": B, "RSS": RSS, "varexpl": varexpl}

    if n_jobs == 1:
        results_list = [_compute_archeptyes(k) for k in k_arr]
    else:
        results_list = Parallel(n_jobs=n_jobs)(delayed(_compute_archeptyes)(k) for k in k_arr)

    # results = {k: result for k, result in results_list}
    results = dict(results_list)  # faster, and see https://docs.astral.sh/ruff/rules/unnecessary-comprehension/

    varexpl_values = np.array([results[k]["varexpl"] for k in k_arr])

    plot_df = pd.DataFrame(
        {
            "k": k_arr,
            "varexpl": varexpl_values,
            "varexpl_ontop": np.insert(np.diff(varexpl_values), 0, varexpl_values[0]),
        }
    )

    # Compute the distance of the explained variance to its projection
    offset_vec = plot_df[["k", "varexpl"]].iloc[0].values
    proj_vec = (plot_df[["k", "varexpl"]].values - offset_vec)[-1, :][:, None]
    proj_mtx = proj_vec @ np.linalg.inv(proj_vec.T @ proj_vec) @ proj_vec.T
    proj_val = (proj_mtx @ (plot_df[["k", "varexpl"]].values - offset_vec).T).T + offset_vec
    proj_df = pd.DataFrame(proj_val, columns=["k", "varexpl"])
    plot_df["dist_to_projected"] = np.linalg.norm(
        plot_df[["k", "varexpl"]].values - proj_df[["k", "varexpl"]].values, axis=1
    )

    adata.uns["AA_var"] = plot_df


def bootstrap_aa(
    adata: sc.AnnData,
    n_bootstrap: int,
    n_archetypes: int,
    optim: str = DEFAULT_OPTIM,
    init: str = DEFAULT_INIT,
    seed: int = 42,
    n_jobs: int = -1,
) -> None:
    """
    Perform bootstrap sampling to compute archetypes and assess their stability.

    This function generates bootstrap samples from the data, computes archetypes for each sample,
    aligns them with the reference archetypes, and stores the results in `adata.uns["AA_bootstrap"]`.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object. The PCA data should be stored in `adata.obsm["X_pca"]`.
    n_bootstrap : int
        The number of bootstrap samples to generate.
    n_archetypes : int
        The number of archetypes to compute for each bootstrap sample.
    optim : str, optional (default=DEFAULT_OPTIM)
        The optimization function to use for Archetypal Analysis.
    init : str, optional (default=DEFAULT_INIT)
        The initialization function to use for Archetypal Analysis.
    seed : int, optional (default=42)
        The random seed for reproducibility.

    Returns
    -------
    None
        The results are stored in `adata.uns["AA_bootstrap"]` as a DataFrame with the following columns:
        - `pc_i`: The coordinates of the archetypes in the i-th principal component.
        - `archetype`: The archetype index.
        - `iter`: The bootstrap iteration index (0 for the reference archetypes).
        - `reference`: A boolean indicating whether the archetype is from the reference model.
        - `mean_variance`: The mean variance of archetype coordinates across bootstrap samples.
    """
    # Validation input
    if "n_pcs" not in adata.uns:
        raise ValueError(
            "n_pcs not found in adata.uns. Please set the dimension for archetypal analysis with set_dimension()"
        )

    X = adata.obsm["X_pca"][:, : adata.uns["n_pcs"]]

    n_samples, n_features = X.shape
    rng = np.random.default_rng(seed)

    # Reference archetypes
    ref_Z = AA(n_archetypes=n_archetypes, optim=optim, init=init).fit(X).Z

    # Generate bootstrap samples
    idx_bootstrap = rng.choice(n_samples, size=(n_bootstrap, n_samples), replace=True)

    # Define function for parallel computation
    def compute_bootstrap_z(idx):
        return AA(n_archetypes=n_archetypes, optim=optim, init=init).fit(X[idx, :]).Z

    # Parallel computation of AA on bootstrap samples
    Z_list = Parallel(n_jobs=n_jobs)(delayed(compute_bootstrap_z)(idx) for idx in idx_bootstrap)

    # Align archetypes
    Z_list = [_align_archetypes(ref_arch=ref_Z.copy(), query_arch=query_Z.copy()) for query_Z in Z_list]

    # Compute variance
    Z_stack = np.stack(Z_list)
    var_per_archetype = Z_stack.var(axis=0).mean(axis=1)
    mean_variance = var_per_archetype.mean()

    # Create result dataframe
    bootstrap_data = [
        pd.DataFrame(Z, columns=[f"pc_{i}" for i in range(n_features)]).assign(
            archetype=np.arange(n_archetypes), iter=i + 1
        )
        for i, Z in enumerate(Z_list)
    ]
    bootstrap_df = pd.concat(bootstrap_data)

    df = pd.DataFrame(ref_Z, columns=[f"pc_{i}" for i in range(n_features)])
    df["archetype"] = np.arange(n_archetypes)
    df["iter"] = 0

    bootstrap_df = pd.concat((bootstrap_df, df), axis=0)
    bootstrap_df["reference"] = bootstrap_df["iter"] == 0
    bootstrap_df["archetype"] = pd.Categorical(bootstrap_df["archetype"])

    bootstrap_df["mean_variance"] = mean_variance

    adata.uns["AA_bootstrap"] = bootstrap_df


def _project_on_affine_subspace(X, Z) -> np.ndarray:
    """
    Projects a set of points X onto the affine subspace spanned by the vertices Z.

    Parameters
    ----------
    X : numpy.ndarray
        A (D x n) array of n points in D-dimensional space to be projected.
    Z : numpy.ndarray
        A (D x k) array of k vertices (archetypes) defining the affine subspace in D-dimensional space.

    Returns
    -------
    proj_coord : numpy.ndarray
        The coordinates of the projected points in the subspace defined by Z.
    """
    D, k = Z.shape

    # Compute the projection vectors (basis for the affine subspace)
    if k == 2:
        # For a line (k=2), the projection vector is simply the difference between the two vertices
        proj_vec = (Z[:, 1] - Z[:, 0])[:, None]
    else:
        # For higher dimensions, compute the projection vectors relative to the first vertex
        proj_vec = Z[:, 1:] - Z[:, 0][:, None]

    # Compute the coordinates of the projected points in the subspace
    proj_coord = np.linalg.inv(proj_vec.T @ proj_vec) @ proj_vec.T @ (X - Z[:, 0][:, None])

    return proj_coord


def compute_t_ratio(
    X: sc.AnnData | np.ndarray,
    Z: np.ndarray | None = None,
) -> float | None:
    """
    Compute the t-ratio, which is the ratio of the volume of the polytope defined by the archetypes (Z)
    to the volume of the convex hull of the data points (X).

    Parameters
    ----------
    X : Union[sc.AnnData, np.ndarray]
        The input data, which can be either:
        - An AnnData object containing the following attributes:
            - `adata.obsm["X_pca"]`: A 2D array of shape (n_samples, n_features) representing the PCA coordinates of the data.
            - `adata.uns["n_pcs"]`: The number of principal components used for AA.
            - `adata.uns["archetypal_analysis"]["Z"]`: A 2D array of shape (n_archetypes, n_features) representing the archetypes.
        - A 2D numpy array of shape (n_samples, n_features) representing the data matrix. In this case, `Z` must be provided.
    Z : np.ndarray, optional
        A 2D array of shape (n_archetypes, n_features) representing the archetypes. Required if `X` is a numpy array.

    Returns
    -------
    Optional[float]
        - If `X` is an AnnData object, the t-ratio is stored in `X.uns["t_ratio"]` and nothing is returned.
        - If `X` is a numpy array, the t-ratio is returned as a float.
    """
    adata = None
    if isinstance(X, np.ndarray):
        if Z is None:
            raise ValueError("Z must be provided when input_data is a numpy.ndarray.")
    else:
        adata = X
        X = adata.obsm["X_pca"][:, : adata.uns["n_pcs"]]
        Z = adata.uns["archetypal_analysis"]["Z"]

    # Extract dimensions D (PCs), and number of archetypes
    D, k = X.shape[1], Z.shape[0]  # type: ignore[union-attr]

    # Input validation
    if k < 2:
        raise ValueError("k must satisfy 2 <= k, meaning you need at least 2 archetypes.")

    if k < D + 1:
        # project onto affine subspace spanned by Z
        proj_X = _project_on_affine_subspace(X.T, Z.T).T  # type: ignore[union-attr]
        proj_Z = _project_on_affine_subspace(Z.T, Z.T).T  # type: ignore[union-attr]

        # Compute the convex hull volumes
        convhull_volume = ConvexHull(proj_X).volume
        polytope_volume = ConvexHull(proj_Z).volume
    else:
        # Compute the convex hull volumes directly
        convhull_volume = ConvexHull(X).volume
        polytope_volume = ConvexHull(Z).volume

    t_ratio = polytope_volume / convhull_volume

    if isinstance(adata, sc.AnnData):
        adata.uns["t_ratio"] = t_ratio
        return None
    else:
        return t_ratio


def t_ratio_significance(adata, iter=1000, seed=42, n_jobs=-1):
    """
    Assesses the significance of the polytope spanned by the archetypes by comparing the t-ratio of the original data to t-ratios computed from randomized datasets.

    Parameters
    ----------
    adata : sc.AnnData
        An AnnData object containing `adata.obsm["X_pca"]` and `adata.uns["n_pcs"], optionally `adata.uns["t_ratio"]`. If `adata.uns["t_ratio"]` doesnt exist it is called and computed.
    rep : int, optional (default=1000)
        Number of randomized datasets to generate.
    seed : int, optional (default=42)
        The random seed for reproducibility.
    n_jobs : int, optional
        Number of jobs for parallelization (default: 1). Use -1 to use all available cores.

    Returns
    -------
    float
        The proportion of randomized datasets with a t-ratio greater than the original t-ratio (p-value).
    """
    # Input validation
    if "X_pca" not in adata.obsm:
        raise ValueError("adata.obsm['X_pca'] not found.")
    if "t_ratio" not in adata.uns:
        print("Computing t-ratio...")
        compute_t_ratio(adata)

    X = adata.obsm["X_pca"][:, : adata.uns["n_pcs"]]
    t_ratio = adata.uns["t_ratio"]
    n_samples, n_features = X.shape
    n_archetypes = adata.uns["archetypal_analysis"]["Z"].shape[0]

    rng = np.random.default_rng(seed)

    def compute_randomized_t_ratio():
        # Shuffle each feature independently
        SimplexRand1 = np.array([rng.permutation(X[:, i]) for i in range(n_features)]).T
        # Compute archetypes and t-ratio for randomized data
        Z_mix = AA(n_archetypes=n_archetypes).fit(SimplexRand1).Z
        return compute_t_ratio(SimplexRand1, Z_mix)

    # Parallelize the computation of randomized t-ratios
    RandRatio = Parallel(n_jobs=n_jobs)(
        delayed(compute_randomized_t_ratio)() for _ in tqdm(range(iter), desc="Randomizing")
    )

    # Calculate the p-value
    p_value = np.sum(np.array(RandRatio) > t_ratio) / iter
    return p_value


def t_ratio_significance_shuffled(adata, iter=1000, seed=42, n_jobs=-1):
    """
    Assesses the significance of the polytope spanned by the archetypes by comparing the t-ratio of the original data to t-ratios computed from randomized datasets.

    Parameters
    ----------
    adata : sc.AnnData
        An AnnData object containing `adata.obsm["X_pca"]` and `adata.uns["n_pcs"], optionally `adata.uns["t_ratio"]`. If `adata.uns["t_ratio"]` doesnt exist it is called and computed.
    rep : int, optional (default=1000)
        Number of randomized datasets to generate.
    seed : int, optional (default=42)
        The random seed for reproducibility.
    n_jobs : int, optional
        Number of jobs for parallelization (default: 1). Use -1 to use all available cores.

    Returns
    -------
    float
        The proportion of randomized datasets with a t-ratio greater than the original t-ratio (p-value).
    """
    # Input validation
    if "X_pca" not in adata.obsm:
        raise ValueError("adata.obsm['X_pca'] not found.")
    if "t_ratio" not in adata.uns:
        print("Computing t-ratio...")
        compute_t_ratio(adata)

    X = adata[:, adata.var["highly_variable"]].copy().X.toarray()
    t_ratio = adata.uns["t_ratio"]
    n_samples, n_features = X.shape
    n_archetypes = adata.uns["archetypal_analysis"]["Z"].shape[0]

    rng = np.random.default_rng(seed)

    def compute_randomized_t_ratio():
        # Shuffle each feature independently
        SimplexRand1 = np.array([rng.permutation(X[:, i]) for i in range(n_features)]).T
        SimplexRand1_pca = sc.pp.pca(SimplexRand1, n_comps=adata.uns["n_pcs"])
        # Compute archetypes and t-ratio for randomized data
        Z_mix = AA(n_archetypes=n_archetypes).fit(SimplexRand1_pca).Z
        return compute_t_ratio(SimplexRand1_pca, Z_mix)

    # Parallelize the computation of randomized t-ratios
    RandRatio = Parallel(n_jobs=n_jobs)(
        delayed(compute_randomized_t_ratio)() for _ in tqdm(range(iter), desc="Randomizing")
    )

    # Calculate the p-value
    p_value = np.sum(np.array(RandRatio) > t_ratio) / iter
    return p_value


def _align_archetypes(ref_arch: np.ndarray, query_arch: np.ndarray) -> np.ndarray:
    """
    Align the archetypes of the query to match the order of archetypes in the reference.

    This function uses the Euclidean distance between archetypes in the reference and query sets
    to determine the optimal alignment. The Hungarian algorithm (linear sum assignment) is used
    to find the best matching pairs, and the query archetypes are reordered accordingly.

    Parameters
    ----------
    ref_arch : np.ndarray
        A 2D array of shape (n_archetypes, n_features) representing the reference archetypes.
    query_arch : np.ndarray
        A 2D array of shape (n_archetypes, n_features) representing the query archetypes.

    Returns
    -------
    np.ndarray
        A 2D array of shape (n_archetypes, n_features) containing the reordered query archetypes.
    """
    # Compute pairwise Euclidean distances
    euclidean_d = cdist(ref_arch, query_arch.copy(), metric="euclidean")

    # Find the optimal assignment using the Hungarian algorithm
    ref_idx, query_idx = linear_sum_assignment(euclidean_d)

    return query_arch[query_idx, :]


def compute_archetypes(
    adata: sc.AnnData,
    n_archetypes: int,
    obsm_key: str = "X_pca",
    init: str | None = None,
    optim: str | None = None,
    weight: None | str = None,
    max_iter: int | None = None,
    derivative_max_iter: int | None = None,
    tol: float | None = None,
    verbose: bool | None = None,
    seed: int = 42,
    save_to_anndata: bool = True,
    archetypes_only: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, list[float | None] | np.ndarray, float] | None:
    """

    Perform Archetypal Analysis (AA) on the input data.

    This function is a wrapper for the AA class, providing a simplified interface for fitting the model,
    and returning the desired outputs or saving them to the AnnData object.

    Parameters
    ----------
    adata : Union[sc.AnnData, np.ndarray]
        The input data, which can be either:
        - An AnnData object containing data in `adata.obsm[obsm_key]`.
    n_archetypes : int
        The number of archetypes to compute.
    obsm_key: str, optional
        Which obsm_key to use, by default "X_pca"
    init : str, optional
        The initialization method for the archetypes. If not provided, the default from the AA class is used.
        Options include:
        - "random": Random initialization.
        - "furthest_sum": Furthest sum initialization.
    optim : str, optional
        The optimization method for fitting the model. If not provided, the default from the AA class is used.
        Options include:
        - "projected_gradients": Projected gradients optimization.
        - "frank_wolfe": Frank-Wolfe optimization.
        - "regularized_nnls": Regularized non-negative least squares optimization.
    weight : str, optional
        The weighting method for the data. If not provided, the default from the AA class is used.
        Options include:
        - "bisquare": Bisquare weighting.
    max_iter : int, optional
        The maximum number of iterations for the optimization. If not provided, the default from the AA class is used.
    derivative_max_iter : int, optional
        The maximum number of iterations for derivative computation. If not provided, the default from the AA class is used.
    tol : float, optional
        The tolerance for convergence. If not provided, the default from the AA class is used.
    verbose : bool, optional
        Whether to print verbose output during fitting. If not provided, the default from the AA class is used.
    seed : int, optional
        Random seed
    save_to_anndata : bool, optional (default=True)
        Whether to save the results to the AnnData object. If `adata` is not an AnnData object, this is ignored.
    archetypes_only : bool, optional (default=True)
        Whether to return only the archetypes matrix. If `save_to_anndata` is True, this parameter determines
        whether only the archetypes are saved to the AnnData object.

    Returns
    -------
    Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]]
        The output depends on the values of `save_to_anndata` and `archetypes_only`:
        - If `archetypes_only` is True:
            - Only the archetype matrix (Z) is returned/ saved
        - If `archetypes_only` is True:
            - returns/ saves a tuple containing:
                - A: The matrix of weights for the data points (n_samples, n_archetypes).
                - B: The matrix of weights for the archetypes (n_archetypes, n_samples).
                - Z: The archetypes matrix (n_archetypes, n_features).
                - RSS: The residual sum of squares.
                - varexpl: The variance explained by the model.
        - If `save_to_anndata` is True:
            - Returns `None`. Results are saved to `adata.uns["archetypal_analysis"]`.
        - If `save_to_anndata` is False:
            - Returns the results.
    """
    # checks
    assert obsm_key in adata.obsm.keys()
    assert "n_pcs" in adata.uns.keys()  # TODO

    # Get the signature of AA.__init__
    signature = inspect.signature(AA.__init__)

    # Create a dictionary of parameter names and their default values
    defaults = {
        param: signature.parameters[param].default
        for param in signature.parameters
        if param != "self" and param != "n_archetypes"
    }

    # Use the provided values or fall back to the defaults
    init = init if init is not None else defaults["init"]
    optim = optim if optim is not None else defaults["optim"]
    weight = weight if weight is not None else defaults["weight"]
    max_iter = max_iter if max_iter is not None else defaults["max_iter"]
    derivative_max_iter = derivative_max_iter if derivative_max_iter is not None else defaults["derivative_max_iter"]
    tol = tol if tol is not None else defaults["tol"]
    verbose = verbose if verbose is not None else defaults["verbose"]

    # Create the AA model with the specified parameters
    model = AA(
        n_archetypes=n_archetypes,
        init=init,
        optim=optim,
        weight=weight,
        max_iter=max_iter,
        derivative_max_iter=derivative_max_iter,
        tol=tol,
        verbose=verbose,
        seed=seed,
    )

    # Extract the data matrix used to fit the archetypes
    X = adata.obsm[obsm_key][:, : adata.uns["n_pcs"]]

    # Fit the model to the data
    model.fit(X)

    # Save the results to the AnnData object if specified
    if save_to_anndata:
        if archetypes_only:
            adata.uns["archetypal_analysis"] = {
                "Z": model.Z,
            }
        else:
            adata.uns["archetypal_analysis"] = {
                "A": model.A,
                "B": model.B,
                "Z": model.Z,
                "RSS": model.RSS_trace,
                "varexpl": model.varexpl,
            }
        return None
    else:
        if archetypes_only:
            return model.Z
        else:
            return model.A, model.B, model.Z, model.RSS_trace, model.varexpl
