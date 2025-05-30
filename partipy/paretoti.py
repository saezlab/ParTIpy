import inspect

import anndata
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from tqdm import tqdm

from ._docs import docs
from .arch import AA
from .const import DEFAULT_INIT, DEFAULT_OPTIM
from .selection import compute_IC


def set_obsm(adata: anndata.AnnData, obsm_key: str, n_dimensions: int | list[int]) -> None:
    """
    Sets the `obsm` key and dimensionality to be used as input for archetypal analysis (AA).

    This function verifies that the specified `obsm_key` exists in `adata.obsm` and that the
    requested number of dimensions does not exceed the available dimensions in that matrix.
    The configuration is stored in `adata.uns["AA_config"]`.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing single-cell data. The specified `obsm_key` should refer to
        a matrix in `adata.obsm` to be used as input for AA.

    obsm_key : str
        Key in `adata.obsm` pointing to the matrix to be used for AA.

    n_dimensions : int | list[int]
        Number of dimensions to retain from `adata.obsm[obsm_key]`. Must be less than or equal
        to the number of columns in that matrix.

    Returns
    -------
    None
        The AA configuration is stored in `adata.uns["AA_config"]`.
    """
    if obsm_key not in adata.obsm:
        raise ValueError(f"'{obsm_key}' not found in adata.obsm. Available keys are: {list(adata.obsm.keys())}")

    available_dim = adata.obsm[obsm_key].shape[1]

    if isinstance(n_dimensions, int):
        n_dimensions = list(range(n_dimensions))

    if max(n_dimensions) > available_dim:
        raise ValueError(
            f"Requested {max(n_dimensions)} dimensions from '{obsm_key}', but only {available_dim} are available."
        )

    if "AA_config" in adata.uns:
        print("Warning: 'aa_config' already exists in adata.uns and will be overwritten.")

    adata.uns["AA_config"] = {
        "obsm_key": obsm_key,
        "n_dimensions": n_dimensions,
    }


def _validate_aa_config(adata: anndata.AnnData) -> None:
    """
    Validates that the AnnData object is properly configured for archetypal analysis (AA).

    This function checks that:
    - `adata.uns["AA_config"]` exists,
    - it contains the keys "obsm_key" and "n_dimensions",
    - the specified `obsm_key` exists in `adata.obsm`,
    - and that the requested number of dimensions does not exceed the available dimensions.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object expected to contain AA configuration in `adata.uns["AA_config"]`.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the configuration is missing, incomplete, or inconsistent with the contents of `adata.obsm`.
    """
    if "AA_config" not in adata.uns:
        raise ValueError("AA configuration not found in `adata.uns['AA_config']`.")

    config = adata.uns["AA_config"]

    if not isinstance(config, dict):
        raise ValueError("`adata.uns['AA_config']` must be a dictionary.")

    required_keys = {"obsm_key", "n_dimensions"}
    missing = required_keys - config.keys()
    if missing:
        raise ValueError(f"Missing keys in `aa_config`: {missing}")

    obsm_key = config["obsm_key"]
    n_dimensions = config["n_dimensions"]

    if obsm_key not in adata.obsm:
        raise ValueError(f"'{obsm_key}' not found in `adata.obsm`. Available keys: {list(adata.obsm.keys())}")

    available_dim = adata.obsm[obsm_key].shape[1]
    if max(n_dimensions) > available_dim:
        raise ValueError(
            f"Configured number of dimensions ({max(n_dimensions)}) exceeds available dimensions ({available_dim}) in `adata.obsm['{obsm_key}']`."
        )


def _validate_aa_results(adata: anndata.AnnData) -> None:
    """
    Validates that the result from Archetypal Analysis is present in the AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.

    Raises
    ------
    ValueError
        If the archetypal analysis result is not found in `adata.uns["AA_results"]`.
    """
    if "AA_results" not in adata.uns:
        raise ValueError(
            "Result from Archetypal Analysis not found in `adata.uns['AA_results']`. "
            "Please run the AA() function first."
        )


@docs.dedent
def compute_selection_metrics(
    adata: anndata.AnnData,
    min_k: int = 2,
    max_k: int = 10,
    n_restarts: int = 5,
    optim: str = DEFAULT_OPTIM,
    init: str = DEFAULT_INIT,
    n_jobs: int = -1,
    seed: int = 42,
    **kwargs,
) -> None:
    """
    Compute the variance explained by Archetypal Analysis (AA) for a range of archetypes.

    This function performs Archetypal Analysis (AA) across a range of archetype counts (`min_k` to `max_k`)
    on the PCA representation stored in `adata.obsm[obsm_key]`. It stores the explained variance and other
    diagnostics in `adata.uns["AA_metrics"]`.

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object containing adata.obsm["obsm_key"].
    min_k : int, default `2`
        Minimum number of archetypes to test.
    max_k : int, default `10`
        Maximum number of archetypes to test.
    %(optim)s
    %(init)s
    %(seed)s
    n_jobs : int, default `-1`
        Number of jobs for parallel computation. `-1` uses all available cores.
    **kwargs:
        Additional keyword arguments passed to `AA` class.

    Returns
    -------
    None
        The results are stored in `adata.uns["AA_metrics"]` as a DataFrame with the following columns:
        - `k`: The number of archetypes.
        - `varexpl`: Variance explained by the AA model with `k` archetypes.
    """
    # input validation
    _validate_aa_config(adata=adata)
    if min_k < 2:
        raise ValueError("`min_k` must be at least 2.")
    if max_k < min_k:
        raise ValueError("`max_k` must be greater than or equal to `min_k`.")

    obsm_key = adata.uns["AA_config"]["obsm_key"]
    n_dimensions = adata.uns["AA_config"]["n_dimensions"]
    X = adata.obsm[obsm_key][:, n_dimensions]

    k_arr = np.arange(min_k, max_k + 1)

    rng = np.random.default_rng(seed)
    seeds = rng.choice(a=int(1e9), size=n_restarts, replace=False)

    # Parallel computation of AA
    def _compute_archetypes(k, seed):
        aa_model = AA(n_archetypes=k, seed=seed, optim=optim, init=init, **kwargs).fit(X)
        A = aa_model.A
        Z = aa_model.Z
        RSS = aa_model.RSS
        varexpl = aa_model.varexpl
        return {"k": k, "Z": Z, "A": A, "RSS": RSS, "varexpl": varexpl, "seed": seed}

    if n_jobs == 1:
        results_list = [_compute_archetypes(k=k, seed=seed) for k in k_arr for seed in seeds]
    else:
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_compute_archetypes)(k=k, seed=seed) for k in k_arr for seed in seeds
        )

    for result_dict in results_list:
        X_tilde = result_dict["A"] @ result_dict["Z"]
        result_dict["IC"] = compute_IC(X=X, X_tilde=X_tilde, n_archetypes=result_dict["k"])

    result_df = pd.DataFrame(
        [{"k": d["k"], "seed": d["seed"], "varexpl": d["varexpl"], "IC": d["IC"]} for d in results_list]
    )
    result_df["seed"] = pd.Categorical(result_df["seed"], categories=seeds)
    adata.uns["AA_metrics"] = result_df


@docs.dedent
def compute_bootstrap_variance(
    adata: anndata.AnnData,
    n_bootstrap: int,
    n_archetypes_list: int | list[int] | None = None,
    optim: str = DEFAULT_OPTIM,
    init: str = DEFAULT_INIT,
    seed: int = 42,
    save_to_anndata: bool = True,
    n_jobs: int = -1,
    verbose: bool = False,
    **optim_kwargs,
) -> None | dict[str, pd.DataFrame]:
    """
    Perform bootstrap sampling to compute archetypes and assess their stability.

    This function generates bootstrap samples from the data, computes archetypes for each sample,
    aligns them with the reference archetypes, and stores the results in `adata.uns["AA_bootstrap"]`.
    It allows assessing the stability of the archetypes across multiple bootstrap iterations.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the data to fit the archetypes. The data should be available in
        `adata.obsm[obsm_key]`.
    n_bootstrap : int
        The number of bootstrap samples to generate.
    n_archetypes_list : Union[int, List[int]], default `list(range(2, 8))`
        A list specifying the numbers of archetypes to evaluate. Can also be a single int.
    %(optim)s
    %(init)s
    %(seed)s
    save_to_anndata : bool, default `True`
        Whether to save the results to `adata.uns["AA_bootstrap"]`. If `False`, the result is returned.
    n_jobs : int, default `-1`
        The number of jobs to run in parallel. `-1` uses all available cores.
    verbose : bool, default `False`
        Whether to print the progress
    **optim_kwargs:
        TODO: Additional keyword arguments passed to `AA` class.

    Returns
    -------
    None
        The results are stored in `adata.uns["AA_bootstrap"]` as a DataFrame with the following columns:
        - `x_i`: The coordinates of the archetypes in the i-th principal component.
        - `archetype`: The archetype index.
        - `iter`: The bootstrap iteration index (0 for the reference archetypes).
        - `reference`: A boolean indicating whether the archetype is from the reference model.
        - `mean_variance`: The mean variance of all archetype coordinates across bootstrap samples.
        - `variance_per_archetype`: The mean variance of each archetype coordinates across bootstrap samples.
    """
    # input validation
    _validate_aa_config(adata=adata)

    if n_archetypes_list is None:
        n_archetypes_list = list(range(2, 8))
    elif isinstance(n_archetypes_list, int):
        n_archetypes_list = [n_archetypes_list]

    obsm_key = adata.uns["AA_config"]["obsm_key"]
    n_dimensions = adata.uns["AA_config"]["n_dimensions"]
    X = adata.obsm[obsm_key][:, n_dimensions]

    n_samples, n_features = X.shape
    rng = np.random.default_rng(seed)

    df_dict = {}
    for k in n_archetypes_list:
        # Reference archetypes
        ref_Z = compute_archetypes(
            adata=adata,
            n_archetypes=k,
            optim=optim,
            init=init,
            seed=seed,
            save_to_anndata=False,
            archetypes_only=True,
            **optim_kwargs,
        )

        # Generate bootstrap samples
        idx_bootstrap = rng.choice(n_samples, size=(n_bootstrap, n_samples), replace=True)

        # Define function for parallel computation
        def compute_bootstrap_z(idx, k=k):
            return AA(n_archetypes=k, optim=optim, init=init, **optim_kwargs).fit(X[idx, :]).Z

        # Parallel computation of AA on bootstrap samples
        if verbose:
            Z_list = Parallel(n_jobs=n_jobs)(
                delayed(compute_bootstrap_z)(idx)
                for idx in tqdm(idx_bootstrap, total=n_bootstrap, desc=f"Testing {k} Archetypes")
            )
        else:
            Z_list = Parallel(n_jobs=n_jobs)(delayed(compute_bootstrap_z)(idx) for idx in idx_bootstrap)

        # Align archetypes
        Z_list = [_align_archetypes(ref_arch=ref_Z.copy(), query_arch=query_Z.copy()) for query_Z in Z_list]  # type: ignore[union-attr]

        # Compute variance per archetype
        Z_stack = np.stack(Z_list)
        var_per_archetype = Z_stack.var(axis=0).mean(axis=1)
        mean_variance = var_per_archetype.mean()

        # Create result dataframe
        bootstrap_data = [
            pd.DataFrame(Z, columns=[f"x{i}" for i in range(n_features)]).assign(archetype=np.arange(k), iter=i + 1)
            for i, Z in enumerate(Z_list)
        ]
        bootstrap_df = pd.concat(bootstrap_data)

        df = pd.DataFrame(ref_Z, columns=[f"x{i}" for i in range(n_features)])
        df["archetype"] = np.arange(k)
        df["iter"] = 0

        bootstrap_df = pd.concat((bootstrap_df, df), axis=0)
        bootstrap_df["reference"] = bootstrap_df["iter"] == 0
        bootstrap_df["archetype"] = pd.Categorical(bootstrap_df["archetype"])

        bootstrap_df["mean_variance"] = mean_variance

        archetype_variance_map = dict(zip(np.arange(k), var_per_archetype, strict=False))
        bootstrap_df["variance_per_archetype"] = bootstrap_df["archetype"].astype(int).map(archetype_variance_map)

        df_dict[str(k)] = bootstrap_df

    if save_to_anndata:
        adata.uns["AA_bootstrap"] = df_dict
        return None
    else:
        return df_dict


# TODO: I could also just use any of the compute_A functions to achieve this more robustly!
def _project_on_affine_subspace(X, Z) -> np.ndarray:
    """
    Projects a set of points X onto the affine subspace spanned by the vertices Z.

    Parameters
    ----------
    X : numpy.ndarray
        N x D array of N points in D-dimensional space to be projected.
    Z : numpy.ndarray
        K x D array of K vertices (archetypes) defining the affine subspace in D-dimensional space.

    Returns
    -------
    X_proj : numpy.ndarray
        The coordinates of the projected points in the subspace defined by Z.
    """
    # arbitrarily define the first archetype as translation vector for the affine subspace spanned by the archetypes
    translation_vector = Z[0, :]  # D x 1 dimensions

    # the other archetypes, then define the linear subspace onto which we project the data
    # (after we subtract the translation vector from the coordinates of the other archetypes)
    projection_matrix = Z[1:, :].copy()
    projection_matrix -= translation_vector
    projection_matrix = projection_matrix.T  # D x (K-1) dimensions
    # pseudoinverse = np.linalg.inv(projection_matrix.T @ projection_matrix) @ projection_matrix.T
    pseudoinverse = np.linalg.pinv(projection_matrix)

    X_proj = X.copy()
    X_proj -= translation_vector
    X_proj = X_proj @ pseudoinverse.T

    return X_proj


def _compute_t_ratio(X: np.ndarray, Z: np.ndarray) -> float:  # pragma: no cover
    """
    Compute the t-ratio: volume(polytope defined by Z) / volume(convex hull of X)

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Data matrix.
    Z : np.ndarray, shape (n_archetypes, n_features)
        Archetypes matrix.

    Returns
    -------
    float
        The t-ratio.
    """
    n_features, n_archetypes = X.shape[1], Z.shape[0]

    if n_archetypes < 2:
        raise ValueError("At least 2 archetypes are required (k >= 2).")

    if n_archetypes < (n_features + 1):
        proj_X = _project_on_affine_subspace(X, Z)
        proj_Z = _project_on_affine_subspace(Z, Z)
        convhull_volume = ConvexHull(proj_X).volume
        polytope_volume = ConvexHull(proj_Z).volume
    else:
        convhull_volume = ConvexHull(X).volume
        polytope_volume = ConvexHull(Z).volume

    return polytope_volume / convhull_volume


def compute_t_ratio(adata) -> None:  # pragma: no cover
    """
    Compute the t-ratio from an AnnData object that contains archetypes, i.e has `adata.uns["AA_results"]["Z"]`.


    Parameters
    ----------
    adata : anndata.AnnData
        If AnnData: must contain `obsm[obsm_key]` and `uns["AA_results"]["Z"]`.

    Returns
    -------
    Optional[float]
        - If input is AnnData, result is stored in `X.uns["t_ratio"]`.
        - If input is ndarray, result is returned as float.
    """
    # input validation
    _validate_aa_config(adata=adata)
    if "AA_results" not in adata.uns or "Z" not in adata.uns["AA_results"]:
        raise ValueError("Missing archetypes in `adata.uns['AA_results']['Z']`.")

    obsm_key = adata.uns["AA_config"]["obsm_key"]
    n_dimensions = adata.uns["AA_config"]["n_dimensions"]
    X = adata.obsm[obsm_key][:, n_dimensions]
    Z = adata.uns["AA_results"]["Z"]
    t_ratio = _compute_t_ratio(X, Z)
    adata.uns["t_ratio"] = t_ratio
    return None


@docs.dedent
def t_ratio_significance(
    adata, n_iter=100, seed=42, n_jobs=-1, save_permutation_results: bool = False
):  # pragma: no cover
    """
    Assesses the significance of the polytope spanned by the archetypes by comparing the t-ratio of the original data to t-ratios computed from randomized datasets.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing `adata.obsm["X_pca"]` and `adata.uns["AA_config"]["n_dimensions"]`, optionally `adata.uns["t_ratio"]`. If `adata.uns["t_ratio"]` doesnt exist it is called and computed.
    n_iter : int, default `100`
        Number of randomized datasets to generate.
    %(seed)s
    n_jobs : int, default `-1`
        Number of jobs for parallelization. Use -1 to use all available cores.

    Returns
    -------
    float
        The proportion of randomized datasets with a t-ratio greater than the original t-ratio (p-value).
    """
    # input validation
    _validate_aa_config(adata=adata)

    if "t_ratio" not in adata.uns:
        print("Computing t-ratio...")
        compute_t_ratio(adata)

    obsm_key = adata.uns["AA_config"]["obsm_key"]
    n_dimensions = adata.uns["AA_config"]["n_dimensions"]
    X = adata.obsm[obsm_key][:, n_dimensions]

    t_ratio = adata.uns["t_ratio"]
    rss = adata.uns["AA_results"]["RSS"][-1]
    n_samples, n_features = X.shape
    n_archetypes = adata.uns["AA_results"]["Z"].shape[0]

    rng_master = np.random.default_rng(seed)
    rng_list = [np.random.default_rng(rng_master.integers(int(1e9))) for _ in range(n_iter)]

    def compute_randomized_metrics(rng_inner):
        X_perm = np.column_stack([rng_inner.permutation(X[:, col_idx]) for col_idx in range(n_features)])
        AA_perm = AA(n_archetypes=n_archetypes)
        AA_perm.fit(X_perm)
        Z_perm = AA_perm.Z
        rss_perm = AA_perm.RSS
        t_ratio_perm = _compute_t_ratio(X_perm, Z_perm)
        return t_ratio_perm, rss_perm

    # Parallel computation
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_randomized_metrics)(rng) for rng in tqdm(rng_list, desc="Randomizing")
    )

    t_ratios_perm, rss_perm = map(np.array, zip(*results, strict=False))

    if save_permutation_results:
        adata.uns["AA_permutation"] = {"t_ratio": t_ratios_perm, "rss": rss_perm}

    # Calculate the p-value
    t_ratio_p_value = 1 - np.mean(t_ratio > t_ratios_perm)
    rss_p_value = 1 - np.mean(rss < rss_perm)

    return {"t_ratio_p_value": t_ratio_p_value, "rss_p_value": rss_p_value}


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


@docs.dedent
def compute_archetypes(
    adata: anndata.AnnData,
    n_archetypes: int,
    n_restarts: int = 5,
    init: str | None = None,
    optim: str | None = None,
    weight: None | str = None,
    max_iter: int | None = None,
    rel_tol: float | None = None,
    verbose: bool | None = None,
    seed: int = 42,
    n_jobs: int = -1,
    save_to_anndata: bool = True,
    archetypes_only: bool = True,
    **optim_kwargs,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, list[float] | np.ndarray, float] | None:
    """
    Perform Archetypal Analysis (AA) on the input data.

    This function is a wrapper around the AA class, offering a simplified interface for fitting the model
    and returning the results, or saving them to the AnnData object. It allows users to customize the
    archetype computation with various parameters for initialization, optimization, convergence, and output.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the data to fit the archetypes. The data should be available in
        `adata.obsm[obsm_key]`.
    n_archetypes : int
        The number of archetypes to compute.
    n_restarts: int
        The optimization with be run with n_restarts. The run with the lowest RSS will be kept.
    %(init)s
    %(optim)s
    %(weight)s
    %(max_iter)s
    %(rel_tol)s
    %(verbose)s
    %(seed)s
    n_jobs : int, default `-1`
        Number of jobs for parallel computation. `-1` uses all available cores.
    save_to_anndata : bool, default `True`
        Whether to save the results to the AnnData object. If False, the results are returned as a tuple. If
        `adata` is not an AnnData object, this is ignored.
    archetypes_only : bool, default `True`
        Whether to save/return only the archetypes matrix `Z` (if det to True) or also the full outputs, including
        the matrices `A`, `B`, `RSS`, and variance explained `varexpl`.
    optim_kwargs : dict | None, default `None`
        Additional arguments that are passed to `partipy.arch.AA`

    Returns
    -------
    np.ndarray or tuple or None
        The output depends on the values of `save_to_anndata` and `archetypes_only`:

        - If `archetypes_only` is True:
            Only the archetype matrix `Z` is returned or saved.

        - If `archetypes_only` is False:
            A tuple is returned or saved, containing:

            - A : ndarray of shape (n_samples, n_archetypes)
                The matrix of weights for the data points.
            - B : ndarray of shape (n_archetypes, n_samples)
                The matrix of weights for the archetypes.
            - Z : ndarray of shape (n_archetypes, n_features)
                The archetypes matrix.
            - RSS : float
                The residual sum of squares from optimization.
            - varexpl : float
                The variance explained by the model.

        - If `save_to_anndata` is True:
            Returns `None`. Results are saved to `adata.uns["AA_results"]`.

        - If `save_to_anndata` is False:
            The results described above are returned.

    """
    # input validation
    _validate_aa_config(adata=adata)

    # Get the signature of AA.__init__
    signature = inspect.signature(AA.__init__)

    # Create a dictionary of parameter names and their default values
    defaults = {
        param: signature.parameters[param].default
        for param in signature.parameters
        if param != "self" and param != "n_archetypes"
    }

    rng = np.random.default_rng(seed)
    seeds = rng.choice(a=int(1e9), size=n_restarts, replace=False)

    # Use the provided values or fall back to the defaults
    init = init if init is not None else defaults["init"]
    optim = optim if optim is not None else defaults["optim"]
    weight = weight if weight is not None else defaults["weight"]
    max_iter = max_iter if max_iter is not None else defaults["max_iter"]
    rel_tol = rel_tol if rel_tol is not None else defaults["rel_tol"]
    verbose = verbose if verbose is not None else defaults["verbose"]

    # Extract the data matrix used to fit the archetypes
    obsm_key = adata.uns["AA_config"]["obsm_key"]
    n_dimensions = adata.uns["AA_config"]["n_dimensions"]
    X = adata.obsm[obsm_key][:, n_dimensions]
    X = X.astype(np.float32)

    # Parallel computation of AA
    def _compute_archeptyes(seed):
        model = AA(
            n_archetypes=n_archetypes,
            init=init,
            optim=optim,
            weight=weight,
            max_iter=max_iter,
            rel_tol=rel_tol,
            verbose=verbose,
            seed=seed,
            **optim_kwargs,
        )
        model.fit(X)
        return {
            "A": model.A,
            "B": model.B,
            "Z": model.Z,
            "RSS": model.RSS_trace,
            "RSS_full": model.RSS,
            "varexpl": model.varexpl,
            "seed": seed,
        }

    if n_jobs == 1:
        results_list = [_compute_archeptyes(seed=seed) for seed in seeds]
    else:
        results_list = Parallel(n_jobs=n_jobs)(delayed(_compute_archeptyes)(seed=seed) for seed in seeds)

    argmax = np.argmax(np.array([r["varexpl"] for r in results_list]))

    result_dict = results_list[argmax]

    # Save the results to the AnnData object if specified
    if save_to_anndata:
        if archetypes_only:
            adata.uns["AA_results"] = {"Z": result_dict["Z"]}
        else:
            adata.uns["AA_results"] = {
                "A": result_dict["A"],
                "B": result_dict["B"],
                "Z": result_dict["Z"],
                "RSS": result_dict["RSS"],
                "RSS_full": result_dict["RSS_full"],
                "varexpl": result_dict["varexpl"],
            }
        return None
    else:
        if archetypes_only:
            return result_dict["Z"]
        else:
            return result_dict["A"], result_dict["B"], result_dict["Z"], result_dict["RSS"], result_dict["varexpl"]
