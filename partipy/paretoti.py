from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, cast

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
from .io import ensure_archetype_config_keys
from .schema import (
    CORESET_ALGS,
    CORESET_ALGS_TYPES,
    DEFAULT_INIT,
    DEFAULT_MAX_ITER,
    DEFAULT_OPTIM,
    DEFAULT_REL_TOL,
    DEFAULT_WEIGHT,
    INIT_ALG_TYPES,
    INIT_ALGS,
    OPTIM_ALGS_TYPES,
    WEIGHT_ALGS,
    WEIGHT_ALGS_TYPES,
    ArchetypeConfig,
    canonicalize_optim,
    query_configs_by_signature,
)
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
        print('Warning: "AA_config" already exists in adata.uns and will be overwritten.')

    adata.uns["AA_config"] = {
        "obsm_key": obsm_key,
        "n_dimensions": n_dimensions,
    }


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


def _validate_n_archetype_list(
    *,
    min_k: int | None,
    max_k: int | None,
    n_archetypes_list: int | list[int] | None,
    default_range: range,
) -> list[int]:
    deprecated_bounds_used = min_k is not None or max_k is not None

    if n_archetypes_list is not None:
        if deprecated_bounds_used:
            warnings.warn(
                "`min_k` and `max_k` are deprecated and ignored when `n_archetypes_list` is provided.",
                DeprecationWarning,
                stacklevel=3,
            )
        values = [n_archetypes_list] if isinstance(n_archetypes_list, int) else list(n_archetypes_list)
    else:
        if deprecated_bounds_used:
            warnings.warn(
                "`min_k` and `max_k` are deprecated. Use `n_archetypes_list` instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if min_k is None or max_k is None:
                raise ValueError("Both `min_k` and `max_k` must be provided when using the deprecated arguments.")
            if min_k < 2:
                raise ValueError("`min_k` must be at least 2.")
            if max_k < min_k:
                raise ValueError("`max_k` must be greater than or equal to `min_k`.")
            values = list(range(min_k, max_k + 1))
        else:
            values = list(default_range)

    if not values:
        raise ValueError("`n_archetypes_list` must contain at least one archetype count.")

    seen: set[int] = set()
    unique_values: list[int] = []
    for k_val in values:
        if k_val in seen:
            continue
        if k_val < 2:
            raise ValueError("All entries in `n_archetypes_list` must be >= 2.")
        seen.add(int(k_val))
        unique_values.append(int(k_val))

    return unique_values


def _normalize_init_literal(init_value: str | None) -> INIT_ALG_TYPES:
    resolved = init_value if init_value is not None else DEFAULT_INIT
    if resolved not in INIT_ALGS:
        raise ValueError(f"Initialization method '{resolved}' is not supported. Must be one of {INIT_ALGS}.")
    return cast(INIT_ALG_TYPES, resolved)


def _normalize_weight_literal(weight_value: str | None) -> WEIGHT_ALGS_TYPES:
    resolved = weight_value if weight_value is not None else DEFAULT_WEIGHT
    if resolved not in WEIGHT_ALGS:
        raise ValueError(f"Weighting method '{resolved}' is not supported. Must be one of {WEIGHT_ALGS}.")
    return cast(WEIGHT_ALGS_TYPES, resolved)


def _normalize_coreset_literal(coreset_value: str | None) -> CORESET_ALGS_TYPES:
    if coreset_value is None:
        return None
    if coreset_value not in CORESET_ALGS:
        raise ValueError(f"Coreset algorithm '{coreset_value}' is not supported. Must be one of {CORESET_ALGS}.")
    return cast(CORESET_ALGS_TYPES, coreset_value)


def _normalize_optim_literal(optim_value: str | None) -> OPTIM_ALGS_TYPES:
    resolved = optim_value if optim_value is not None else DEFAULT_OPTIM
    return canonicalize_optim(resolved)


def compute_selection_metrics(
    adata: anndata.AnnData,
    min_k: int | None = None,
    max_k: int | None = None,
    n_archetypes_list: int | list[int] | None = None,
    n_restarts: int = 5,
    init: str | None = None,
    optim: str | None = None,
    weight: None | str = None,
    max_iter: int | None = None,
    early_stopping: bool = True,
    rel_tol: float | None = None,
    coreset_algorithm: None | str = None,
    coreset_fraction: float = 0.1,
    coreset_size: None | int = None,
    delta: float = 0.0,
    seed: int = 42,
    save_to_anndata: bool = True,
    return_result: bool = False,
    verbose: bool = False,
    force_recompute: bool = False,
    **optim_kwargs,
) -> None | pd.DataFrame:
    """
    Compute selection diagnostics for Archetypal Analysis (AA) across different archetype counts.

    This function fits AA models for each value in `n_archetypes_list`, optionally across multiple restarts,
    and records variance explained, information criterion, and residual sum of squares. Results are cached in
    `adata.uns["AA_selection_metrics"]` keyed by the AA optimization configuration, and the corresponding AA fits
    are stored in `adata.uns["AA_results"]` via :func:`~partipy.compute_archetypes`.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the matrix configured through `set_obsm`.
    min_k : int | None, optional
        Deprecated. Minimum number of archetypes to test. Use `n_archetypes_list` instead.
    max_k : int | None, optional
        Deprecated. Maximum number of archetypes to test. Use `n_archetypes_list` instead.
    n_archetypes_list : int | list[int] | None, optional
        Number(s) of archetypes to evaluate. Defaults to `range(2, 11)` when not provided.
    n_restarts : int, default `5`
        Number of random restarts per archetype count.
    %(init)s
    %(optim)s
    %(weight)s
    %(max_iter)s
    %(early_stopping)s
    %(rel_tol)s
    %(coreset_algorithm)s
    %(coreset_fraction)s
    %(coreset_size)s
    %(delta)s
    %(seed)s
    save_to_anndata : bool, default `True`
        Whether to cache the results in the AnnData object.
    return_result : bool, default `False`
        If True, return the aggregated results DataFrame.
    verbose : bool, default `False`
        Whether to run AA in verbose mode.
    force_recompute : bool, default `False`
        Recompute metrics even if cached results for the configuration exist.
    **optim_kwargs :
        Additional keyword arguments forwarded to the `AA` class.

    Returns
    -------
    None | pandas.DataFrame
        Returns None unless `return_result` is True, in which case the aggregated DataFrame is returned.
        Cached per-configuration tables can later be concatenated via :func:`~partipy.summarize_aa_metrics`.
    """
    _validate_aa_config(adata=adata)

    if n_restarts < 1:
        raise ValueError("`n_restarts` must be at least 1.")

    n_archetypes_seq = _validate_n_archetype_list(
        min_k=min_k,
        max_k=max_k,
        n_archetypes_list=n_archetypes_list,
        default_range=range(2, 11),
    )

    obsm_key = adata.uns["AA_config"]["obsm_key"]
    n_dimensions = adata.uns["AA_config"]["n_dimensions"]
    X = adata.obsm[obsm_key][:, n_dimensions].astype(np.float32)

    init = _normalize_init_literal(init)
    optim = _normalize_optim_literal(optim)
    weight = _normalize_weight_literal(weight)
    coreset_algorithm = _normalize_coreset_literal(coreset_algorithm)
    max_iter = max_iter if max_iter is not None else DEFAULT_MAX_ITER
    rel_tol = rel_tol if rel_tol is not None else DEFAULT_REL_TOL

    ensure_archetype_config_keys(adata, uns_keys=("AA_selection_metrics",))
    cache = adata.uns.get("AA_selection_metrics")
    if cache is not None and not isinstance(cache, dict):
        raise ValueError("`adata.uns['AA_selection_metrics']` must be a dictionary if present.")

    metrics_frames: list[pd.DataFrame] = []

    base_compute_kwargs = {
        "init": init,
        "optim": optim,
        "weight": weight,
        "max_iter": max_iter,
        "early_stopping": early_stopping,
        "rel_tol": rel_tol,
        "coreset_algorithm": coreset_algorithm,
        "coreset_fraction": coreset_fraction,
        "coreset_size": coreset_size,
        "delta": delta,
        "verbose": verbose,
    }

    def _compute_best_result(
        n_archetypes: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float] | np.ndarray, float]:
        if save_to_anndata:
            payload = compute_archetypes(
                adata=adata,
                n_archetypes=n_archetypes,
                n_restarts=n_restarts,
                seed=seed,
                save_to_anndata=True,
                return_result=True,
                archetypes_only=False,
                force_recompute=force_recompute,
                **base_compute_kwargs,
                **optim_kwargs,
            )

        else:
            payload = compute_archetypes(
                adata=adata,
                n_archetypes=n_archetypes,
                n_restarts=n_restarts,
                seed=seed,
                save_to_anndata=False,
                return_result=True,
                archetypes_only=False,
                force_recompute=force_recompute,
                **base_compute_kwargs,
                **optim_kwargs,
            )

        if payload is None:
            raise RuntimeError("Failed to retrieve archetypal analysis results for selection metrics.")

        return payload

    for n_archetypes in n_archetypes_seq:
        archetype_config = ArchetypeConfig(
            n_archetypes=n_archetypes,
            init=init,
            optim=optim,
            weight=weight,
            max_iter=max_iter,
            rel_tol=rel_tol,
            early_stopping=early_stopping,
            coreset_algorithm=coreset_algorithm,
            coreset_fraction=coreset_fraction,
            coreset_size=coreset_size,
            delta=delta,
            seed=seed,
            optim_kwargs=optim_kwargs,
            obsm_key=obsm_key,
            n_dimensions=tuple(n_dimensions),
        )

        cached_df: pd.DataFrame | None = None
        if cache is not None and archetype_config in cache and not force_recompute:
            candidate = cache[archetype_config]
            if not isinstance(candidate, pd.DataFrame):
                raise ValueError(
                    "`adata.uns['AA_selection_metrics']` must contain pandas DataFrames for each configuration."
                )
            if candidate.empty:
                cached_df = None
            elif ("n_restarts" in candidate.columns and candidate["n_restarts"].iloc[0] != n_restarts) or (
                "seed" in candidate.columns and candidate["seed"].iloc[0] != seed
            ):
                cached_df = None
            else:
                cached_df = candidate

        if cached_df is None:
            A, _B, Z, rss_trace, varexpl = _compute_best_result(n_archetypes=n_archetypes)
            rss_arr = np.asarray(rss_trace, dtype=np.float64).reshape(-1)
            rss_full = float(rss_arr[-1])
            X_tilde = A @ Z
            ic = float(compute_IC(X=X, X_tilde=X_tilde, n_archetypes=n_archetypes))

            record = {
                "k": n_archetypes,
                "n_archetypes": n_archetypes,
                "n_restarts": n_restarts,
                "seed": seed,
                "varexpl": float(varexpl),
                "IC": ic,
                "RSS": rss_full,
            }

            df = pd.DataFrame([record])
            df["seed"] = pd.Categorical(df["seed"])

            if save_to_anndata:
                if cache is None:
                    cache = adata.uns["AA_selection_metrics"] = {}
                cache[archetype_config] = df.copy()

            cached_df = df
        else:
            cached_df = cached_df.copy()

        metrics_frames.append(cached_df)

    result_df = pd.concat(metrics_frames, axis=0, ignore_index=True)

    if return_result:
        return result_df

    return None


def summarize_aa_metrics(adata: anndata.AnnData, /, **filters) -> pd.DataFrame:
    """
    Concatenate cached selection metrics across archetype counts for a single AA configuration.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing cached selection metrics in ``adata.uns["AA_selection_metrics"]``.
    **filters :
        ArchetypeConfig fields used to select which configuration(s) to summarize. All matched entries
        must share identical parameters except for ``n_archetypes``.

    Returns
    -------
    pandas.DataFrame
        Concatenated metrics with an added ``n_archetypes`` column if missing.

    Raises
    ------
    ValueError
        If no cached metrics match the filters, or if the matched configurations differ in fields other than ``n_archetypes``.
    """
    metrics_dict = _ensure_metrics_dict(adata)
    if not metrics_dict:
        raise ValueError("`adata.uns['AA_selection_metrics']` is empty.")

    filters = dict(filters or {})
    matching_items = [(cfg, df) for cfg, df in metrics_dict.items() if not filters or _matches(cfg, filters)]

    if not matching_items:
        raise ValueError(
            "No selection metrics match the provided filters. "
            "Ensure `compute_selection_metrics` was run with matching parameters."
        )

    configs = [cfg for cfg, _ in matching_items]
    reference = configs[0]
    equivalent = set(
        query_configs_by_signature(
            configs,
            reference,
            ignore_fields=("n_archetypes",),
        )
    )

    if any(cfg not in equivalent for cfg in configs):
        raise ValueError(
            "Multiple optimization configurations match the provided filters. "
            "Please add more specific filters (e.g., init, optim, weight)."
        )

    frames: list[pd.DataFrame] = []
    for cfg, df in matching_items:
        if not isinstance(df, pd.DataFrame):
            continue
        copy_df = df.copy()
        if "n_archetypes" not in copy_df.columns:
            copy_df["n_archetypes"] = cfg.n_archetypes
        if "k" not in copy_df.columns:
            copy_df["k"] = cfg.n_archetypes
        frames.append(copy_df)

    if not frames:
        raise ValueError("Matched selection metrics but no tabular data available.")

    return pd.concat(frames, axis=0, ignore_index=True)


@docs.dedent
def compute_bootstrap_variance(
    adata: anndata.AnnData,
    n_bootstrap: int,
    n_archetypes_list: int | list[int] | None = None,
    init: str | None = None,
    optim: str | None = None,
    weight: None | str = None,
    max_iter: int | None = None,
    early_stopping: bool = True,
    rel_tol: float | None = None,
    coreset_algorithm: None | str = None,
    coreset_fraction: float = 0.1,
    coreset_size: None | int = None,
    delta: float = 0.0,
    seed: int = 42,
    save_to_anndata: bool = True,
    return_result: bool = False,
    n_jobs: int = -1,
    verbose: bool = False,
    force_recompute: bool = False,
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
    force_recompute : bool, default `False`
        Recompute bootstrap samples even if cached results for the given configuration already exist.
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

    n_archetypes_list = _validate_n_archetype_list(
        min_k=None,
        max_k=None,
        n_archetypes_list=n_archetypes_list,
        default_range=range(2, 8),
    )

    obsm_key = adata.uns["AA_config"]["obsm_key"]
    n_dimensions = adata.uns["AA_config"]["n_dimensions"]
    X = adata.obsm[obsm_key][:, n_dimensions]

    n_samples, n_features = X.shape
    rng = np.random.default_rng(seed)

    if save_to_anndata:
        ensure_archetype_config_keys(adata, uns_keys=("AA_bootstrap",))
        if "AA_bootstrap" not in adata.uns.keys():
            adata.uns["AA_bootstrap"] = {}

    if return_result:
        df_dict = {}

    # Use the provided values or fall back to the defaults
    init = _normalize_init_literal(init)
    optim = _normalize_optim_literal(optim)
    weight = _normalize_weight_literal(weight)
    max_iter = max_iter if max_iter is not None else DEFAULT_MAX_ITER
    rel_tol = rel_tol if rel_tol is not None else DEFAULT_REL_TOL
    coreset_algorithm = _normalize_coreset_literal(coreset_algorithm)
    verbose = verbose if verbose is not None else False

    for n_archetypes in n_archetypes_list:
        archetype_config = ArchetypeConfig(
            n_archetypes=n_archetypes,
            init=init,
            optim=optim,
            weight=weight,
            max_iter=max_iter,
            rel_tol=rel_tol,
            early_stopping=early_stopping,
            coreset_algorithm=coreset_algorithm,
            coreset_fraction=coreset_fraction,
            coreset_size=coreset_size,
            delta=delta,
            seed=seed,
            optim_kwargs=optim_kwargs,
            obsm_key=obsm_key,
            n_dimensions=tuple(n_dimensions),
        )

        cached_df = None
        if save_to_anndata:
            bootstrap_store = adata.uns.get("AA_bootstrap")
            if bootstrap_store is None:
                bootstrap_store = adata.uns["AA_bootstrap"] = {}
            if not isinstance(bootstrap_store, dict):
                raise ValueError("`adata.uns['AA_bootstrap']` must be a dictionary if present.")

            if archetype_config in bootstrap_store and not force_recompute:
                candidate = bootstrap_store[archetype_config]
                if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                    if "iter" in candidate.columns and candidate["iter"].max() == n_bootstrap:
                        cached_df = candidate

        if cached_df is not None:
            if return_result:
                df_dict[str(n_archetypes)] = cached_df.copy()
            continue

        # Reference archetypes
        _A, _B, ref_Z, _RSS, _varexpl = compute_archetypes(
            adata=adata,
            n_archetypes=n_archetypes,
            init=init,
            optim=optim,
            weight=weight,
            max_iter=max_iter,
            early_stopping=early_stopping,
            rel_tol=rel_tol,
            coreset_algorithm=coreset_algorithm,
            coreset_fraction=coreset_fraction,
            coreset_size=coreset_size,
            delta=delta,
            seed=seed,
            save_to_anndata=save_to_anndata,
            return_result=True,
            archetypes_only=False,
            force_recompute=force_recompute,
            **optim_kwargs,
        )

        idx_bootstrap = rng.choice(n_samples, size=(n_bootstrap, n_samples), replace=True)

        def compute_bootstrap_z(idx, n_archetypes=n_archetypes):
            return (
                AA(
                    n_archetypes=n_archetypes,
                    optim=optim,
                    init=init,
                    weight=weight,
                    max_iter=max_iter,
                    early_stopping=early_stopping,
                    rel_tol=rel_tol,
                    coreset_algorithm=coreset_algorithm,
                    coreset_fraction=coreset_fraction,
                    coreset_size=coreset_size,
                    delta=delta,
                    seed=seed,
                    **optim_kwargs,
                )
                .fit(X[idx, :])
                .Z
            )

        if verbose:
            Z_list = Parallel(n_jobs=n_jobs)(
                delayed(compute_bootstrap_z)(idx)
                for idx in tqdm(idx_bootstrap, total=n_bootstrap, desc=f"Testing {n_archetypes} Archetypes")
            )
        else:
            Z_list = Parallel(n_jobs=n_jobs)(delayed(compute_bootstrap_z)(idx) for idx in idx_bootstrap)

        Z_list = [_align_archetypes(ref_arch=ref_Z.copy(), query_arch=query_Z.copy()) for query_Z in Z_list]

        Z_stack = np.stack(Z_list)
        assert Z_stack.shape == (n_bootstrap, n_archetypes, n_features)

        var_per_archetype_per_coordinate = Z_stack.var(axis=0)
        var_per_archetype = var_per_archetype_per_coordinate.mean(axis=1)
        var_mean = var_per_archetype.mean()

        bootstrap_data = [
            pd.DataFrame(Z, columns=[f"{obsm_key}_{dim}" for dim in n_dimensions]).assign(
                archetype=np.arange(n_archetypes), iter=i + 1
            )
            for i, Z in enumerate(Z_list)
        ]
        bootstrap_df = pd.concat(bootstrap_data)

        df = pd.DataFrame(ref_Z, columns=[f"{obsm_key}_{dim}" for dim in n_dimensions])
        df["archetype"] = np.arange(n_archetypes)
        df["iter"] = 0

        bootstrap_df = pd.concat((bootstrap_df, df), axis=0)
        bootstrap_df["reference"] = bootstrap_df["iter"] == 0
        bootstrap_df["archetype"] = pd.Categorical(bootstrap_df["archetype"])

        bootstrap_df["mean_variance"] = var_mean

        archetype_variance_map = dict(zip(np.arange(n_archetypes), var_per_archetype, strict=False))
        bootstrap_df["variance_per_archetype"] = bootstrap_df["archetype"].astype(int).map(archetype_variance_map)

        if save_to_anndata:
            bootstrap_store[archetype_config] = bootstrap_df

        if return_result:
            df_dict[str(n_archetypes)] = bootstrap_df.copy()

    if return_result:
        return df_dict

    return None


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


def _resolve_aa_result(
    adata: anndata.AnnData, result_filters: Mapping[str, Any] | None = None
) -> tuple[ArchetypeConfig, Mapping[str, Any]]:
    if "AA_results" not in adata.uns:
        raise ValueError("Missing archetypal analysis results in `adata.uns['AA_results']`.")

    ensure_archetype_config_keys(adata, uns_keys=("AA_results",))

    results = adata.uns["AA_results"]

    if not isinstance(results, Mapping) or not results:
        raise ValueError("`adata.uns['AA_results']` must be a non-empty mapping keyed by `ArchetypeConfig`.")

    first_key = next(iter(results))
    if not isinstance(first_key, ArchetypeConfig):
        raise TypeError("`adata.uns['AA_results']` must be keyed by `ArchetypeConfig` instances.")

    config, payload = get_aa_result(adata, return_config=True, **(result_filters or {}))
    if "Z" not in payload:
        raise ValueError("Selected AA result does not contain archetypes ('Z').")
    return config, payload


def compute_t_ratio(
    adata: anndata.AnnData,
    /,
    *,
    result_filters: Mapping[str, Any] | None = None,
    save_to_anndata: bool = True,
    return_result: bool = False,
) -> float | None:  # pragma: no cover
    """
    Compute the t-ratio from an AnnData object that contains archetypes.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with AA configuration and stored archetypes.
    result_filters : Mapping[str, Any] | None, optional
        Filters passed to :func:`~partipy.get_aa_result` to disambiguate which AA result to use when multiple
        configurations are cached. Ignored when `adata.uns["AA_results"]` uses the legacy format.
    save_to_anndata : bool, default `True`
        Whether to store the computed t-ratio in `adata.uns["AA_t_ratio"]`.
    return_result : bool, default `False`
        If True, return the t-ratio value.

    Returns
    -------
    Optional[float]
        The computed t-ratio when `return_result` is True or `save_to_anndata` is False. Otherwise, None.
    """
    # input validation
    _validate_aa_config(adata=adata)

    config, payload = _resolve_aa_result(adata, result_filters=result_filters)

    obsm_key = adata.uns["AA_config"]["obsm_key"]
    n_dimensions = adata.uns["AA_config"]["n_dimensions"]
    X = adata.obsm[obsm_key][:, n_dimensions]

    Z = payload["Z"]
    if Z.shape[0] <= 2:
        raise ValueError("Number of archetypes must be greater than 2")
    t_ratio = _compute_t_ratio(X, Z)

    if save_to_anndata:
        ensure_archetype_config_keys(adata, uns_keys=("AA_t_ratio",))
        store = adata.uns.get("AA_t_ratio")
        if not isinstance(store, dict):
            store = {}
            adata.uns["AA_t_ratio"] = store
        store[config] = t_ratio

    if return_result or not save_to_anndata:
        return t_ratio

    return None


@docs.dedent
def t_ratio_significance(
    adata: anndata.AnnData,
    *,
    n_iter: int = 100,
    seed: int = 42,
    n_jobs: int = -1,
    save_permutation_results: bool = False,
    result_filters: Mapping[str, Any] | None = None,
):  # pragma: no cover
    """
    Assesses the significance of the polytope spanned by the archetypes by comparing the t-ratio of the original data to t-ratios computed from randomized datasets.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing `adata.obsm["X_pca"]` and `adata.uns["AA_config"]["n_dimensions"]`, optionally
        `adata.uns["AA_t_ratio"]`. If `adata.uns["AA_t_ratio"]` doesn't exist it is computed.
    n_iter : int, default `100`
        Number of randomized datasets to generate.
    %(seed)s
    n_jobs : int, default `-1`
        Number of jobs for parallelization. Use -1 to use all available cores.
    result_filters : Mapping[str, Any] | None, optional
        Filters forwarded to :func:`~partipy.get_aa_result` to select which cached AA result is evaluated when multiple
        configurations are present.

    Returns
    -------
    float
        The proportion of randomized datasets with a t-ratio greater than the original t-ratio (p-value).
    """
    # input validation
    _validate_aa_config(adata=adata)

    t_ratio = compute_t_ratio(
        adata,
        result_filters=result_filters,
        save_to_anndata=True,
        return_result=True,
    )
    assert t_ratio is not None

    config, payload = _resolve_aa_result(adata, result_filters=result_filters)
    obsm_key = adata.uns["AA_config"]["obsm_key"]
    n_dimensions = adata.uns["AA_config"]["n_dimensions"]
    X = adata.obsm[obsm_key][:, n_dimensions]

    rss_trace = payload.get("RSS")
    if rss_trace is None:
        raise ValueError("Selected AA result does not contain 'RSS' trace.")
    rss_arr = np.asarray(rss_trace, dtype=np.float64).reshape(-1)
    rss = float(rss_arr[-1])

    _n_samples, n_features = X.shape

    aa_kwargs = {
        "n_archetypes": config.n_archetypes,
        "init": config.init,
        "optim": config.optim,
        "weight": config.weight,
        "max_iter": config.max_iter,
        "early_stopping": config.early_stopping,
        "rel_tol": config.rel_tol,
        "coreset_algorithm": config.coreset_algorithm,
        "coreset_fraction": config.coreset_fraction,
        "coreset_size": config.coreset_size,
        "delta": config.delta,
        "seed": config.seed,
    }

    config_optim_kwargs = dict(config.optim_kwargs)
    aa_kwargs.update(config_optim_kwargs)

    rng_master = np.random.default_rng(seed)
    rng_list = [np.random.default_rng(rng_master.integers(int(1e9))) for _ in range(n_iter)]

    def compute_randomized_metrics(rng_inner):
        X_perm = np.column_stack([rng_inner.permutation(X[:, col_idx]) for col_idx in range(n_features)])
        AA_perm = AA(**aa_kwargs)
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
        ensure_archetype_config_keys(adata, uns_keys=("AA_permutation",))
        store = adata.uns.get("AA_permutation")
        if not isinstance(store, dict):
            store = {}
            adata.uns["AA_permutation"] = store
        store[config] = {"AA_t_ratio": t_ratios_perm, "rss": rss_perm}

    # Calculate the p-value
    t_ratio_p_value = 1 - np.mean(np.abs(1 - t_ratio) < np.abs(1 - t_ratios_perm))
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
    _ref_idx, query_idx = linear_sum_assignment(euclidean_d)

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
    early_stopping: bool = True,
    rel_tol: float | None = None,
    coreset_algorithm: None | str = None,
    coreset_fraction: float = 0.1,
    coreset_size: None | int = None,
    delta: float = 0.0,
    verbose: bool | None = None,
    seed: int = 42,
    n_jobs: int = -1,
    save_to_anndata: bool = True,
    return_result: bool = False,
    archetypes_only: bool = False,
    force_recompute: bool = False,
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
    %(early_stopping)s
    %(rel_tol)s
    %(coreset_algorithm)s
    %(coreset_fraction)s
    %(coreset_size)s
    %(delta)s
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
    # Input validation
    _validate_aa_config(adata=adata)

    rng = np.random.default_rng(seed)
    seeds = rng.choice(a=int(1e9), size=n_restarts, replace=False)

    # Use the provided values or fall back to the defaults
    init = _normalize_init_literal(init)
    optim = _normalize_optim_literal(optim)
    weight = _normalize_weight_literal(weight)
    max_iter = max_iter if max_iter is not None else DEFAULT_MAX_ITER
    rel_tol = rel_tol if rel_tol is not None else DEFAULT_REL_TOL
    coreset_algorithm = _normalize_coreset_literal(coreset_algorithm)
    verbose = verbose if verbose is not None else False

    # Extract the data matrix used to fit the archetypes
    obsm_key = adata.uns["AA_config"]["obsm_key"]
    n_dimensions = adata.uns["AA_config"]["n_dimensions"]
    X = adata.obsm[obsm_key][:, n_dimensions]
    X = X.astype(np.float32)

    # Generate hash for the optimization and data parameters
    archetype_config = ArchetypeConfig(
        n_archetypes=n_archetypes,
        init=init,
        optim=optim,
        weight=weight,
        max_iter=max_iter,
        rel_tol=rel_tol,
        early_stopping=early_stopping,
        coreset_algorithm=coreset_algorithm,
        coreset_fraction=coreset_fraction,
        coreset_size=coreset_size,
        delta=delta,
        seed=seed,
        optim_kwargs=optim_kwargs,
        obsm_key=obsm_key,
        n_dimensions=tuple(n_dimensions),
    )

    # ------------------ cache short-circuit ------------------
    ensure_archetype_config_keys(adata, uns_keys=("AA_results",))
    # Create the container if absent
    if "AA_results" not in adata.uns:
        adata.uns["AA_results"] = {}

    if archetype_config in adata.uns["AA_results"] and not force_recompute:
        if verbose:
            print("Using cached AA result from `adata.uns['AA_results']`.")

        # Return from cache according to caller's request
        cached = adata.uns["AA_results"][archetype_config]
        if archetypes_only:
            if "Z" not in cached:
                raise ValueError(
                    "Cached AA entry exists but does not contain 'Z'. This should not happen; try recomputing."
                )
            return cached["Z"]
        else:
            needed = {"A", "B", "Z", "RSS", "varexpl"}
            if not needed.issubset(cached.keys()):
                raise ValueError(
                    "AA result already cached only with 'Z', but full outputs were requested "
                    "(A, B, Z, RSS, varexpl). Recompute with `archetypes_only=False` to cache full results."
                )
            return cached["A"], cached["B"], cached["Z"], cached["RSS"], cached["varexpl"]

    # ------------------ compute (cache miss or forced) ------------------
    def _compute_archetypes_single(seed_val: int):
        model = AA(
            n_archetypes=n_archetypes,
            init=init,
            optim=optim,
            weight=weight,
            max_iter=max_iter,
            rel_tol=rel_tol,
            early_stopping=early_stopping,
            coreset_algorithm=coreset_algorithm,
            coreset_fraction=coreset_fraction,
            coreset_size=coreset_size,
            delta=delta,
            verbose=verbose,
            seed=seed_val,
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
            "seed": seed_val,
        }

    if n_jobs == 1:
        results_list = [_compute_archetypes_single(seed_val=s) for s in seeds]
    else:
        results_list = Parallel(n_jobs=n_jobs)(delayed(_compute_archetypes_single)(seed_val=s) for s in seeds)

    # Select the run with maximal variance explained
    argmax = int(np.argmax([r["varexpl"] for r in results_list]))
    best = results_list[argmax]

    # ------------------ persist or return ------------------
    if save_to_anndata:
        if archetypes_only:
            adata.uns["AA_results"][archetype_config] = {
                "Z": best["Z"],
                "cell_index": adata.obs.index.to_numpy().copy(),
            }
        else:
            adata.uns["AA_results"][archetype_config] = {
                "A": best["A"],
                "B": best["B"],
                "Z": best["Z"],
                "RSS": best["RSS"],
                "RSS_full": best["RSS_full"],
                "varexpl": best["varexpl"],
                "cell_index": adata.obs.index.to_numpy().copy(),
            }

    if return_result:
        if archetypes_only:
            return best["Z"]
        else:
            return best["A"], best["B"], best["Z"], best["RSS"], best["varexpl"]
    else:
        return None


def _ensure_results_dict(adata) -> dict:
    ensure_archetype_config_keys(adata, uns_keys=("AA_results",))
    try:
        d = adata.uns["AA_results"]
    except KeyError as kerr:
        raise KeyError("No AA_results found in `adata.uns`.") from kerr
    if not isinstance(d, dict) or len(d) == 0:
        raise ValueError("`adata.uns['AA_results']` is empty or not a dict.")
    first_key = next(iter(d))
    if not isinstance(first_key, ArchetypeConfig):
        raise TypeError("`adata.uns['AA_results']` must be keyed by `ArchetypeConfig` instances.")
    return d


def _normalize_query_kv(k: str, v: Any) -> Any:
    # Normalize query values to match ArchetypeConfig’s field types
    if k == "n_dimensions":
        # Accept list[int] in query; stored key uses tuple[int,...]
        if isinstance(v, list):
            return tuple(v)
        return v
    if k == "optim":
        return canonicalize_optim(v)
    if k == "optim_kwargs":
        # Accept dict-like and normalize to subset-dict for matching
        if isinstance(v, Mapping):
            return dict(v)
        raise TypeError("optim_kwargs filter must be a mapping (e.g., dict).")
    return v


def _matches(config, query: dict[str, Any]) -> bool:
    for k, v in query.items():
        v = _normalize_query_kv(k, v)
        if not hasattr(config, k):
            raise KeyError(f"Unknown field in query: {k!r}")
        cfg_val = getattr(config, k)

        if k == "optim_kwargs":
            # config.optim_kwargs is already frozen+sorted; convert to dict for subset check
            cfg_ok = dict(cfg_val) if isinstance(cfg_val, tuple | list) else dict(cfg_val.items())
            for kk, vv in v.items():
                if kk not in cfg_ok or cfg_ok[kk] != vv:
                    return False
            continue

        # Plain equality for all other fields
        if cfg_val != v:
            return False
    return True


def get_aa_result(adata, /, *, return_config: bool = False, **filters):
    """
    Fuzzy-get: with no filters require exactly one result; with filters require exactly one match.
    Filters may be any ArchetypeConfig field (e.g., delta=0.5, init="plus_plus", n_dimensions=[0,1,2]).
    For optim_kwargs, subset matching is used (e.g., optim_kwargs={"lr": 1e-2}).
    """
    results = _ensure_results_dict(adata)

    if not filters:
        if len(results) == 1:
            ((cfg, payload),) = results.items()
            return (cfg, payload) if return_config else payload
        else:
            raise ValueError(
                f"Multiple AA results present ({len(results)}). "
                "Specify filters (e.g., delta=..., init=..., n_archetypes=...)."
            )

    # Apply filters
    matches = [(cfg, payload) for cfg, payload in results.items() if _matches(cfg, filters)]

    if len(matches) == 0:
        raise ValueError(f"No AA result matches filters: {filters!r}")
    if len(matches) > 1:
        # Provide a short disambiguation hint (list a few distinguishing fields)
        examples = [
            {
                "n_archetypes": m[0].n_archetypes,
                "init": m[0].init,
                "optim": m[0].optim,
                "weight": m[0].weight,
                "delta": m[0].delta,
                "seed": m[0].seed,
            }
            for m in matches[:5]
        ]
        raise ValueError(
            f"{len(matches)} AA results match filters {filters!r}. "
            f"Please add more filters. First few matches: {examples}"
        )

    cfg, payload = matches[0]
    return (cfg, payload) if return_config else payload


def delete_aa_result(adata, /, **filters):
    """
    Fuzzy-delete: same matching rules as get_aa_result.
    Deletes the uniquely identified entry from adata.uns['AA_results'] and returns its payload.
    """
    results = _ensure_results_dict(adata)

    # Reuse the matching logic to locate exactly one item
    # but do not rely on get_aa_result's return since we need the key to pop
    if not filters and len(results) == 1:
        ((cfg, payload),) = results.items()
        results.pop(cfg)
        return payload

    matches = [(cfg, payload) for cfg, payload in results.items() if _matches(cfg, filters)]

    if len(matches) == 0:
        raise ValueError(f"No AA result matches filters: {filters!r}")
    if len(matches) > 1:
        examples = [
            {
                "n_archetypes": m[0].n_archetypes,
                "init": m[0].init,
                "optim": m[0].optim,
                "weight": m[0].weight,
                "delta": m[0].delta,
                "seed": m[0].seed,
            }
            for m in matches[:5]
        ]
        raise ValueError(
            f"{len(matches)} AA results match filters {filters!r}. "
            f"Please add more filters. First few matches: {examples}"
        )

    cfg, payload = matches[0]
    results.pop(cfg)
    return payload


def _ensure_cell_weights_dict(adata) -> dict:
    ensure_archetype_config_keys(adata, uns_keys=("AA_cell_weights",))
    weights = adata.uns.get("AA_cell_weights")
    if weights is None:
        raise ValueError(
            "No AA cell weights found in `adata.uns['AA_cell_weights']`. Run `compute_archetype_weights` first."
        )
    if not isinstance(weights, dict) or len(weights) == 0:
        raise ValueError("`adata.uns['AA_cell_weights']` is empty or not a dict.")
    return weights


def _ensure_metrics_dict(adata) -> dict:
    ensure_archetype_config_keys(adata, uns_keys=("AA_selection_metrics",))
    metrics = adata.uns.get("AA_selection_metrics")
    if metrics is None:
        raise ValueError(
            "No AA selection metrics found in `adata.uns['AA_selection_metrics']`. "
            "Run `compute_selection_metrics` first."
        )
    if not isinstance(metrics, dict) or len(metrics) == 0:
        raise ValueError("`adata.uns['AA_selection_metrics']` is empty or not a dict.")
    return metrics


def _ensure_bootstrap_dict(adata) -> dict:
    ensure_archetype_config_keys(adata, uns_keys=("AA_bootstrap",))
    bootstrap = adata.uns.get("AA_bootstrap")
    if bootstrap is None:
        raise ValueError(
            "No AA bootstrap results found in `adata.uns['AA_bootstrap']`. Run `compute_bootstrap_variance` first."
        )
    if not isinstance(bootstrap, dict) or len(bootstrap) == 0:
        raise ValueError("`adata.uns['AA_bootstrap']` is empty or not a dict.")
    return bootstrap


def get_aa_cell_weights(adata, /, *, return_config: bool = False, **filters):
    """
    Retrieve archetype-analysis cell weights stored in ``adata.uns['AA_cell_weights']``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing stored cell weights.
    return_config : bool, default ``False``
        If True, return a tuple ``(ArchetypeConfig, weights)``, otherwise return only the weights.
    **filters :
        ArchetypeConfig fields used to disambiguate which weight matrix to return (e.g., ``n_archetypes=5``).

    Returns
    -------
    numpy.ndarray | tuple[ArchetypeConfig, numpy.ndarray]
        The cell-weight matrix, optionally with the associated ``ArchetypeConfig``.
    """
    weights_dict = _ensure_cell_weights_dict(adata)

    if not filters:
        if len(weights_dict) == 1:
            ((cfg, weights),) = weights_dict.items()
            return (cfg, weights) if return_config else weights
        raise ValueError(
            f"Multiple AA cell weight matrices present ({len(weights_dict)}). "
            "Specify filters (e.g., n_archetypes=..., init=...)."
        )

    matches = [(cfg, weights) for cfg, weights in weights_dict.items() if _matches(cfg, filters)]

    if len(matches) == 0:
        raise ValueError(f"No AA cell weights match filters: {filters!r}")
    if len(matches) > 1:
        examples = [
            {
                "n_archetypes": m[0].n_archetypes,
                "init": m[0].init,
                "optim": m[0].optim,
                "weight": m[0].weight,
                "delta": m[0].delta,
                "seed": m[0].seed,
            }
            for m in matches[:5]
        ]
        raise ValueError(
            f"{len(matches)} AA cell weight matrices match filters {filters!r}. "
            f"Please add more filters. First few matches: {examples}"
        )

    cfg, weights = matches[0]
    return (cfg, weights) if return_config else weights


def get_aa_metrics(adata, /, *, return_config: bool = False, **filters):
    """
    Retrieve archetypal analysis selection metrics stored in ``adata.uns['AA_selection_metrics']``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing AA selection diagnostics.
    return_config : bool, default ``False``
        If True, return a tuple ``(ArchetypeConfig, dataframe)``, otherwise return only the DataFrame.
    **filters :
        ArchetypeConfig fields used to disambiguate which metrics table to return (e.g., ``n_archetypes=5``).

    Returns
    -------
    pandas.DataFrame | tuple[ArchetypeConfig, pandas.DataFrame]
        The selection-metrics DataFrame, optionally with the associated ``ArchetypeConfig``.
    """
    metrics_dict = _ensure_metrics_dict(adata)

    if not filters:
        if len(metrics_dict) == 1:
            ((cfg, df),) = metrics_dict.items()
            return (cfg, df) if return_config else df
        raise ValueError(
            f"Multiple AA selection metrics present ({len(metrics_dict)}). "
            "Specify filters (e.g., n_archetypes=..., init=...)."
        )

    matches = [(cfg, df) for cfg, df in metrics_dict.items() if _matches(cfg, filters)]

    if len(matches) == 0:
        raise ValueError(f"No AA selection metrics match filters: {filters!r}")
    if len(matches) > 1:
        examples = [
            {
                "n_archetypes": m[0].n_archetypes,
                "init": m[0].init,
                "optim": m[0].optim,
                "weight": m[0].weight,
                "delta": m[0].delta,
                "seed": m[0].seed,
            }
            for m in matches[:5]
        ]
        raise ValueError(
            f"{len(matches)} AA selection metric tables match filters {filters!r}. "
            f"Please add more filters. First few matches: {examples}"
        )

    cfg, df = matches[0]
    return (cfg, df) if return_config else df


def get_aa_bootstrap(adata, /, *, return_config: bool = False, **filters):
    """
    Retrieve bootstrap variance results stored in ``adata.uns['AA_bootstrap']``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing cached bootstrap runs.
    return_config : bool, default ``False``
        If True, return a tuple ``(ArchetypeConfig, dataframe)``, otherwise return only the DataFrame.
    **filters :
        ArchetypeConfig fields used to disambiguate which bootstrap table to return (e.g., ``n_archetypes=5``).

    Returns
    -------
    pandas.DataFrame | tuple[ArchetypeConfig, pandas.DataFrame]
        The bootstrap DataFrame, optionally with the associated ``ArchetypeConfig``.
    """
    bootstrap_dict = _ensure_bootstrap_dict(adata)

    if not filters:
        if len(bootstrap_dict) == 1:
            ((cfg, df),) = bootstrap_dict.items()
            return (cfg, df) if return_config else df
        raise ValueError(
            f"Multiple AA bootstrap entries present ({len(bootstrap_dict)}). "
            "Specify filters (e.g., n_archetypes=..., init=...)."
        )

    matches = [(cfg, df) for cfg, df in bootstrap_dict.items() if _matches(cfg, filters)]

    if len(matches) == 0:
        raise ValueError(f"No AA bootstrap entries match filters: {filters!r}")
    if len(matches) > 1:
        examples = [
            {
                "n_archetypes": m[0].n_archetypes,
                "init": m[0].init,
                "optim": m[0].optim,
                "weight": m[0].weight,
                "delta": m[0].delta,
                "seed": m[0].seed,
            }
            for m in matches[:5]
        ]
        raise ValueError(
            f"{len(matches)} AA bootstrap entries match filters {filters!r}. "
            f"Please add more filters. First few matches: {examples}"
        )

    cfg, df = matches[0]
    return (cfg, df) if return_config else df
