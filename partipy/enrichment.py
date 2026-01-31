"""Functions to calculate which features (e.g. genes or covariates) are enriched at each archetype."""

from collections.abc import Mapping
from typing import Any, Literal

import anndata
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.spatial.distance import cdist
from scipy.stats import hypergeom, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests

from .io import ensure_archetype_config_keys
from .paretoti import _resolve_aa_result, _validate_aa_config, _validate_aa_results, get_aa_cell_weights


def compute_archetype_weights(
    adata: anndata.AnnData,
    mode: str = "automatic",
    length_scale: None | float = None,
    save_to_anndata: bool = True,
    result_filters: Mapping[str, Any] | None = None,
) -> None | np.ndarray:
    """
    Calculate weights for the data points based on their distance to archetypes using a squared exponential kernel.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing the data and archetypes. The data should be stored in `adata.obsm[obsm_key]`
        and the archetypes in `adata.uns["AA_results"]["Z"]`.
    mode : str, default `automatic`
        The mode for determining the length scale of the kernel:
        - "automatic": The length scale is calculated as half the median distance from the data centroid to the archetypes.
        - "manual": The length scale is provided by the user via the `length_scale` parameter.
    length_scale : float, default `None`
        If `mode="manual"`, this is the user-defined length scale for the kernel. If `mode="automatic"`, it is calculated automatically.
    save_to_anndata : bool, default `True`
        If `True`, the weights are saved to `adata.uns["AA_cell_weights"]` under the resolved AA configuration. If `False`,
        the weights are returned as a NumPy array.
    result_filters : Mapping[str, Any] | None, default `None`
        Filters forwarded to ``_resolve_aa_result`` to select the AA configuration for which weights are computed.

    Returns
    -------
    np.ndarray
        - If `save_to_anndata` is True, weights are stored in ``adata.uns["AA_cell_weights"]`` and ``None`` is returned.
        - If `save_to_anndata` is False, the computed weights are returned as a NumPy array.
    """
    # input validation
    _validate_aa_config(adata=adata)
    _validate_aa_results(adata=adata)

    config, payload = _resolve_aa_result(adata, result_filters=result_filters)

    obsm_key = config.obsm_key
    n_dimensions = list(config.n_dimensions)
    X = adata.obsm[obsm_key][:, n_dimensions]
    Z = payload.get("Z")
    if Z is None:
        raise ValueError("Matched AA payload does not contain 'Z'.")

    # Calculate or validate length_scale based on mode
    if mode == "automatic":
        centroid = np.mean(X, axis=0).reshape(1, -1)
        length_scale = np.median(cdist(centroid, Z)) / 2
    elif mode == "manual":
        if length_scale is None:
            raise ValueError("For 'manual' mode, 'length_scale' must be provided.")
    else:
        raise ValueError("Mode must be either 'automatic' or 'manual'.")
    print(f"Applied length scale is {length_scale:.2f}.")

    # Weight calculation
    euclidean_dist = cdist(X, Z)
    weights = np.exp(-(euclidean_dist**2) / (2 * length_scale**2))  # type: ignore[operator]
    weights /= weights.sum(axis=1, keepdims=True)
    weights = weights.astype(np.float32)

    if save_to_anndata:
        ensure_archetype_config_keys(adata, uns_keys=("AA_cell_weights",))
        weights_store = adata.uns.get("AA_cell_weights")
        if weights_store is None or not isinstance(weights_store, Mapping):
            adata.uns["AA_cell_weights"] = {config: weights}
        else:
            updated = dict(weights_store)
            updated[config] = weights
            adata.uns["AA_cell_weights"] = updated
        return None

    return weights


def compute_archetype_expression(
    adata: anndata.AnnData,
    layer: str | None = None,
    result_filters: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Calculate a weighted average gene expression profile for each archetype.

    This function computes the weighted average of gene expression across cells for each archetype.
    The weights should be based on the distance of cells to the archetypes, as computed by `calculate_weights`.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing the gene expression data and weights. The weights should be stored in
        ``adata.uns["AA_cell_weights"]`` keyed by the corresponding ``ArchetypeConfig``.
    layer : str, default `None`
        The layer of the AnnData object to use for gene expression. If `None`, ``adata.X`` is used. For Pareto analysis of AA data,
        z-scaled data is recommended.
    result_filters : Mapping[str, Any] | None, default `None`
        Filters applied to ``ArchetypeConfig`` entries to select the optimization configuration whose weights should be used.

    Returns
    -------
    pd.DataFrame
        A DataFrame of shape (n_archetypes, n_genes) with weighted pseudobulk expression profiles.
    """
    filters = dict(result_filters or {})
    cfg, weights = get_aa_cell_weights(adata, return_config=True, **filters)
    weights = weights.T

    if layer is None:
        expr = adata.X
    elif layer not in adata.layers:
        raise ValueError("Invalid layer")
    else:
        expr = adata.layers[layer]

    pseudobulk = weights @ expr

    pseudobulk_df = pd.DataFrame(pseudobulk, columns=adata.var_names)
    pseudobulk_df.columns.name = None

    return pseudobulk_df


def extract_enriched_processes(
    est: pd.DataFrame,
    pval: pd.DataFrame,
    order: str = "desc",
    n: int = 20,
    p_threshold: float = 0.05,
) -> dict[int, pd.DataFrame]:
    """
    Extract top enriched biological processes for each archetype based on significance and enrichment score.

    This function filters and ranks biological processes using enrichment estimates (`est`) and p-values (`pval`)
    from decoupler output. For each archetype, it selects the top `n` processes with p-values below `p_threshold`,
    optionally sorting by the highest or lowest enrichment scores. It also computes a "specificity" score indicating
    how uniquely enriched a process is for a given archetype compared to others.

    Parameters
    ----------
    est : `pd.DataFrame`
        A DataFrame of shape (n_archetypes, n_processes) containing the estimated enrichment scores
        for each process and archetype.
    pval : `pd.DataFrame`
        A DataFrame of shape (n_archetypes, n_processes) containing the p-values corresponding to
        the enrichment scores in `est`.
    order : str, default `"desc"`
        The sorting order for selecting the top processes. Options are:

        - "desc": Selects the top `n` processes with the highest enrichment scores.
        - "asc": Selects the top `n` processes with the lowest enrichment scores.
    n : int, default `20`
        The number of top processes to extract per archetype.
    p_threshold : float, default `0.05`
        The p-value threshold for filtering processes. Only processes with p-values below this
        threshold are considered.

    Returns
    -------
    dict[int, pd.DataFrame]
        A dictionary mapping each archetype index to a DataFrame of the top `n` enriched processes.
        Each DataFrame has the following columns:
        - "Process": Name of the biological process.
        - "{archetype indices}": Enrichment score for that process.
        - "specificity": A score indicating how uniquely enriched the process is in the given archetype.
    """
    # Validate input
    if not ((p_threshold > 0.0) and (p_threshold <= 1.0)):
        raise ValueError("`p_threshold` must be a valid p value")
    if est.shape != pval.shape:
        raise ValueError("`est` and `pval` must have the same shape.")

    if order not in ["desc", "asc"]:
        raise ValueError("`order` must be either 'desc' or 'asc'.")

    results = {}
    for arch_idx in range(est.shape[0]):
        # Filter processes based on p-value threshold
        significant_processes = pval.columns[pval.iloc[arch_idx] < p_threshold]

        # compute specificity score
        top_processes = est[significant_processes].T
        arch_z_score = top_processes[[str(arch_idx)]].values
        other_z_scores = top_processes[[c for c in top_processes.columns if c != str(arch_idx)]].values
        top_processes["specificity"] = (arch_z_score - other_z_scores).min(axis=1)

        # filter
        if order == "desc":
            top_processes = top_processes.nlargest(n=n, columns=f"{arch_idx}").reset_index(names="Process")
        else:
            top_processes = top_processes.nsmallest(n=n, columns=f"{arch_idx}").reset_index(names="Process")

        results[arch_idx] = top_processes

    return results


def extract_specific_processes(
    est: pd.DataFrame,
    pval: pd.DataFrame,
    n: int = 20,
    p_threshold: float = 0.05,
) -> dict[int, pd.DataFrame]:
    """
    Extract the top biological processes that are uniquely enriched in each archetype.

    This function identifies the top `n` biological processes for each archetype based on their
    enrichment scores (`est`) and associated p-values (`pval`). Only processes with p-values below
    `p_threshold` in a given archetype are considered. A "specificity" score is computed for each
    process, reflecting how much more enriched it is in the target archetype compared to others.

    Parameters
    ----------
    est : `pd.DataFrame`
        A DataFrame of shape (n_archetypes, n_processes) containing the estimated enrichment scores
        for each process and archetype.
    pval : `pd.DataFrame`
        A DataFrame of shape (n_archetypes, n_processes) containing the p-values corresponding to
        the enrichment scores in `est`.
    n : int, default: `20`
        The number of top processes to extract per archetype.
    p_threshold : float, default: `0.05`
        The p-value threshold for filtering processes. Only processes with p-values below this
        threshold are considered.

    Returns
    -------
    dict : [int, pd.DataFrame]
        A dictionary mapping each archetype index to a DataFrame containing the top `n` processes
        specific to that archetype. Each DataFrame includes:
        - "Process": Name of the biological process.
        - "{archetype indices}": Enrichment score in the given archetype.
        - "specificity": Score indicating how uniquely enriched the process is compared to other archetypes.
    """
    # Validate input
    if not ((p_threshold > 0.0) and (p_threshold <= 1.0)):
        raise ValueError("`p_threshold` must be a valid p value")
    if est.shape != pval.shape:
        raise ValueError("`est` and `pval` must have the same shape.")

    results = {}
    for arch_idx in range(est.shape[0]):
        # Filter processes based on p-value threshold
        significant_processes = pval.columns[pval.iloc[arch_idx] < p_threshold]

        # compute specificity score
        top_processes = est[significant_processes].T
        arch_z_score = top_processes[[str(arch_idx)]].values
        other_z_scores = top_processes[[c for c in top_processes.columns if c != str(arch_idx)]].values
        top_processes["specificity"] = (arch_z_score - other_z_scores).min(axis=1)
        top_processes = top_processes.nlargest(n=n, columns="specificity").reset_index(names="Process")

        results[arch_idx] = top_processes.copy()

    return results


def compute_meta_enrichment(
    adata: anndata.AnnData,
    meta_col: str,
    datatype: str = "automatic",
    result_filters: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Compute the enrichment of metadata categories across archetypes.

    This function estimates how enriched each metadata category is within each archetype using
    a weighted average approach. Weights are based on each cell’s contribution to each archetype
    (``adata.uns["AA_cell_weights"]``). It supports both categorical and continuous metadata.

    Steps for categorical data:
        1. One-hot encode the metadata column from `adata.obs[meta_col]`.
        2. Normalize the metadata so that the sum for each category equals 1 (column-wise).
        3. Compute weighted enrichment using cell weights.
        4. Normalize the resulting enrichment scores across metadata categories for each archetype (row-wise).

    Steps for continuous data:
        1. Compute the weighted average of the metadata per archetype.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with categorical metadata in `adata.obs[meta_col]` and archetype weights
        stored in ``adata.uns["AA_cell_weights"]``.
    meta_col : str
        The name of the categorical metadata column in `adata.obs` to use for enrichment analysis.
    datatype : str, default `automatic`
        Specifies how to interpret the metadata column:
        - "automatic": infers type based on column dtype.
        - "categorical": treats the column as categorical and one-hot encodes it.
        - "continuous": treats the column as numeric and computes weighted averages.
    result_filters : Mapping[str, Any] | None, default `None`
        Filters applied to ``ArchetypeConfig`` entries to select the optimization configuration whose weights should be used.

    Returns
    -------
    pd.DataFrame
        A DataFrame of shape (n_archetypes, n_categories) for categorical data or
        (n_archetypes, 1) for continuous data, containing normalized enrichment scores
        or weighted averages respectively.
    """
    if meta_col not in adata.obs:
        raise ValueError("Metadata column does not exist")
    metadata = adata.obs[meta_col]
    _, weights = get_aa_cell_weights(adata, return_config=True, **dict(result_filters or {}))
    weights = weights.T

    if datatype == "automatic":
        if pd.api.types.is_numeric_dtype(metadata):
            mode = "continuous"
            metadata = metadata.to_numpy(dtype="float")
        elif pd.api.types.is_string_dtype(metadata):
            mode = "categorical"
        else:
            raise ValueError("Not a valid data type detected")
    elif datatype == "continuous" or datatype == "categorical":
        mode = datatype
    else:
        raise ValueError("Not a valid data type")

    if mode == "categorical":
        # One-hot encoding of metadata
        df_encoded = pd.get_dummies(metadata).astype(float)
        # Normalization
        df_encoded = df_encoded / df_encoded.values.sum(axis=0, keepdims=True)

        # Compute weighted enrichment
        weighted_meta = np.einsum("ij,jk->ik", weights, df_encoded)

        # Normalization
        weighted_meta = weighted_meta / np.sum(weighted_meta, axis=1, keepdims=True)
        weighted_meta_df = pd.DataFrame(weighted_meta, columns=df_encoded.columns)

    elif mode == "continuous":
        metadata = np.asarray(metadata, dtype=float).reshape(-1, 1)
        weights = weights / weights.sum(axis=1, keepdims=True)  # ensure weights sum to 1 for the archetypes

        # Compute weighted enrichment
        weighted_meta = np.einsum("ij,jk->ik", weights, metadata)

        weighted_meta_df = pd.DataFrame(weighted_meta, columns=[meta_col])

    return weighted_meta_df


def compute_quantile_based_gene_enrichment(
    adata: anndata.AnnData,
    result_filters: Mapping[str, Any] | None = None,
    n_bins: int = 10,
    test: Literal["ttest", "wilcox"] = "wilcox",
    *,
    alpha: float = 0.10,
    p_adjust_scope: Literal["global", "per_archetype"] = "per_archetype",
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    require_max_in_bin0: bool = True,
    require_positive_effect: bool = True,
):
    """
    Quantile-based distance-bin enrichment for genes

    Reference: https://github.com/AlonLabWIS/ParTI/blob/f51c5af896bbd12c8da0e1104d3ff7557bea0c30/ContinuousEnrichment.m

    Literature: {cite:p}`Adler2019,Korem2015,Hart2015`

    Implements the ParTI-style default contrast (closest bin vs all remaining cells)
    with:
      - per-gene Mann–Whitney U (Wilcoxon rank-sum) or Welch t-test,
      - BH-FDR correction,
      - optional ParTI-like "maximal in closest bin" criterion computed from binned medians,
      - optional positive-effect constraint (median_diff > 0 by default).
    """
    # --- input validation / config resolution (your project-specific helpers) ---
    _validate_aa_config(adata=adata)
    _validate_aa_results(adata=adata)

    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")

    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"Unsupported alternative: {alternative}")

    config, payload = _resolve_aa_result(adata, result_filters=result_filters)

    obsm_key = config.obsm_key
    n_dimensions = list(config.n_dimensions)
    X = adata.obsm[obsm_key][:, n_dimensions]

    Z = payload.get("Z")
    if Z is None:
        raise ValueError("Matched AA payload does not contain 'Z'.")

    # distances: (n_cells, n_archetypes)
    euclidean_dist = cdist(X, Z)

    enrichment_rows: list[dict[str, Any]] = []

    n_cells = adata.n_obs
    gene_names = list(adata.var.index)

    for archetype_index in range(Z.shape[0]):
        d_vec = euclidean_dist[:, archetype_index]

        # ranks 0..n_cells-1 (ties broken by sort order)
        r_vec = np.argsort(np.argsort(d_vec))

        # equal-size bins in {0,...,n_bins-1}
        bin_id = (r_vec * n_bins) // n_cells
        bin_id = np.minimum(bin_id, n_bins - 1)  # defensive

        mask0 = bin_id == 0
        mask_rest = ~mask0

        # guard: if something pathological happens (shouldn't for n_bins<=n_cells)
        if mask0.sum() < 2 or mask_rest.sum() < 2:
            raise ValueError(
                f"Too few cells in foreground/background for archetype {archetype_index}: "
                f"n0={mask0.sum()}, n1={mask_rest.sum()}."
            )

        for gene in gene_names:
            # extract gene vector (n_cells,)
            gene_vec = adata[:, gene].X
            if sp.issparse(gene_vec):
                gene_vec = np.asarray(gene_vec.todense()).ravel()
            else:
                gene_vec = np.asarray(gene_vec).ravel()

            x0 = gene_vec[mask0]
            x1 = gene_vec[mask_rest]

            # effect sizes
            mean0, mean1 = float(np.mean(x0)), float(np.mean(x1))
            median0, median1 = float(np.median(x0)), float(np.median(x1))

            mean_diff = mean0 - mean1
            median_diff = median0 - median1

            # ParTI-like "maximal in closest bin" criterion based on binned medians
            # (matches the spirit of ContinuousEnrichment.m)
            max_in_bin0 = True
            if require_max_in_bin0:
                binned_medians = np.empty(n_bins, dtype=float)
                for b in range(n_bins):
                    xb = gene_vec[bin_id == b]
                    # with equal-size bins, xb should be non-empty; still be robust:
                    binned_medians[b] = np.median(xb) if xb.size else -np.inf
                # True iff the (first) maximum is at bin 0
                max_in_bin0 = int(np.nanargmax(binned_medians) == 0) == 1

            # optional positive-effect constraint
            pos_effect = True
            if require_positive_effect:
                # use median as primary effect direction (consistent with max criterion)
                pos_effect = bool(median_diff > 0)

            # significance test (default: closest bin vs rest)
            match test:
                case "ttest":
                    stat, pval = ttest_ind(
                        a=x0,
                        b=x1,
                        equal_var=False,
                        alternative=alternative,
                        nan_policy="omit",
                    )
                case "wilcox":
                    # Mann–Whitney U; use asymptotic/auto method (SciPy >= 1.7)
                    stat, pval = mannwhitneyu(
                        x=x0,
                        y=x1,
                        alternative=alternative,
                        method="auto",
                    )
                case _:
                    raise ValueError(f"{test} not supported")

            enrichment_rows.append(
                {
                    "arch_idx": int(archetype_index),
                    "gene": gene,
                    "stat": float(stat),
                    "pval": float(pval),
                    "mean_diff": float(mean_diff),
                    "median_diff": float(median_diff),
                    "mean_bin0": mean0,
                    "mean_rest": mean1,
                    "median_bin0": median0,
                    "median_rest": median1,
                    "max_in_bin0": bool(max_in_bin0),
                    "pos_effect": bool(pos_effect),
                    # filled later:
                    "pval_adj": np.nan,
                    "signif": False,
                    "enriched": False,
                }
            )

    enrichment_df = pd.DataFrame(enrichment_rows)

    # --- multiple testing correction ---
    if p_adjust_scope == "global":
        _, pval_adj, _, _ = multipletests(enrichment_df["pval"].values, alpha=alpha, method="fdr_bh")
        enrichment_df["pval_adj"] = pval_adj
        enrichment_df["signif"] = enrichment_df["pval_adj"] <= alpha

    elif p_adjust_scope == "per_archetype":
        enrichment_df["pval_adj"] = np.nan
        enrichment_df["signif"] = False

        for _arch_idx, idx in enrichment_df.groupby("arch_idx").groups.items():
            idx = np.asarray(list(idx), dtype=int)
            _, pval_adj, _, _ = multipletests(enrichment_df.loc[idx, "pval"].values, alpha=alpha, method="fdr_bh")
            enrichment_df.loc[idx, "pval_adj"] = pval_adj
            enrichment_df.loc[idx, "signif"] = enrichment_df.loc[idx, "pval_adj"] <= alpha

    else:
        raise ValueError(f"Unsupported p_adjust_scope: {p_adjust_scope}")

    # final "enriched" call: significance + (optional) ParTI-like maximality + (optional) positive effect
    if require_max_in_bin0 and require_positive_effect:
        enrichment_df["enriched"] = enrichment_df["signif"] & enrichment_df["max_in_bin0"] & enrichment_df["pos_effect"]
    elif require_max_in_bin0:
        enrichment_df["enriched"] = enrichment_df["signif"] & enrichment_df["max_in_bin0"]
    elif require_positive_effect:
        enrichment_df["enriched"] = enrichment_df["signif"] & enrichment_df["pos_effect"]
    else:
        enrichment_df["enriched"] = enrichment_df["signif"]

    return enrichment_df


def compute_quantile_based_continuous_enrichment(
    adata: anndata.AnnData,
    colnames: str | list[str],
    n_bins: int = 10,
    test: Literal["ttest", "wilcox"] = "wilcox",
    result_filters: Mapping[str, Any] | None = None,
    *,
    alpha: float = 0.10,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    require_max_in_bin0: bool = True,
    require_positive_effect: bool = True,
    ignore_nans: bool = False,
):
    """
    Quantile-based distance-bin enrichment for continuous obs columns.

    Cells are ranked by distance to each archetype, split into equal-size bins, and
    the closest bin (bin 0) is contrasted against all remaining cells. This mirrors
    compute_quantile_based_gene_enrichment but evaluates numeric ``adata.obs`` columns.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with AA configuration and archetype results.
    colnames : str or list[str]
        One or more numeric columns in ``adata.obs`` to test.
    n_bins : int, default 10
        Number of equal-size distance bins.
    test : {"ttest", "wilcox"}, default "wilcox"
        Statistical test used for the foreground vs background contrast.
    result_filters : Mapping[str, Any] | None, default None
        Filters forwarded to ``_resolve_aa_result`` to select the AA configuration.
    alpha : float, default 0.10
        Raw p-value threshold for significance.
    alternative : {"two-sided", "greater", "less"}, default "two-sided"
        Alternative hypothesis for the selected test.
    require_max_in_bin0 : bool, default True
        Require the bin-wise median to be maximal in the closest bin.
    require_positive_effect : bool, default True
        Require ``median_bin0 > median_rest``.
    ignore_nans : bool, default False
        If True, drop NaNs per column before binning; otherwise raise.

    Returns
    -------
    pd.DataFrame
        One row per archetype and column with:
        ``arch_idx``, ``colname``, ``stat``, ``pval``, ``mean_diff``, ``median_diff``,
        ``mean_bin0``, ``mean_rest``, ``median_bin0``, ``median_rest``,
        ``max_in_bin0``, ``pos_effect``, ``signif``, ``enriched``.
    """
    # input validation
    _validate_aa_config(adata=adata)
    _validate_aa_results(adata=adata)

    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")

    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"Unsupported alternative: {alternative}")

    if isinstance(colnames, str):
        col_list = [colnames]
    else:
        col_list = list(colnames)
        if not col_list:
            raise ValueError("colnames must contain at least one column name.")

    for col in col_list:
        if col not in adata.obs.columns:
            raise KeyError(f"Column not found in adata.obs: {col}")
        if pd.api.types.is_bool_dtype(adata.obs[col]):
            raise TypeError(f"Column must be numeric (bool not allowed): {col}")
        if not pd.api.types.is_numeric_dtype(adata.obs[col]):
            raise TypeError(f"Column must be numeric: {col}")

    config, payload = _resolve_aa_result(adata, result_filters=result_filters)

    obsm_key = config.obsm_key
    n_dimensions = list(config.n_dimensions)
    X = adata.obsm[obsm_key][:, n_dimensions]
    Z = payload.get("Z")
    if Z is None:
        raise ValueError("Matched AA payload does not contain 'Z'.")

    euclidean_dist = cdist(X, Z)

    enrichment_rows: list[dict[str, Any]] = []

    for archetype_index in range(Z.shape[0]):
        d_vec_full = euclidean_dist[:, archetype_index]

        for col in col_list:
            obs_vec_full = adata.obs[col].to_numpy()

            if np.isnan(obs_vec_full).any():
                if not ignore_nans:
                    raise ValueError(f"NaNs found in adata.obs['{col}'].")
                valid_mask = ~np.isnan(obs_vec_full)
                d_vec = d_vec_full[valid_mask]
                obs_vec = obs_vec_full[valid_mask]
            else:
                d_vec = d_vec_full
                obs_vec = obs_vec_full

            n_cells = d_vec.shape[0]
            if n_cells < 2:
                raise ValueError(f"Too few valid cells for column {col} at archetype {archetype_index}.")

            # ranks 0..n_cells-1 (ties broken by sort order)
            r_vec = np.argsort(np.argsort(d_vec))

            # equal-size bins in {0,...,n_bins-1}
            bin_id = (r_vec * n_bins) // n_cells
            bin_id = np.minimum(bin_id, n_bins - 1)  # defensive

            mask0 = bin_id == 0
            mask_rest = ~mask0

            if mask0.sum() < 2 or mask_rest.sum() < 2:
                raise ValueError(
                    f"Too few cells in foreground/background for archetype {archetype_index} "
                    f"and column {col}: n0={mask0.sum()}, n1={mask_rest.sum()}."
                )

            x0 = obs_vec[mask0]
            x1 = obs_vec[mask_rest]

            # effect sizes
            mean0 = float(np.mean(x0))
            mean1 = float(np.mean(x1))
            median0 = float(np.median(x0))
            median1 = float(np.median(x1))

            mean_diff = mean0 - mean1
            median_diff = median0 - median1

            # ParTI-like "maximal in closest bin" criterion based on binned medians
            max_in_bin0 = True
            if require_max_in_bin0:
                binned_medians = np.empty(n_bins, dtype=float)
                for b in range(n_bins):
                    xb = obs_vec[bin_id == b]
                    binned_medians[b] = np.median(xb) if xb.size else -np.inf
                max_in_bin0 = int(np.nanargmax(binned_medians) == 0) == 1

            # optional positive-effect constraint
            pos_effect = True
            if require_positive_effect:
                pos_effect = bool(median_diff > 0)

            # significance test (closest bin vs rest)
            match test:
                case "ttest":
                    stat, pval = ttest_ind(
                        a=x0,
                        b=x1,
                        equal_var=False,
                        alternative=alternative,
                        nan_policy="omit",
                    )
                case "wilcox":
                    stat, pval = mannwhitneyu(
                        x=x0,
                        y=x1,
                        alternative=alternative,
                        method="auto",
                    )
                case _:
                    raise ValueError(f"{test} not supported")

            signif = pval <= alpha

            enriched = signif
            if require_max_in_bin0 and require_positive_effect:
                enriched = signif and max_in_bin0 and pos_effect
            elif require_max_in_bin0:
                enriched = signif and max_in_bin0
            elif require_positive_effect:
                enriched = signif and pos_effect

            enrichment_rows.append(
                {
                    "arch_idx": int(archetype_index),
                    "colname": col,
                    "stat": float(stat),
                    "pval": float(pval),
                    "mean_diff": float(mean_diff),
                    "median_diff": float(median_diff),
                    "mean_bin0": mean0,
                    "mean_rest": mean1,
                    "median_bin0": median0,
                    "median_rest": median1,
                    "max_in_bin0": bool(max_in_bin0),
                    "pos_effect": bool(pos_effect),
                    "signif": bool(signif),
                    "enriched": bool(enriched),
                }
            )

    return pd.DataFrame(enrichment_rows)


def compute_quantile_based_categorical_enrichment(
    adata: anndata.AnnData,
    colnames: str | list[str],
    n_bins: int = 10,
    test: Literal["hypergeometric"] = "hypergeometric",
    result_filters: Mapping[str, Any] | None = None,
    *,
    alpha: float = 0.10,
    contrast: Literal["rest", "furthest"] = "rest",
    require_max_in_bin0: bool = True,
    min_category_count: int = 100,
    ignore_nans: bool = False,
):
    """
    Quantile-based distance-bin enrichment for categorical obs columns.

    Cells are ranked by distance to each archetype, split into equal-size bins, and
    category over-representation near each archetype is tested using a hypergeometric
    test (one-sided). By default, the closest bin is contrasted against all remaining
    cells, with an option to use the furthest bin as background.

    The max-in-bin0 criterion is computed using an enrichment curve defined as the
    ratio of bin frequency to global frequency for each category.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with AA configuration and archetype results.
    colnames : str or list[str]
        One or more categorical columns in ``adata.obs`` to test.
    n_bins : int, default 10
        Number of equal-size distance bins.
    test : {"hypergeometric"}, default "hypergeometric"
        Statistical test used for the foreground vs background contrast.
    result_filters : Mapping[str, Any] | None, default None
        Filters forwarded to ``_resolve_aa_result`` to select the AA configuration.
    alpha : float, default 0.10
        Raw p-value threshold for significance.
    contrast : {"rest", "furthest"}, default "rest"
        Background definition: all remaining cells or only the furthest bin.
    require_max_in_bin0 : bool, default True
        Require the enrichment curve to be maximal in the closest bin.
    min_category_count : int, default 100
        Minimum global count required for a category to be tested.
    ignore_nans : bool, default False
        If True, drop NaNs per column before binning; otherwise raise.

    Returns
    -------
    pd.DataFrame
        One row per archetype, column, and category with:
        ``arch_idx``, ``colname``, ``category``, ``stat``, ``pval``, ``signif``,
        ``enriched``, ``max_in_bin0``, ``n_bin0``, ``n_rest``, ``k_bin0``, ``k_rest``.
    """
    # input validation
    _validate_aa_config(adata=adata)
    _validate_aa_results(adata=adata)

    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")

    if test != "hypergeometric":
        raise ValueError(f"{test} not supported")

    if contrast not in ("rest", "furthest"):
        raise ValueError(f"Unsupported contrast: {contrast}")

    if min_category_count < 1:
        raise ValueError("min_category_count must be >= 1.")

    if isinstance(colnames, str):
        col_list = [colnames]
    else:
        col_list = list(colnames)
        if not col_list:
            raise ValueError("colnames must contain at least one column name.")

    for col in col_list:
        if col not in adata.obs.columns:
            raise KeyError(f"Column not found in adata.obs: {col}")
        series = adata.obs[col]
        if pd.api.types.is_bool_dtype(series):
            continue
        if (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue
        raise TypeError(f"Column must be categorical, string/object, or bool: {col}")

    config, payload = _resolve_aa_result(adata, result_filters=result_filters)

    obsm_key = config.obsm_key
    n_dimensions = list(config.n_dimensions)
    X = adata.obsm[obsm_key][:, n_dimensions]
    Z = payload.get("Z")
    if Z is None:
        raise ValueError("Matched AA payload does not contain 'Z'.")

    euclidean_dist = cdist(X, Z)

    enrichment_rows: list[dict[str, Any]] = []

    for archetype_index in range(Z.shape[0]):
        d_vec_full = euclidean_dist[:, archetype_index]

        for col in col_list:
            obs_series = adata.obs[col]
            obs_vec_full = obs_series.to_numpy()

            na_mask = pd.isna(obs_vec_full)
            if na_mask.any():
                if not ignore_nans:
                    raise ValueError(f"NaNs found in adata.obs['{col}'].")
                valid_mask = ~na_mask
                d_vec = d_vec_full[valid_mask]
                obs_vec = obs_vec_full[valid_mask]
            else:
                d_vec = d_vec_full
                obs_vec = obs_vec_full

            n_cells = d_vec.shape[0]
            if n_cells < 2:
                raise ValueError(f"Too few valid cells for column {col} at archetype {archetype_index}.")

            # ranks 0..n_cells-1 (ties broken by sort order)
            r_vec = np.argsort(np.argsort(d_vec))

            # equal-size bins in {0,...,n_bins-1}
            bin_id = (r_vec * n_bins) // n_cells
            bin_id = np.minimum(bin_id, n_bins - 1)  # defensive

            mask0 = bin_id == 0
            if contrast == "rest":
                mask_bg = ~mask0
            else:
                mask_bg = bin_id == (n_bins - 1)

            if mask0.sum() < 2 or mask_bg.sum() < 2:
                raise ValueError(
                    f"Too few cells in foreground/background for archetype {archetype_index} "
                    f"and column {col}: n0={mask0.sum()}, n1={mask_bg.sum()}."
                )

            if isinstance(obs_series.dtype, pd.CategoricalDtype):
                categories = obs_series.cat.categories
            else:
                categories = pd.unique(obs_vec)

            global_counts = pd.Series(obs_vec).value_counts(dropna=False)
            eligible_categories = [cat for cat in categories if global_counts.get(cat, 0) >= min_category_count]
            if not eligible_categories:
                raise ValueError(f"No categories in {col} meet min_category_count={min_category_count}.")

            for cat in eligible_categories:
                k_total = int(global_counts.get(cat, 0))
                global_freq = k_total / n_cells

                # enrichment curve based on bin-wise frequency / global frequency
                max_in_bin0 = True
                if require_max_in_bin0:
                    binned_enrichment = np.empty(n_bins, dtype=float)
                    for b in range(n_bins):
                        bmask = bin_id == b
                        n_b = int(bmask.sum())
                        if n_b == 0:
                            binned_enrichment[b] = -np.inf
                            continue
                        k_b = int((obs_vec[bmask] == cat).sum())
                        freq_b = k_b / n_b
                        binned_enrichment[b] = freq_b / global_freq if global_freq > 0 else -np.inf
                    max_in_bin0 = int(np.nanargmax(binned_enrichment) == 0) == 1

                k_bin0 = int((obs_vec[mask0] == cat).sum())
                n_bin0 = int(mask0.sum())

                if contrast == "rest":
                    k_bg = int((obs_vec[mask_bg] == cat).sum())
                    n_bg = int(mask_bg.sum())
                    M = n_cells
                    K = k_total
                else:
                    k_bg = int((obs_vec[mask_bg] == cat).sum())
                    n_bg = int(mask_bg.sum())
                    M = n_bin0 + n_bg
                    K = k_bin0 + k_bg

                # one-sided over-representation
                pval = float(hypergeom.sf(k_bin0 - 1, M, K, n_bin0))
                signif = pval <= alpha

                enriched = signif
                if require_max_in_bin0:
                    enriched = signif and max_in_bin0

                enrichment_rows.append(
                    {
                        "arch_idx": int(archetype_index),
                        "colname": col,
                        "category": cat,
                        "stat": float(k_bin0),
                        "pval": pval,
                        "signif": bool(signif),
                        "enriched": bool(enriched),
                        "max_in_bin0": bool(max_in_bin0),
                        "n_bin0": n_bin0,
                        "n_rest": n_bg,
                        "k_bin0": k_bin0,
                        "k_rest": k_bg,
                    }
                )

    return pd.DataFrame(enrichment_rows)
