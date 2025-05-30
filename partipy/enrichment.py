"""Functions to calculate which features (e.g. genes or covariates) are enriched at each archetype."""

import anndata
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .paretoti import _validate_aa_config, _validate_aa_results


def compute_archetype_weights(
    adata: anndata.AnnData,
    mode: str = "automatic",
    length_scale: None | float = None,
    save_to_anndata: bool = True,
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
        If `True`, the weights are saved to `adata.obsm["cell_weights"]`. If `False`, the function returns the weights as a numpy array.

    Returns
    -------
    np.ndarray
        - If `X` is an AnnData object, the weights are added to `X.obsm["cell_weights"]` and nothing is returned.
        - If `X` is a numpy array, a 2D array of shape (n_samples, n_archetypes) representing the weights for each cell-archetype pair.
    """
    # input validation
    _validate_aa_config(adata=adata)
    _validate_aa_results(adata=adata)

    obsm_key = adata.uns["AA_config"]["obsm_key"]
    n_dimensions = adata.uns["AA_config"]["n_dimensions"]
    X = adata.obsm[obsm_key][:, n_dimensions]
    Z = adata.uns["AA_results"]["Z"]

    # Calculate or validate length_scale based on mode
    if mode == "automatic":
        centroid = np.mean(X, axis=0).reshape(1, -1)
        length_scale = np.median(cdist(centroid, Z)) / 2
    elif mode == "manual":
        if length_scale is None:
            raise ValueError("For 'manual' mode, 'length_scale' must be provided.")
    else:
        raise ValueError("Mode must be either 'automatic' or 'manual'.")
    print(f"Applied length scale is {length_scale}.")

    # Weight calculation
    euclidean_dist = cdist(X, Z)
    weights = np.exp(-(euclidean_dist**2) / (2 * length_scale**2))  # type: ignore[operator]
    weights = weights.astype(np.float32)

    if save_to_anndata:
        adata.obsm["cell_weights"] = weights
        return None
    else:
        return weights


# compute_characteristic_gene_expression_per_archetype
def compute_archetype_expression(adata: anndata.AnnData, layer: str | None = None) -> pd.DataFrame:
    """
    Calculate a weighted average gene expression profile for each archetype.

    This function computes the weighted average of gene expression across cells for each archetype.
    The weights should be based on the distance of cells to the archetypes, as computed by `calculate_weights`.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing the gene expression data and weights. The weights should be stored in
        `adata.obsm["cell_weights"]` as a 2D array of shape (n_samples, n_archetypes).
    layer : str, default `None`
        The layer of the AnnData object to use for gene expression. If `None`, `adata.X` is used. For Pareto analysis of AA data,
        z-scaled data is recommended.

    Returns
    -------
    pd.DataFrame
        A DataFrame of shape (n_archetypes, n_genes) with weighted pseudobulk expression profiles.
    """
    if "cell_weights" not in adata.obsm:
        raise ValueError("No weights available. Please run compute_archetype_weights()")
    weights = adata.obsm["cell_weights"].T

    if layer is None:
        expr = adata.X
    elif layer not in adata.layers:
        raise ValueError("Invalid layer")
    else:
        expr = adata.layers[layer]

    pseudobulk = np.einsum("ij,jk->ik", weights, expr)
    pseudobulk /= weights.sum(axis=1, keepdims=True)

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


def compute_meta_enrichment(adata: anndata.AnnData, meta_col: str, datatype: str = "automatic") -> pd.DataFrame:
    """
    Compute the enrichment of metadata categories across archetypes.

    This function estimates how enriched each metadata category is within each archetype using
    a weighted average approach. Weights are based on each cell’s contribution to each archetype
    (`adata.obsm["cell_weights"]`).It supports both categorical and continuous metadata.

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
        in `adata.obsm["cell_weights"]`
    meta_col : str
        The name of the categorical metadata column in `adata.obs` to use for enrichment analysis.
    datatype : str, default `automatic`
        Specifies how to interpret the metadata column:
        - "automatic": infers type based on column dtype.
        - "categorical": treats the column as categorical and one-hot encodes it.
        - "continuous": treats the column as numeric and computes weighted averages.

    Returns
    -------
    pd.DataFrame
        A DataFrame of shape (n_archetypes, n_categories) for categorical data or
        (n_archetypes, 1) for continuous data, containing normalized enrichment scores
        or weighted averages respectively.
    """
    if meta_col not in adata.obs:
        raise ValueError("Metadata column does not exist")
    if "cell_weights" not in adata.obsm:
        raise ValueError("No weights available. Please run compute_archetype_weights()")

    metadata = adata.obs[meta_col]
    weights = adata.obsm["cell_weights"].T

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
        weighted_meta /= weights.sum(axis=1, keepdims=True)

        # Normalization
        weighted_meta = weighted_meta / np.sum(weighted_meta, axis=1, keepdims=True)
        weighted_meta_df = pd.DataFrame(weighted_meta, columns=df_encoded.columns)

    elif mode == "continuous":
        metadata = np.asarray(metadata, dtype=float).reshape(-1, 1)

        # Compute weighted enrichment
        weighted_meta = np.einsum("ij,jk->ik", weights, metadata)
        weighted_meta /= weights.sum(axis=1, keepdims=True)

        weighted_meta_df = pd.DataFrame(weighted_meta, columns=[meta_col])

    return weighted_meta_df
