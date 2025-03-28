"""Functions to calculate which features (e.g. genes or covariates) are enriched at each archetype."""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cdist


def compute_archetype_weights(
    X: np.ndarray | sc.AnnData,
    Z: np.ndarray | None = None,
    mode: str = "automatic",
    length_scale: None | float = None,
) -> None | tuple[np.ndarray | None]:
    """
    Calculate weights for cells based on their distance to archetypes using a squared exponential kernel.

    Parameters
    ----------
    X : Union[np.ndarray, sc.AnnData]
        The input data, which can be either:
        - A 2D array of shape (n_samples, n_features) representing the PCA coordinates of the cells.
        - An AnnData object containing the PCA coordinates in `.obsm["X_pca"]` and archetypes in `.uns["archetypal_analysis"]["Z"]`.
    Z : np.ndarray, optional
        A 2D array of shape (n_archetypes, n_features) representing the PCA coordinates of the archetypes.
        Required if `X` is not an AnnData object.
    mode : str, optional (default="automatic")
        The mode for determining the length scale of the kernel:
        - "automatic": The length scale is calculated as half the median distance from the data centroid to the archetypes.
        - "manual": The length scale is provided by the user via the `length_scale` parameter.
    length_scale : float, optional
        The length scale of the kernel. Required if `mode="manual"`.

    Returns
    -------
    np.ndarray
        - If `X` is an AnnData object, the weights are added to `X.obsm["cell_weights"]` and nothing is returned.
        - If `X` is a numpy array, a 2D array of shape (n_samples, n_archetypes) representing the weights for each cell-archetype pair.
    """
    # Handle and validate input data
    adata = None
    if isinstance(X, sc.AnnData):
        adata = X
        if "archetypal_analysis" not in X.uns:
            raise ValueError("Result from Archetypal Analysis not found in adata.uns. Please run AA()")
        Z = X.uns["archetypal_analysis"]["Z"]
        X = X.obsm["X_pca"][:, : X.uns["n_pcs"]]

    if Z is None:
        raise ValueError("Please add the archetypes coordinates as input Z")

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

    if isinstance(adata, sc.AnnData):
        adata.obsm["cell_weights"] = weights
        return None
    else:
        return weights


# compute_characteristic_gene_expression_per_archetype
def compute_archetype_expression(adata: sc.AnnData, layer: str | None = None) -> np.ndarray:
    """
    Calculate a weighted pseudobulk expression profile for each archetype.

    This function computes the weighted average of gene expression across cells for each archetype.
    The weights should be based on the distance of cells to the archetypes, as computed by `calculate_weights`.

    Parameters
    ----------
    adata : sc.AnnData
        An AnnData object containing the gene expression data and weights. The weights should be stored in
        `adata.obsm["cell_weights"]` as a 2D array of shape (n_samples, n_archetypes).
    layer : str, optional (default=None)
        The layer of the AnnData object to use for gene expression. If `None`, `adata.X` is used. For Pareto analysis of AA data,
        z-scaled data is recommended.

    Returns
    -------
    np.ndarray
        A 2D array of shape (n_archetypes, n_genes) representing the weighted pseudobulk expression profiles.
    """
    weights = adata.obsm["cell_weights"].T
    if layer is None:
        expr = adata.X
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
    Extract the top enriched biological processes based on statistical significance.

    This function filters and ranks the most enriched biological processes from the decoupler output
    based on estimated enrichment scores (`est`) and corresponding p-values (`pval`) below the
    specified threshold (`p_treshold`).

    Parameters
    ----------
    est : pd.DataFrame
        A DataFrame of shape (n_archetypes, n_processes) containing the estimated enrichment scores
        for each process and archetype.
    pval : pd.DataFrame
        A DataFrame of shape (n_archetypes, n_processes) containing the p-values corresponding to
        the enrichment scores in `est`.
    order : str, optional (default="desc")
        The sorting order for selecting the top processes:
        - "desc": Selects the top `n` processes with the highest enrichment scores.
        - "asc": Selects the top `n` processes with the lowest enrichment scores.
    n : int, optional (default=20)
        The number of top processes to extract per archetype.
    p_threshold : float, optional (default=0.05)
        The p-value threshold for filtering processes. Only processes with p-values below this
        threshold are considered.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary where keys are of the form "archetype_X" and values are
        DataFrames containing the top `n` enriched processes for each archetype. Each DataFrame
        has two columns:
        - "Process": The name of the biological process.
        - "Score": The enrichment score for the process.
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
    Extract the top enriched biological processes that are specific to each archetype.

    This function identifies the most enriched biological processes for each archetype based on
    estimated enrichment scores (`est`) and corresponding p-values (`pval`) from the decoupler output below the
    specified threshold (`p_treshold`). It ensures that the selected processes are specific to the archetype by
    enforcing that their enrichment scores are below a specified threshold (`drop_threshold`) in all other archetypes.

    Parameters
    ----------
    est : pd.DataFrame
        A DataFrame of shape (n_archetypes, n_processes) containing the estimated enrichment scores
        for each process and archetype.
    pval : pd.DataFrame
        A DataFrame of shape (n_archetypes, n_processes) containing the p-values corresponding to
        the enrichment scores in `est`.
    n : int, optional (default=20)
        The number of top processes to extract per archetype.
    p_threshold : float, optional (default=0.05)
        The p-value threshold for filtering processes. Only processes with p-values below this
        threshold are considered.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary where keys are of the form "archetype_X" and values are
        DataFrames containing the top `n` enriched processes for each archetype that are below a score of
        `drop_threshold` for all other archetypes.
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


def compute_meta_enrichment(adata: sc.AnnData, meta_col: str) -> pd.DataFrame:
    """
    Compute the weighted enrichment of metadata categories across archetypes.

    This function performs the following steps:
    1. One-hot encodes the categorical metadata.
    2. Normalizes the one-hot encoded metadata to sum to 1 for each category.
    3. Computes the weighted enrichment of each metadata category for each archetype using the weights stored in `adata.obsm["cell_weights"]`.

    Parameters
    ----------
    adata : sc.AnnData
        An AnnData object containing the metadata in `adata.obs[meta]` and weights in `adata.obsm["cell_weights"]`.
    meta_col : str
        The name of the categorical metadata column in `adata.obs` to use for enrichment analysis.

    Returns
    -------
    pd.DataFrame
        A DataFrame of shape (n_archetypes, n_categories) containing the normalized enrichment of a metadata category for a given archetypes.
    """
    metadata = adata.obs[meta_col]
    weights = adata.obsm["cell_weights"].T

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

    return weighted_meta_df
