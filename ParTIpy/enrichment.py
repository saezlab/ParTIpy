"""
Functions to calculate the genes enriched at an archetype
"""
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scanpy as sc
import pandas as pd


def euclidean_distance(
        X: np.ndarray, 
        Z: np.ndarray
    ) -> np.ndarray:
    """
    Calculate the Euclidean distance between each cell and each archetype.

    This function computes the pairwise Euclidean distances between the cells (rows of `X`)
    and the archetypes (columns of `Z`). The result is a distance matrix where each entry
    represents the distance between a cell and the archetype.

    Parameters:
    -----------
    X : np.ndarray
        A 2D array of shape (n_samples, n_features) representing the PCA coordinates of the cells.
    Z : np.ndarray
        A 2D array of shape (n_features, n_archetypes) representing the PCA coordinates of the archetypes.

    Returns:
    --------
    np.ndarray
        A 2D array of shape (n_archetypes, n_samples) representing the pairwise Euclidean distances.
    """
    SX = np.sum(X**2, axis=1, keepdims=True)
    SZ = np.sum(Z**2, axis=0, keepdims=True)

    squared_distances = SX + SZ - 2 * X.dot(Z)

    # Ensure no negative values due to numerical precision issues
    squared_distances = np.maximum(squared_distances, 0)

    distances = np.sqrt(squared_distances)

    return distances.T

def calculate_weights(
        X: Union[np.ndarray, sc.AnnData],
        Z: Optional[np.ndarray] = None,
        mode: str = "automatic",
        length_scale: Optional[float] = None,
    ) -> Tuple[np.ndarray, Optional[float]]:
    """
    Calculate weights for cells based on their distance to archetypes using a squared exponential kernel.

    Parameters:
    -----------
    X : Union[np.ndarray, sc.AnnData]
        The input data, which can be either:
        - A 2D array of shape (n_samples, n_features) representing the PCA coordinates of the cells.
        - An AnnData object containing the PCA coordinates in `.obsm["X_pca_reduced"]` and archetypes in `.uns["archetypal_analysis"]["Z"]`.
    Z : np.ndarray, optional
        A 2D array of shape (n_archetypes, n_features) representing the PCA coordinates of the archetypes.
        Required if `X` is not an AnnData object.
    mode : str, optional (default="automatic")
        The mode for determining the length scale of the kernel:
        - "automatic": The length scale is calculated as half the median distance from the data centroid to the archetypes.
        - "manual": The length scale is provided by the user via the `length_scale` parameter.
    length_scale : float, optional
        The length scale of the kernel. Required if `mode="manual"`.

    Returns:
    --------
    Tuple[np.ndarray, Optional[float]]
        - A 2D array of shape (n_archetypes, n_samples) representing the weights for each cell-archetype pair.
        - The calculated length scale (if `mode="automatic"`), otherwise `None`.
    """
    # Handle and validate input data
    if isinstance(X, sc.AnnData):
        if "archetypal_analysis" not in X.uns:
            raise ValueError("Result from Archetypal Analysis not found in adata.uns. Please run AA()")
        
        Z = X.uns["archetypal_analysis"]["Z"]
        X = X.obsm["X_pca_reduced"]
            
    if Z is None:
        raise ValueError("Please add the archetypes coordinates as input Z")

    # Calculate or validate length_scale based on mode
    if mode == "automatic":
        centroid = np.mean(X, axis=0).reshape(1, -1)
        length_scale = np.median(euclidean_distance(centroid, Z.T)) / 2

    elif mode == "manual":
        if length_scale is None:
            raise ValueError("For 'manual' mode, 'length_scale' must be provided.")
        
    else:
        raise ValueError("Mode must be either 'automatic' or 'manual'.")

    # Weight calculation
    euclidean_dist = euclidean_distance(X, Z.T)
    weights = np.exp(-(euclidean_dist**2) / (2 * length_scale**2))

    return (weights, length_scale) if mode == "automatic" else weights

def weighted_expr(
        weights: np.ndarray, 
        expr: np.ndarray
    ) -> np.ndarray:
    """
    Calculate a weighted pseudobulk expression profile for each archetype.

    This function computes the weighted average of gene expression across cells for each archetype.
    The weights should be based on the distance of cells to the archetypes, as computed by `calculate_weights`.

    Parameters:
    -----------
    weights : np.ndarray
        A 2D array of shape (n_archetypes, n_samples) representing the weights for each cell-archetype pair.
    expr : np.ndarray
        A 2D array of shape (n_samples, n_genes) representing the gene expression matrix. Z-scaled data is recommended for 
        gene expression analysis between the archetypes.

    Returns:
    --------
    np.ndarray
        A 2D array of shape (n_archetypes, n_genes) representing the weighted pseudobulk expression profiles.
    """
    pseudobulk = np.einsum('ij,jk->ik', weights, expr)
    pseudobulk /= weights.sum(axis=1, keepdims=True)
    
    return pseudobulk

def extract_top_processes(
        est: pd.DataFrame,
        pval: pd.DataFrame,
        order: str = "desc",
        n: int = 20,
        p_threshold: float = 0.05,
    ) -> Dict[str, pd.DataFrame]:
    """
    Extract the top enriched biological processes based on statistical significance.

    This function filters and ranks the most enriched biological processes from the decoupler output
    based on estimated enrichment scores (`est`) and corresponding p-values (`pval`) below the 
    specified threshold (`p_treshold`).

    Parameters:
    -----------
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

    Returns:
    --------
    Dict[str, pd.DataFrame]
        A dictionary where keys are of the form "archetype_X" and values are
        DataFrames containing the top `n` enriched processes for each archetype. Each DataFrame
        has two columns:
        - "Process": The name of the biological process.
        - "Score": The enrichment score for the process.
    """
    # Validate input 
    if est.shape != pval.shape:
        raise ValueError("`est` and `pval` must have the same shape.")

    if order not in ["desc", "asc"]:
        raise ValueError("`order` must be either 'desc' or 'asc'.")

    results = {}  
    for archetype in range(est.shape[0]):
        # Filter processes based on p-value threshold
        significant_processes  = pval.iloc[archetype] < p_threshold
        filtered_scores  = est.iloc[archetype, list(significant_processes )]

        # Sort and select top processes
        if order == "desc":
            top_processes = filtered_scores.nlargest(n).reset_index()
        else: 
            top_processes = filtered_scores.nsmallest(n).reset_index()

        top_processes.columns = ["Process", "Score"]
        results[f"archetype_{archetype}"] = top_processes  

    return results

def meta_enrichment(
        meta: pd.Series, 
        weights: np.ndarray
    ) -> pd.DataFrame:
    """
    Compute the weighted enrichment of metadata categories across archetypes.

    This function performs the following steps:
    1. One-hot encodes the categorical metadata.
    2. Normalizes the one-hot encoded metadata to sum to 1 for each category.
    3. Computes the weighted enrichment of each metadata category for each archetype using the provided weights.

    Parameters
    ----------
    meta : pd.Series
        A Pandas Series of shape (n_samples,) containing categorical metadata values.
        
    weights : np.ndarray
        A 2D array of shape (n_archetypes, n_samples) representing the weights for each cell-archetype pair.

    Returns:
    --------
    pd.DataFrame
        A DataFrame of shape (n_archetypes, n_categories) containing the weighted enrichment values.
    """
    # Validation input
    if meta.shape[0] != weights.shape[1]:
        raise ValueError("Number of rows in `weights` must match the length of `meta`.")
    
    # One-hot encoding of metadata
    df_encoded = pd.get_dummies(meta).astype(float)
    # Normalization
    df_encoded = df_encoded / df_encoded.values.sum(axis=0, keepdims=True)

    # Compute weighted enrichment
    weighted_meta = weighted_expr(weights, df_encoded)
    # Normalization
    weighted_meta = weighted_meta / np.sum(weighted_meta, axis=1, keepdims=True)
    weighted_meta_df = pd.DataFrame(weighted_meta, columns=df_encoded.columns)
    
    return weighted_meta_df