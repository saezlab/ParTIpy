"""Functions to calculate which features (e.g. genes or covariates) are enriched at each archetype."""

from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as pn
import scanpy as sc
from scipy.spatial.distance import cdist


def calculate_weights(
    X: np.ndarray | sc.AnnData,
    Z: np.ndarray | None = None,
    mode: str = "automatic",
    length_scale: None | float = None,
) -> None | tuple[np.ndarray, float | None]:
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


def weighted_expr(adata: sc.AnnData, layer: str | None = None) -> np.ndarray:
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

    return pseudobulk_df


def extract_top_processes(
    est: pd.DataFrame,
    pval: pd.DataFrame,
    order: str = "desc",
    n: int = 20,
    p_threshold: float = 0.05,
) -> dict[str, pd.DataFrame]:
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
    if est.shape != pval.shape:
        raise ValueError("`est` and `pval` must have the same shape.")

    if order not in ["desc", "asc"]:
        raise ValueError("`order` must be either 'desc' or 'asc'.")

    results = {}
    for archetype in range(est.shape[0]):
        # Filter processes based on p-value threshold
        significant_processes = pval.iloc[archetype] < p_threshold
        filtered_scores = est.iloc[archetype, list(significant_processes)]

        # Sort and select top processes
        if order == "desc":
            top_processes = filtered_scores.nlargest(n).reset_index()
        else:
            top_processes = filtered_scores.nsmallest(n).reset_index()

        top_processes.columns = ["Process", "Score"]
        results[f"archetype_{archetype}"] = top_processes

    return results


def extract_top_specific_processes(
    est: pd.DataFrame,
    pval: pd.DataFrame,
    drop_threshold: int = 0,
    n: int = 20,
    p_threshold: float = 0.05,
):
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
    drop_threshold : int, optional (default=20)
      The enrichment threshold below which processes are dropped.
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
    if est.shape != pval.shape:
        raise ValueError("`est` and `pval` must have the same shape.")

    results = {}
    for archetype in range(est.shape[0]):
        # Filter processes based on p-value threshold
        significant_processes = pval.iloc[archetype] < p_threshold
        top_processes = est.iloc[archetype, list(significant_processes)].nlargest(n).index

        # Filter processes based on drop threshold
        subset = est.loc[:, top_processes]
        subset.index = subset.index.astype(int)
        filtered_processes = top_processes[(subset.drop(index=archetype) < drop_threshold).all(axis=0)]

        results[f"archetype_{archetype}"] = est.loc[:, filtered_processes].copy()

    return results


def meta_enrichment(adata: sc.AnnData, meta: str) -> pd.DataFrame:
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
    meta : str
        The name of the categorical metadata column in `adata.obs` to use for enrichment analysis.

    Returns
    -------
    pd.DataFrame
        A DataFrame of shape (n_archetypes, n_categories) containing the normalized enrichment of a metadata category for a given archetypes.
    """
    metadata = adata.obs[meta]
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


def barplot_meta_enrichment(meta_enrich: pd.DataFrame, meta: str = "Meta"):
    """
    Parameters
    ----------
    meta_enrich: pd.DataFrame
        Output of meta_enrichment(), a pd.DataFrame containing the enrichment of meta categories (columns) for all archetypes (rows).
    meta: str, optional
        The name for the metadata.

    Returns
    -------
    pn.ggplot.ggplot
        A stacked bar plot.
    """
    # Prepare data
    meta_enrich = meta_enrich.reset_index().rename(columns={"index": "archetype"})
    meta_enrich_long = meta_enrich.melt(id_vars=["archetype"], var_name="Meta", value_name="Normalized_Enrichment")

    # Create plot
    plot = (
        pn.ggplot(
            meta_enrich_long,
            pn.aes(x="factor(archetype)", y="Normalized_Enrichment", fill="Meta"),
        )
        + pn.geom_bar(stat="identity", position="stack")
        + pn.theme_matplotlib()
        + pn.scale_fill_brewer(type="qual", palette="Dark2")
        + pn.labs(
            title="Meta Enrichment Across Archetypes",
            x="Archetype",
            y="Normalized Enrichment",
            fill=meta,
        )
    )
    return plot


def heatmap_meta_enrichment(meta_enrich: pd.DataFrame, meta: str | None = "Meta"):
    """
    Parameters
    ----------
    meta_enrich: pd.DataFrame
        Output of meta_enrichment(), a pd.DataFrame containing the enrichment of meta categories (columns) for all archetypes (rows).
    meta: str, optional
        The name for the metadata.

    Returns
    -------
    pn.ggplot.ggplot
        A heatmap.
    """
    # Prepare data
    meta_enrich = meta_enrich.reset_index().rename(columns={"index": "archetype"})
    meta_enrich_long = meta_enrich.melt(id_vars=["archetype"], var_name="Meta", value_name="Normalized_Enrichment")

    # Create plot
    plot = (
        pn.ggplot(meta_enrich_long, pn.aes("archetype", "Meta", fill="Normalized_Enrichment"))
        + pn.geom_tile()
        + pn.scale_fill_continuous(cmap_name="Blues")
        + pn.theme_matplotlib()
        + pn.labs(title="Heatmap", x="Archetype", y=meta, fill=" Normalized \nEnrichment")
    )
    return plot


def radarplot_meta_enrichment(meta_enrich: pd.DataFrame):
    """
    Parameters
    ----------
    meta_enrich: pd.DataFrame
        Output of meta_enrichment(), a pd.DataFrame containing the enrichment of meta categories (columns) for all archetypes (rows).

    Returns
    -------
    plt.pyplot.Figure
        Radar plots for all archetypes.
    """
    # Prepare data
    meta_enrich = meta_enrich.T.reset_index().rename(columns={"index": "Meta_feature"})

    # Function to create a radar plot for a given row
    def make_radar(row, title, color):
        # Set number of meta categories
        categories = list(meta_enrich)[1:]
        N = len(categories)

        # Calculate angles for the radar plot
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        # Initialise the radar plot
        ax = plt.subplot(int(np.ceil(len(meta_enrich) / 2)), 2, row + 1, polar=True)

        # Put first axis on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        # One axe per variable and add labels
        archetype_label = [f"A{i}" for i in range(len(list(meta_enrich)[1:]))]
        plt.xticks(angles[:-1], archetype_label, color="grey", size=8)

        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks(
            [0, 0.25, 0.5, 0.75, 1],
            ["0", "0.25", "0.50", "0.75", "1.0"],
            color="grey",
            size=7,
        )
        plt.ylim(0, 1)

        # Draw plot
        values = meta_enrich.loc[row].drop("Meta_feature").values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, linestyle="solid")
        ax.fill(angles, values, color=color, alpha=0.4)

        # Add a title
        plt.title(title, size=11, color=color, y=1.065)

    # Initialize the figure
    my_dpi = 96
    plt.figure(figsize=(1000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)

    # Create a color palette:
    my_palette = plt.colormaps.get_cmap("Dark2")

    # Loop to plot
    for row in range(0, len(meta_enrich.index)):
        make_radar(
            row=row,
            title=f"Feature: {meta_enrich['Meta_feature'][row]}",
            color=my_palette(row),
        )

    return plt


def plot_functional_enrichment(top_features, show: bool = True):
    """
    Generate bar plots for functional enrichment data across archetypes.

    Parameters
    ----------
    top_features : dict
        A dictionary where keys are archetype names ('archetype_0', 'archetype_1',...) and values are pd.DataFrames
        containing the data to plot. Each DataFrame should have a column for the feature ('Process') and a column
        for the score ('Score')."
    show: bool, optional
        If the plots should be printed.

    Returns
    -------
    list
        A list of `plotnine.ggplot` objects, one for each archetype.
    """
    plots = []
    # Loop through archetypes
    for i in range(len(top_features)):
        key = f"archetype_{i}"  # Construct the key dynamically
        data = top_features[key]

        # Order column
        data["Process"] = pd.Categorical(data["Process"], categories=data["Process"].tolist(), ordered=True)

        # Create plot
        plot = (
            pn.ggplot(data, pn.aes(x="Process", y="Score", fill="Score"))
            + pn.geom_bar(stat="identity")
            + pn.labs(
                title=f"Enrichment at archetype {i}",
                x="Feature",
                y="Enrichment score",
                fill="ProEnrichment scorecess",
            )
            + pn.theme_matplotlib()
            + pn.theme(figure_size=(15, 5))
            + pn.coord_flip()
            + pn.scale_fill_gradient2(
                low="blue",
                mid="lightgrey",
                high="red",
                midpoint=0,
            )
        )
        if show:
            plot.show()
        plots.append(plot)

    # Return the list of plots
    return plots


def plot_enrichment_comparison(est: pd.DataFrame, features: str | list[str] | pd.Series):
    """
    Plots a grouped bar plot comparing enrichment scores across archetypes for a given set of features.

    Parameters
    ----------
    est : pandas.DataFrame
        A DataFrame containing enrichment scores. Rows represent archetypes, and columns represent features.
    features : str, list of str, or pd.Series
        A list of feature names (columns in `est`) to include in the plot.

    Returns
    -------
    plot : plotnine.ggplot.ggplot
        A grouped bar plot visualizing the enrichment scores for the specified features across archetypes."
    """
    # Subset the DataFrame to include only the specified features
    enrich_subset = est[features].reset_index().rename(columns={"index": "archetype"})

    # Convert the DataFrame from wide to long format for plotting
    enrich_long = enrich_subset.melt(id_vars=["archetype"], var_name="Feature", value_name="Enrichment")

    # Create plot
    plot = (
        pn.ggplot(enrich_long, pn.aes(x="Feature", y="Enrichment", fill="factor(archetype)"))
        + pn.geom_bar(stat="identity", position=pn.position_dodge())
        + pn.theme_matplotlib()
        + pn.scale_fill_brewer(type="qual", palette="Dark2")
        + pn.labs(
            x="Features",
            y="Enrichment score",
            fill="Archetype",
            title="Enrichment Comparison",
        )
        + pn.theme(figure_size=(10, 5))
        + pn.coord_flip()
    )
    return plot
