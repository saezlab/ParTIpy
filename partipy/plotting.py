import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotnine as pn
import scanpy as sc
from scipy.spatial import ConvexHull

from .paretoti import var_explained_aa


def plot_var_explained(
    adata: sc.AnnData,
) -> pn.ggplot:
    """
    Generate an elbow plot of the variance explained by Archetypal Analysis (AA) for a range of archetypes.

    This function creates a plot showing the variance explained by AA models with different numbers of archetypes.
    The data is retrieved from `adata.uns["AA_var"]`. If `AA_var` is not found, `var_explained_aa` is called.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing the variance explained data in `adata.uns["AA_var"]`.

    Returns
    -------
    pn.ggplot
        A ggplot object showing the variance explained plot.
    """
    # Validation input
    if "AA_var" not in adata.uns:
        print("AA_var not found in adata.uns. Computing variance explained by archetypal analysis...")
        var_explained_aa(adata=adata)

    plot_df = adata.uns["AA_var"]

    # Create data for the diagonal line
    diag_data = pd.DataFrame(
        {
            "k": [plot_df["k"].min(), plot_df["k"].max()],
            "varexpl": [plot_df["varexpl"].min(), plot_df["varexpl"].max()],
        }
    )

    p = (
        pn.ggplot(plot_df)
        + pn.geom_line(mapping=pn.aes(x="k", y="varexpl"), color="black")
        + pn.geom_point(mapping=pn.aes(x="k", y="varexpl"), color="black")
        + pn.geom_line(data=diag_data, mapping=pn.aes(x="k", y="varexpl"), color="gray")
        + pn.labs(x="Number of Archetypes (k)", y="Variance Explained")
        + pn.lims(y=[0, 1])
        + pn.scale_x_continuous(breaks=list(np.arange(plot_df["k"].min(), plot_df["k"].max() + 1)))
        + pn.theme_matplotlib()
    )
    return p


def plot_bootstrap_2D(adata: sc.AnnData) -> pn.ggplot:
    """
    Create an static 2D scatter plot showing the positions of archetypes
    computed from bootstrap samples, stored in `adata.uns["AA_bootstrap"]`.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data object containing the archetype bootstrap data in `adata.uns["AA_bootstrap"]`.

    Returns
    -------
    pn.ggplot
        2D plot of bootstrap results for the archetypes.
    """
    # Validation input
    if "AA_bootstrap" not in adata.uns:
        raise ValueError("AA_bootstrap not found in adata.uns. Please run bootstrap_aa() to compute")

    # Generate the 2D scatter plot
    bootstrap_df = adata.uns["AA_bootstrap"]

    p = pn.ggplot(bootstrap_df) + pn.geom_point(pn.aes(x="pc_0", y="pc_1", color="archetype", shape="reference"))
    return p


def plot_bootstrap_3D(adata: sc.AnnData) -> go.Figure:
    """
    Create an interactive 3D scatter plot showing the positions of archetypes
    computed from bootstrap samples, stored in `adata.uns["AA_bootstrap"]`.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data object containing the archetype bootstrap data in `adata.uns["AA_bootstrap"]`.

    Returns
    -------
    go.Figure:
        3D plot of bootstrap results for the archetypes.
    """
    # Validation input
    if "AA_bootstrap" not in adata.uns:
        raise ValueError("AA_bootstrap not found in adata.uns. Please run bootstrap_aa() to compute")

    # Generate the 3D scatter plot
    bootstrap_df = adata.uns["AA_bootstrap"]
    fig = px.scatter_3d(
        bootstrap_df,
        x="pc_0",
        y="pc_1",
        z="pc_2",
        color="archetype",
        symbol="reference",
        labels={
            "pc_0": "PC 1",
            "pc_1": "PC 2",
            "pc_2": "PC 3",
        },
        title="Archetypes on bootstrapepd data",
        size_max=10,
        hover_data=["iter", "archetype", "reference"],
        opacity=0.5,
    )
    fig.update_layout(template="none")

    return fig


def plot_archetypes_2D(adata: sc.AnnData, color: str | None = None) -> pn.ggplot:
    """TODO"""
    if "archetypal_analysis" not in adata.uns:
        raise ValueError("Result from Archetypal Analysis not found in adata.uns. Please run AA()")
    Z = adata.uns["archetypal_analysis"]["Z"]
    X = adata.obsm["X_pca"][:, : adata.uns["n_pcs"]]
    color_vec = sc.get.obs_df(adata, color).values.flatten() if color else None
    plot = plot_2D(X=X, Z=Z, color_vec=color_vec)
    return plot


def plot_2D(
    X: np.ndarray,
    Z: np.ndarray,
    color_vec: np.ndarray | None = None,
) -> pn.ggplot:
    """
    2D plot of the datapoints in X and the 2D polytope enclosed by the archetypes in Z.

    Parameters
    ----------
    X : Union[np.ndarray, sc.AnnData]
        The input data, which can be either:
        - A 2D array of shape (n_samples, n_features) representing the data points.
        - An AnnData object containing the PCA data in `X.obsm["X_pca"]` and archetypes in `X.uns["archetypal_analysis"]["Z"]`.
    Z : np.ndarray, optional
        A 2D array of shape (n_archetypes, n_features) representing the archetype coordinates.
        Required if `X` is not an AnnData object.
    color_vec : np.ndarray, optional
        A 1D array of shape (n_samples,) containing values for coloring the data points in `X`.

    Returns
    -------
    pn.ggplot
        2D plot of X and polytope enclosed by Z
    """
    if X.shape[1] < 2 or Z.shape[1] < 2:
        raise ValueError("Both X and Z must have at least 2 columns (PCs).")

    X_plot, Z_plot = X[:, :2], Z[:, :2]

    # Order archetypes for plotting the polytope
    plot_df = pd.DataFrame(X_plot, columns=["x0", "x1"])
    order = np.argsort(np.arctan2(Z_plot[:, 1] - np.mean(Z_plot[:, 1]), Z_plot[:, 0] - np.mean(Z_plot[:, 0])))

    arch_df = pd.DataFrame(Z_plot, columns=["x0", "x1"])
    arch_df["archetype_label"] = np.arange(arch_df.shape[0])
    arch_df = arch_df.iloc[order].reset_index(drop=True)
    arch_df = pd.concat([arch_df, arch_df.iloc[:1]], ignore_index=True)

    # Generate plot
    plot = pn.ggplot()

    if color_vec is not None:
        if len(color_vec) != len(plot_df):
            raise ValueError("color_vec must have the same length as X.")
        plot_df["color_vec"] = np.array(color_vec)
        plot += pn.geom_point(data=plot_df, mapping=pn.aes(x="x0", y="x1", color="color_vec"), alpha=0.5)
    else:
        plot += pn.geom_point(data=plot_df, mapping=pn.aes(x="x0", y="x1"), color="black", alpha=0.5)

    plot += pn.geom_point(data=arch_df, mapping=pn.aes(x="x0", y="x1"), color="red", size=1)
    plot += pn.geom_path(data=arch_df, mapping=pn.aes(x="x0", y="x1"), color="red", size=1)
    plot += pn.geom_label(data=arch_df, mapping=pn.aes(x="x0", y="x1", label="archetype_label"), color="black", size=12)

    plot += pn.labs(x="PC 1", y="PC 2")
    plot += pn.theme_matplotlib()

    return plot


def plot_archetypes_3D(adata: sc.AnnData, color: str | None = None) -> pn.ggplot:
    """TODO"""
    if "archetypal_analysis" not in adata.uns:
        raise ValueError("Result from Archetypal Analysis not found in adata.uns. Please run AA()")
    Z = adata.uns["archetypal_analysis"]["Z"]
    X = adata.obsm["X_pca"][:, : adata.uns["n_pcs"]]
    color_vec = sc.get.obs_df(adata, color).values.flatten() if color else None
    plot = plot_3D(X=X, Z=Z, color_vec=color_vec)
    return plot


def plot_3D(
    X: np.ndarray | sc.AnnData,
    Z: np.ndarray | None = None,
    color_vec: np.ndarray | None = None,
    marker_size: int = 4,
    color_polyhedron: str = "green",
) -> go.Figure:
    """
    3D plot of the datapoints in X and the 3D polytope enclosed by the archetypes in Z.

    Parameters
    ----------
    X : Union[np.ndarray, sc.AnnData]
        The input data, which can be either:
        - A 2D array of shape (n_samples, n_features) representing the data points.
        - An AnnData object containing the PCA data in `X.obsm["X_pca"]` and archetypes in `X.uns["archetypal_analysis"]["Z"]`.
    Z : np.ndarray, optional
        A 2D array of shape (n_archetypes, n_features) representing the archetype coordinates.
        Required if `X` is not an AnnData object.
    color_vec : np.ndarray, optional
        A 1D array of shape (n_samples,) containing values for coloring the data points in `X`.
    marker_size : int, optional (default=4)
        The size of the markers for the data points in `X`.
    color_polyhedron : str, optional (default="green")
        The color of the polytope (convex hull) defined by the archetypes.

    Returns
    -------
    go.Figuret
        3D plot of X and polytope enclosed by Z
    """
    # Validation input
    if Z is None:
        raise ValueError("Please add the archetypes coordinates as input Z")

    if X.shape[1] < 3 or Z.shape[1] < 3:
        raise ValueError("Both X and Z must have at least 3 columns (PCs).")

    X_plot, Z_plot = X[:, :3], Z[:, :3]

    plot_df = pd.DataFrame(X_plot, columns=["x0", "x1", "x2"])
    plot_df["marker_size"] = np.repeat(marker_size, X_plot.shape[0])

    # Create the 3D scatter plot
    if color_vec is not None:
        if len(color_vec) != len(plot_df):
            raise ValueError("color_vec must have the same length as X.")
        plot_df["color_vec"] = np.array(color_vec)
        fig = px.scatter_3d(
            plot_df,
            x="x0",
            y="x1",
            z="x2",
            labels={"x0": "PC 1", "x1": "PC 2", "x2": "PC 3"},
            title="3D polytope",
            color="color_vec",
            size="marker_size",
            size_max=10,
            opacity=0.5,
        )
    else:
        fig = px.scatter_3d(
            plot_df,
            x="x0",
            y="x1",
            z="x2",
            labels={"x0": "PC 1", "x1": "PC 2", "x2": "PC 3"},
            title="3D polytope",
            size="marker_size",
            size_max=10,
            opacity=0.5,
        )

    # Compute the convex hull of the archetypes
    hull = ConvexHull(Z_plot)

    # Add archetypes to the plot
    archetype_labels = [f"Archetype {i}" for i in range(Z_plot.shape[0])]
    fig.add_trace(
        go.Scatter3d(
            x=Z_plot[:, 0],
            y=Z_plot[:, 1],
            z=Z_plot[:, 2],
            mode="markers",
            text=archetype_labels,
            marker=dict(size=4, color=color_polyhedron, symbol="circle"),  # noqa: C408
            hoverinfo="text",
            name="Archetypes",
        )
    )

    # Add the polytope (convex hull) to the plot
    fig.add_trace(
        go.Mesh3d(
            x=Z_plot[:, 0],
            y=Z_plot[:, 1],
            z=Z_plot[:, 2],
            i=hull.simplices[:, 0],
            j=hull.simplices[:, 1],
            k=hull.simplices[:, 2],
            color=color_polyhedron,
            opacity=0.1,
        )
    )

    # Add edges of the polytope to the plot
    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])
        fig.add_trace(
            go.Scatter3d(
                x=Z_plot[simplex, 0],
                y=Z_plot[simplex, 1],
                z=Z_plot[simplex, 2],
                mode="lines",
                line={"color": color_polyhedron, "width": 4},
                showlegend=False,
            )
        )

    fig.update_layout(template="none")
    return fig


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


def barplot_functional_enrichment(top_features, show: bool = True):
    """
    Generate bar plots for functional enrichment data across archetypes.

    Parameters
    ----------
    top_features : dict
        A dictionary where keys are archetype indices (0, 1,...) and values are pd.DataFrames
        containing the data to plot. Each DataFrame should have a column for the feature ('Process') and a column
        for the archetype (0, 1, ...)
    show: bool, optional
        If the plots should be printed.

    Returns
    -------
    list
        A list of `plotnine.ggplot` objects, one for each archetype.
    """
    plots = []
    # Loop through archetypes
    for key in range(len(top_features)):
        data = top_features[key]

        # Order column
        data["Process"] = pd.Categorical(data["Process"], categories=data["Process"].tolist(), ordered=True)

        # Create plot
        plot = (
            pn.ggplot(data, pn.aes(x="Process", y=str(key), fill=str(key)))
            + pn.geom_bar(stat="identity")
            + pn.labs(
                title=f"Enrichment at archetype {key}",
                x="Feature",
                y="Enrichment score",
                fill="Enrichment score",
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


def barplot_enrichment_comparison(specific_processes_arch: pd.DataFrame):
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
    process_order = specific_processes_arch.sort_values("specificity", ascending=False)["Process"].to_list()
    arch_columns = specific_processes_arch.drop(columns=["Process", "specificity"]).columns
    plot_df = specific_processes_arch.drop(columns="specificity").melt(
        id_vars=["Process"], value_vars=arch_columns, var_name="Archetype", value_name="Enrichment"
    )
    plot_df["Process"] = pd.Categorical(plot_df["Process"], categories=process_order)

    plot = (
        pn.ggplot(plot_df, pn.aes(x="Process", y="Enrichment", fill="factor(Archetype)"))
        + pn.geom_bar(stat="identity", position=pn.position_dodge())
        + pn.theme_matplotlib()
        + pn.scale_fill_brewer(type="qual", palette="Dark2")
        + pn.labs(
            x="Process",
            y="Enrichment score",
            fill="Archetype",
            title="Enrichment Comparison",
        )
        + pn.theme(figure_size=(10, 5))
        + pn.coord_flip()
    )
    return plot
