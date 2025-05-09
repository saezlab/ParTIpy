import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotnine as pn
import scanpy as sc
from mizani.palettes import hue_pal
from scipy.spatial import ConvexHull

from .paretoti import _validate_aa_config, _validate_aa_results, var_explained_aa


def plot_var_explained(adata: sc.AnnData) -> pn.ggplot:
    """
    Generate an elbow plot of the variance explained by Archetypal Analysis (AA) for a range of archetypes.

    This function creates a plot showing the variance explained by AA models with different numbers of archetypes.
    The data is retrieved from `adata.uns["AA_var"]`. If `adata.uns["AA_var"]` is not found, `var_explained_aa` is called.

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
        + pn.theme(panel_grid_major=pn.element_line(color="gray", size=0.5, alpha=0.5), figure_size=(6, 3))
    )
    return p


def plot_IC(adata: sc.AnnData) -> pn.ggplot:
    """
    Generate a plot showing an information criteria for a range of archetypes.

    This function creates a plot showing the variance explained by AA models with different numbers of archetypes.
    The data is retrieved from `adata.uns["AA_var"]`. If `adata.uns["AA_var"]` is not found, `var_explained_aa` is called.

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

    p = (
        pn.ggplot(plot_df)
        + pn.geom_line(mapping=pn.aes(x="k", y="IC"), color="black")
        + pn.geom_point(mapping=pn.aes(x="k", y="IC"), color="black")
        + pn.labs(x="Number of Archetypes (k)", y="Information Criteria")
        + pn.scale_x_continuous(breaks=list(np.arange(plot_df["k"].min(), plot_df["k"].max() + 1)))
        + pn.theme_matplotlib()
        + pn.theme(panel_grid_major=pn.element_line(color="gray", size=0.5, alpha=0.5), figure_size=(6, 3))
    )
    return p


def plot_bootstrap_2D(adata: sc.AnnData, show_two_panels: bool = True) -> pn.ggplot:
    """
    Visualize the distribution and stability of archetypes across bootstrap samples in 2D PCA space.

    Creates a static 2D scatter plot showing the positions of archetypes
    computed from bootstrap samples, stored in `adata.uns["AA_bootstrap"]`.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data object containing the archetype bootstrap data in `adata.uns["AA_bootstrap"]`.

    Returns
    -------
    pn.ggplot
        A 2D scatter plot visualizing the bootstrap results for the archetypes.
    """
    # Validation input
    if "AA_bootstrap" not in adata.uns:
        raise ValueError("AA_bootstrap not found in adata.uns. Please run bootstrap_aa() to compute")

    # Generate the 2D scatter plot
    plot_df = adata.uns["AA_bootstrap"].copy()

    if ("x2" in plot_df.columns.to_list()) and show_two_panels:
        plot_df = plot_df.melt(
            id_vars=["x0", "archetype", "reference"], value_vars=["x1", "x2"], var_name="variable", value_name="value"
        )
        p = (
            pn.ggplot(plot_df)
            + pn.geom_point(pn.aes(x="x0", y="value", color="archetype", shape="reference"))
            + pn.facet_wrap(facets="variable", scales="fixed")
            + pn.labs(x="First Axis", y="Second / Third Axis")
            + pn.coord_equal()
        )
    else:
        p = (
            pn.ggplot(plot_df)
            + pn.geom_point(pn.aes(x="x0", y="x1", color="archetype", shape="reference"))
            + pn.coord_equal()
        )
    return p


def plot_bootstrap_3D(adata: sc.AnnData) -> go.Figure:
    """
    Interactive 3D visualization of archetypes from bootstrap samples to assess their variability.

    Create an interactive 3D scatter plot showing the positions of archetypes
    computed from bootstrap samples, stored in `adata.uns["AA_bootstrap"]`.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data object containing the archetype bootstrap data in `adata.uns["AA_bootstrap"]`.

    Returns
    -------
    go.Figure
        A 3D scatter plot visualizing the bootstrap results for the archetypes.
    """
    # Validation input
    if "AA_bootstrap" not in adata.uns:
        raise ValueError("AA_bootstrap not found in adata.uns. Please run bootstrap_aa() to compute")

    # Generate the 3D scatter plot
    bootstrap_df = adata.uns["AA_bootstrap"]
    fig = px.scatter_3d(
        bootstrap_df,
        x="x0",
        y="x1",
        z="x2",
        color="archetype",
        symbol="reference",
        title="Archetypes on bootstrapepd data",
        size_max=10,
        hover_data=["iter", "archetype", "reference"],
        opacity=0.5,
    )
    fig.update_layout(template=None)

    return fig


def plot_bootstrap_multiple_k(adata: sc.AnnData) -> pn.ggplot:
    """
    Visualize archetype stability as a function of the number of archetypes.

    This function generates a plot summarizing the stability of archetypes across different
    numbers of archetypes (`k`), based on bootstrap variance metrics. It displays individual
    archetype variances as points, along with summary statistics (median and maximum variance)
    as lines.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data object containing the results from `bootstrap_aa_multiple_k` in
        `adata.uns["AA_boostrap_multiple_k"]`.

    Returns
    -------
    pn.ggplot
        A ggplot object displaying:
        - Scatter points for individual archetype variances (`variance_per_archetype`) as a function of `n_archetypes`.
        - Lines and points for the median and maximum variance across archetypes at each `n_archetypes`.
    """
    if "AA_boostrap_multiple_k" not in adata.uns:
        raise ValueError(
            "bootstrap_aa_multiple_k not found in adata.uns. Please run bootstrap_aa_multiple_k() to compute"
        )
    df = adata.uns["AA_boostrap_multiple_k"]
    df_summary = df.groupby("n_archetypes")["variance_per_archetype"].agg(["median", "max"]).reset_index()
    df_summary = df_summary.melt(id_vars="n_archetypes", value_vars=["median", "max"])
    p = (
        pn.ggplot()
        + pn.geom_point(data=df, mapping=pn.aes(x="n_archetypes", y="variance_per_archetype"), alpha=0.5, size=3)
        + pn.geom_line(data=df_summary, mapping=pn.aes(x="n_archetypes", y="value", color="variable"))
        + pn.geom_point(data=df_summary, mapping=pn.aes(x="n_archetypes", y="value", color="variable"))
        + pn.labs(x="Number of Archetypes", y="Value", color="Variance\nSummary")
    )
    return p


def plot_archetypes_2D(
    adata: sc.AnnData, color: str | None = None, alpha: float = 1.0, show_two_panels: bool = True
) -> pn.ggplot:
    """
    Generate a static 2D scatter plot showing data points, archetypes and the polytope they span.

    This function visualizes the archetypes computed via Archetypal Analysis (AA)
    in PCA space, along with the data points. An optional color vector can be used
    to annotate the data points.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data object containing the archetypes in `adata.uns["AA_results"]["Z"]`
        and PCA-reduced data in `adata.obsm["X_pca"]`.
    color : str or None, optional
        Column name in `adata.obs` to use for coloring the data points. If None, no coloring is applied.

    Returns
    -------
    pn.ggplot
        A static 2D scatter plot showing the data and archetypes.
    """
    _validate_aa_config(adata)
    _validate_aa_results(adata)
    obsm_key = adata.uns["aa_config"]["obsm_key"]
    n_dimensions = adata.uns["aa_config"]["n_dimension"]
    X = adata.obsm[obsm_key][:, :n_dimensions]
    Z = adata.uns["AA_results"]["Z"]
    color_vec = sc.get.obs_df(adata, color).values.flatten() if color else None
    plot = plot_2D(X=X, Z=Z, color_vec=color_vec, alpha=alpha, show_two_panels=show_two_panels)
    return plot


def plot_2D(
    X: np.ndarray, Z: np.ndarray, color_vec: np.ndarray | None = None, alpha: float = 1.0, show_two_panels: bool = True
) -> pn.ggplot:
    """
    2D plot of the datapoints in X and the 2D polytope enclosed by the archetypes in Z.

    Parameters
    ----------
    X : np.ndarray
        A 2D array of shape (n_samples, n_features) representing the data points.
    Z : np.ndarray
        A 2D array of shape (n_archetypes, n_features) representing the archetype coordinates.
    color_vec : np.ndarray, optional
        A 1D array of shape (n_samples,) containing values for coloring the data points in `X`.

    Returns
    -------
    pn.ggplot
        2D plot of X and polytope enclosed by Z.
    """
    if X.shape[1] < 2 or Z.shape[1] < 2:
        raise ValueError("Both X and Z must have at least 2 columns (PCs).")
    if color_vec is not None:
        if len(color_vec) != len(X):
            raise ValueError("color_vec must have the same length as X.")

    if (X.shape[1] > 2) and show_two_panels:
        data_df = pd.DataFrame(X[:, :3], columns=["x0", "x1", "x2"])
        if color_vec is not None:
            data_df["color_vec"] = np.array(color_vec)
            data_df = data_df.melt(
                id_vars=["x0", "color_vec"], value_vars=["x1", "x2"], var_name="variable", value_name="value"
            )
        else:
            data_df = data_df.melt(id_vars=["x0"], value_vars=["x1", "x2"], var_name="variable", value_name="value")
        arch_df_list = []
        for dim in range(1, 3):
            order = np.argsort(np.arctan2(Z[:, dim] - np.mean(Z[:, dim]), Z[:, 0] - np.mean(Z[:, 0])))
            arch_df = pd.DataFrame(Z[:, [0, dim]], columns=["x0", "value"])
            arch_df["variable"] = f"x{dim}"
            arch_df["archetype_label"] = np.arange(arch_df.shape[0])
            arch_df = arch_df.iloc[order].reset_index(drop=True)
            arch_df = pd.concat([arch_df, arch_df.iloc[:1]], ignore_index=True)
            arch_df_list.append(arch_df)
        arch_df = pd.concat(arch_df_list)
    else:
        data_df = pd.DataFrame(X[:, :2], columns=["x0", "value"])
        if color_vec is not None:
            data_df["color_vec"] = np.array(color_vec)
        data_df["variable"] = "x1"
        order = np.argsort(np.arctan2(Z[:, 1] - np.mean(Z[:, 1]), Z[:, 0] - np.mean(Z[:, 0])))
        arch_df = pd.DataFrame(Z[:, [0, 1]], columns=["x0", "value"])
        arch_df["variable"] = "x1"
        arch_df["archetype_label"] = np.arange(arch_df.shape[0])
        arch_df = arch_df.iloc[order].reset_index(drop=True)
        arch_df = pd.concat([arch_df, arch_df.iloc[:1]], ignore_index=True)

    # Generate plot
    plot = pn.ggplot()

    if color_vec is not None:
        plot += pn.geom_point(data=data_df, mapping=pn.aes(x="x0", y="value", color="color_vec"), alpha=alpha)
    else:
        plot += pn.geom_point(data=data_df, mapping=pn.aes(x="x0", y="value"), color="black", alpha=alpha)

    plot += pn.geom_point(data=arch_df, mapping=pn.aes(x="x0", y="value"), color="red", size=1)
    plot += pn.geom_path(data=arch_df, mapping=pn.aes(x="x0", y="value"), color="red", size=1)
    plot += pn.geom_label(
        data=arch_df, mapping=pn.aes(x="x0", y="value", label="archetype_label"), color="black", size=12
    )
    plot += pn.facet_wrap(facets="variable", scales="fixed")
    plot += pn.labs(x="First Axis", y="Second / Third Axis")
    plot += pn.coord_equal()

    return plot


def plot_archetypes_3D(adata: sc.AnnData, color: str | None = None) -> pn.ggplot:
    """
    Create an interactive 3D scatter plot showing data points, archetypes and the polytope they span.

    This function uses the first three principal components from `adata.obsm["X_pca"]`
    and visualizes the archetypes stored in `adata.uns["AA_results"]["Z"]`.
    If a color key is provided, it colors data points by the corresponding values from `adata.obs`.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data object containing the PCA-reduced data in `obsm["X_pca"]` and
        archetypes in `uns["AA_results"]["Z"]`.
    color : str, optional
        Name of a column in `adata.obs` to color the data points by.

    Returns
    -------
    go.Figure
        A Plotly figure object showing a 3D scatter plot of the data and archetypes.
    """
    _validate_aa_config(adata)
    _validate_aa_results(adata)
    obsm_key = adata.uns["aa_config"]["obsm_key"]
    n_dimensions = adata.uns["aa_config"]["n_dimension"]
    X = adata.obsm[obsm_key][:, :n_dimensions]
    Z = adata.uns["AA_results"]["Z"]
    color_vec = sc.get.obs_df(adata, color).values.flatten() if color else None
    plot = plot_3D(X=X, Z=Z, color_vec=color_vec)
    return plot


def plot_3D(
    X: np.ndarray,
    Z: np.ndarray,
    color_vec: np.ndarray | None = None,
    marker_size: int = 4,
    color_polyhedron: str = "green",
) -> go.Figure:
    """
    Generates a 3D plot of data points and the polytope formed by archetypes.

    Parameters
    ----------
    X : np.ndarray
        A 2D array of shape (n_samples, n_features) representing the data points.
    Z : np.ndarray
        A 2D array of shape (n_archetypes, n_features) representing the archetype coordinates.
    color_vec : np.ndarray, optional
        A 1D array of shape (n_samples,) containing values for coloring the data points in `X`.
    marker_size : int, optional (default=4)
        The size of the markers for the data points in `X`.
    color_polyhedron : str, optional (default="green")
        The color of the polytope defined by the archetypes.

    Returns
    -------
    go.Figure
        3D plot of X and polytope enclosed by Z.
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

    fig.update_layout(template=None)
    return fig


def barplot_meta_enrichment(meta_enrich: pd.DataFrame, meta: str = "Meta"):
    """
    Generate a stacked bar plot showing metadata enrichment across archetypes.

    Parameters
    ----------
    meta_enrich: pd.DataFrame
        Output of `meta_enrichment()`, a DataFrame where rows are archetypes and columns are metadata categories,
        with values representing normalized enrichment scores.
    meta : str, optional
        Label to use for the metadata category legend in the plot. Default is "Meta".

    Returns
    -------
    pn.ggplot.ggplot
        A stacked bar plot of metadata enrichment per archetype.
    """
    # prepare data
    meta_enrich = meta_enrich.reset_index().rename(columns={"index": "archetype"})
    meta_enrich_long = meta_enrich.melt(id_vars=["archetype"], var_name="Meta", value_name="Normalized_Enrichment")

    # get unique categories and assign colors
    categories = meta_enrich_long["Meta"].unique()
    color_palette = hue_pal()(len(categories))

    # Create plot
    plot = (
        pn.ggplot(
            meta_enrich_long,
            pn.aes(x="factor(archetype)", y="Normalized_Enrichment", fill="Meta"),
        )
        + pn.geom_bar(stat="identity", position="stack")
        + pn.theme_matplotlib()
        # + pn.scale_fill_brewer(type="qual", palette="Dark2")
        + pn.scale_fill_manual(values=color_palette)
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
    Generate a heatmap showing metadata enrichment across archetypes.

    Parameters
    ----------
    meta_enrich: pd.DataFrame
        Output of `meta_enrichment()`, a DataFrame where rows are archetypes and columns are metadata categories,
        with values representing normalized enrichment scores.
    meta : str, optional
        Label to use for the metadata category legend in the plot. Default is "Meta".

    Returns
    -------
    pn.ggplot.ggplot
        A heatmap of normalized enrichment scores per archetype and metadata category.
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


def barplot_functional_enrichment(top_features: dict, show: bool = True):
    """
    Generate bar plots showing functional enrichment scores for each archetype.

    Each plot displays the top enriched features (e.g., biological processes) for one archetype.

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
    specific_processes_arch : pd.DataFrame
            Output from `extract_specific_processes`. Must contain a 'Process' column, a 'specificity' score,
            and one column per archetype with enrichment values.

    Returns
    -------
    plotnine.ggplot.ggplot
        A grouped bar plot visualizing the enrichment scores for the specified features across archetypes."
    """
    # Subset the DataFrame to include only the specified features
    process_order = specific_processes_arch.sort_values("specificity", ascending=False)["Process"].to_list()
    arch_columns = specific_processes_arch.drop(columns=["Process", "specificity"]).columns.to_list()
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
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # Initialise the radar plot
        ax = plt.subplot(int(np.ceil(len(meta_enrich) / 2)), 2, row + 1, polar=True)

        # Put first axis on top:
        ax.set_theta_offset(np.pi / 2)
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
