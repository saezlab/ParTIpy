import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotnine as pn
import scanpy as sc
from matplotlib import ticker
from mizani.palettes import hue_pal
from scipy.spatial import ConvexHull

from .paretoti import _validate_aa_config, _validate_aa_results, compute_selection_metrics


def plot_var_explained(adata: anndata.AnnData, ymin: None | float = None, ymax: None | float = None) -> pn.ggplot:
    """
    Generate an elbow plot of the variance explained by Archetypal Analysis (AA) for a range of archetypes.

    This function creates a plot showing the variance explained by AA models with different numbers of archetypes.
    The data is retrieved from `adata.uns["AA_metrics"]`. If `adata.uns["AA_metrics"]` is not found, `var_explained_aa` is called.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the variance explained data in `adata.uns["AA_metrics"]`.
    ymin : None | float

    ymax : None | float
        specify y

    Returns
    -------
    pn.ggplot
        A ggplot object showing the variance explained plot.
    """
    # Validation input
    if "AA_metrics" not in adata.uns:
        print("AA_var not found in adata.uns. Computing variance explained by archetypal analysis...")
        compute_selection_metrics(adata=adata)
    if ymin:
        assert (ymin >= 0.0) and (ymin < 1.0)
    if ymax:
        assert (ymax > 0.0) and (ymax <= 1.0)
    if ymin and ymax:
        assert ymax > ymin

    plot_df = adata.uns["AA_metrics"]
    plot_df_summary = plot_df.groupby("k")["varexpl"].mean().reset_index()

    # Create data for the diagonal line
    diag_data = pd.DataFrame(
        {
            "k": [plot_df_summary["k"].min(), plot_df_summary["k"].max()],
            "varexpl": [plot_df_summary["varexpl"].min(), plot_df_summary["varexpl"].max()],
        }
    )

    p = (
        pn.ggplot()
        + pn.geom_line(data=plot_df_summary, mapping=pn.aes(x="k", y="varexpl"), color="black", linetype="dashed")
        + pn.geom_point(data=plot_df, mapping=pn.aes(x="k", y="varexpl"), color="black")
        + pn.geom_line(data=diag_data, mapping=pn.aes(x="k", y="varexpl"), color="gray")
        + pn.labs(x="Number of Archetypes (k)", y="Variance Explained")
        + pn.scale_x_continuous(breaks=list(np.arange(plot_df["k"].min(), plot_df["k"].max() + 1)))
        + pn.theme_matplotlib()
        + pn.theme(panel_grid_major=pn.element_line(color="gray", size=0.5, alpha=0.5), figure_size=(6, 3))
    )
    if ymin and ymax:
        p += pn.ylim((ymin, ymax))
    elif ymin:
        p += pn.ylim((ymin, None))
    elif ymax:
        p += pn.ylim((None, ymax))
    return p


def plot_IC(adata: anndata.AnnData) -> pn.ggplot:
    """
    Generate a plot showing an information criteria for a range of archetypes.

    This function creates a plot showing the variance explained by AA models with different numbers of archetypes.
    The data is retrieved from `adata.uns["AA_metrics"]`. If `adata.uns["AA_metrics"]` is not found, `var_explained_aa` is called.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the variance explained data in `adata.uns["AA_metrics"]`.

    Returns
    -------
    pn.ggplot
        A ggplot object showing the variance explained plot.
    """
    # Validation input
    if "AA_metrics" not in adata.uns:
        print("AA_var not found in adata.uns. Computing variance explained by archetypal analysis...")
        compute_selection_metrics(adata=adata)

    plot_df = adata.uns["AA_metrics"]
    plot_df_summary = plot_df.groupby("k")["IC"].mean().reset_index()

    p = (
        pn.ggplot()
        + pn.geom_line(data=plot_df_summary, mapping=pn.aes(x="k", y="IC"), color="black", linetype="dashed")
        + pn.geom_point(data=plot_df, mapping=pn.aes(x="k", y="IC"), color="black")
        + pn.labs(x="Number of Archetypes (k)", y="Information Criteria")
        + pn.scale_x_continuous(breaks=list(np.arange(plot_df["k"].min(), plot_df["k"].max() + 1)))
        + pn.theme_matplotlib()
        + pn.theme(panel_grid_major=pn.element_line(color="gray", size=0.5, alpha=0.5), figure_size=(6, 3))
    )
    return p


def plot_bootstrap_2D(
    adata: anndata.AnnData,
    n_archetypes: int,
    show_two_panels: bool = False,
    alpha: float = 1.0,
    size: float | None = None,
) -> pn.ggplot:
    """
    Visualize the distribution and stability of archetypes across bootstrap samples in 2D PCA space.

    Creates a static 2D scatter plot showing the positions of archetypes
    computed from bootstrap samples, stored in `adata.uns["AA_bootstrap"]`.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing the archetype bootstrap data in `adata.uns["AA_bootstrap"]`.
    n_archetypes : int
        The number of archetypes used in the bootstrap analysis to visualize. This should match the a number in adata.uns["AA_bootstrap"] keys.
    show_two_panels : bool, default `False`
        If True, the plot will be split into two panels showing the archetypes from different orientations if there are more than 2 dimensions in the data.
    alpha : float, default `1.0`
        Opacity of the points in the scatter plot (0.0 to 1.0).
    size : float | None, default `None`
        Size of the points in the scatter plot. If None, uses the default size of the plotting library.

    Returns
    -------
    pn.ggplot
        A 2D scatter plot visualizing the bootstrap results for the archetypes.
    """
    n_archetypes_str = str(n_archetypes)
    # Validation input
    if "AA_bootstrap" not in adata.uns:
        raise ValueError("AA_bootstrap not found in adata.uns. Please run bootstrap_aa() to compute")

    if n_archetypes_str not in adata.uns["AA_bootstrap"].keys():
        raise ValueError(
            f"n_archetypes {n_archetypes_str} not found in adata.uns['AA_bootstrap']. Available keys: {list(adata.uns['AA_bootstrap'].keys())}"
        )
    # Generate the 2D scatter plot
    plot_df = adata.uns["AA_bootstrap"][n_archetypes_str].copy()

    point_args = {"alpha": alpha}
    if size is not None:
        point_args["size"] = size

    if ("x2" in plot_df.columns.to_list()) and show_two_panels:
        plot_df = plot_df.melt(
            id_vars=["x0", "archetype", "reference"], value_vars=["x1", "x2"], var_name="variable", value_name="value"
        )
        p = pn.ggplot(plot_df) + pn.geom_point(
            pn.aes(x="x0", y="value", color="archetype", shape="reference"),
            **point_args,  # type: ignore[arg-type]
        )
        p += pn.facet_wrap(facets="variable", scales="fixed")
        p += pn.labs(x="First PC", y="Second / Third PC")
    else:
        p = pn.ggplot(plot_df) + pn.geom_point(
            pn.aes(x="x0", y="x1", color="archetype", shape="reference"),
            **point_args,  # type: ignore[arg-type]
        )
    p += pn.coord_equal()

    return p


def plot_bootstrap_3D(
    adata: anndata.AnnData,
    n_archetypes: int,
    size: float = 6,
    alpha: float = 0.5,
) -> go.Figure:
    """
    Interactive 3D visualization of archetypes from bootstrap samples to assess their variability.

    Create an interactive 3D scatter plot showing the positions of archetypes
    computed from bootstrap samples, stored in `adata.uns["AA_bootstrap"]`.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing the archetype bootstrap data in `adata.uns["AA_bootstrap"]`.
    n_archetypes : int
        The number of archetypes used in the bootstrap analysis to visualize. This should match the a number in adata.uns["AA_bootstrap"] keys.
    size : float, default `6`
        Size of the points in the scatter plot.
    alpha : float, default `0.5`
        Opacity of the points in the scatter plot (0.0 to 1.0).

    Returns
    -------
    go.Figure
        A 3D scatter plot visualizing the bootstrap results for the archetypes.
    """
    n_archetypes_str = str(n_archetypes)
    # Validation input
    if "AA_bootstrap" not in adata.uns:
        raise ValueError("AA_bootstrap not found in adata.uns. Please run bootstrap_aa() to compute")

    if n_archetypes_str not in adata.uns["AA_bootstrap"].keys():
        raise ValueError(
            f"n_archetypes {n_archetypes_str} not found in adata.uns['AA_bootstrap']. Available keys: {list(adata.uns['AA_bootstrap'].keys())}"
        )

    # Generate the 3D scatter plot
    bootstrap_df = adata.uns["AA_bootstrap"][n_archetypes_str].copy()
    fig = px.scatter_3d(
        bootstrap_df,
        x="x0",
        y="x1",
        z="x2",
        color="archetype",
        symbol="reference",
        title="Archetypes on bootstrapepd data",
        hover_data=["iter", "archetype", "reference"],
        opacity=alpha,
    )
    fig.update_traces(marker={"size": size})
    fig.update_layout(template=None)

    return fig


def plot_bootstrap_variance(adata: anndata.AnnData) -> pn.ggplot:
    """
    Visualize archetype stability as a function of the number of archetypes.

    This function generates a plot summarizing the stability of archetypes across different
    numbers of archetypes (`k`), based on bootstrap variance metrics. It displays individual
    archetype variances as points, along with summary statistics (median and maximum variance)
    as lines.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing the results from `bootstrap_aa` in
        `adata.uns["AA_boostrap"]`.

    Returns
    -------
    pn.ggplot
        A ggplot object displaying:
        - Scatter points for individual archetype variances (`variance_per_archetype`) as a function of `n_archetypes`.
        - Lines and points for the median and maximum variance across archetypes at each `n_archetypes`.
    """
    if "AA_bootstrap" not in adata.uns:
        raise ValueError(
            "bootstrap_aa_multiple_k not found in adata.uns. Please run bootstrap_aa_multiple_k() to compute"
        )
    df_list = []
    df_dict = adata.uns["AA_bootstrap"]
    for n_archetypes, df in df_dict.items():
        # Add 'n_archetypes' column
        df = df.copy()
        df["n_archetypes"] = int(n_archetypes)

        # Drop duplicates
        df = df[["archetype", "variance_per_archetype", "n_archetypes"]].drop_duplicates()

        df_list.append(df)

    # Combine all into one DataFrame
    full_df = pd.concat(df_list, axis=0, ignore_index=True)

    # Group and summarize
    df_summary = full_df.groupby("n_archetypes")["variance_per_archetype"].agg(["median", "max"]).reset_index()

    # Melt for plotting or tabular output
    df_melted = df_summary.melt(
        id_vars="n_archetypes", value_vars=["median", "max"], var_name="variable", value_name="value"
    )

    p = (
        pn.ggplot()
        + pn.geom_line(
            data=df_melted,
            mapping=pn.aes(x="n_archetypes", y="value", linetype="variable"),
            size=1.5,
            alpha=0.5,
            color="grey",
        )
        + pn.geom_point(data=full_df, mapping=pn.aes(x="n_archetypes", y="variance_per_archetype"), alpha=0.5, size=3)
        + pn.labs(x="Number of Archetypes", y="Variance per Archetype", linetype="Variance\nSummary")
        + pn.scale_linetype_manual(values={"median": "dotted", "max": "solid"})
        + pn.theme(figure_size=(6, 3))
    )
    return p


def plot_archetypes_2D(
    adata: anndata.AnnData,
    color: str | None = None,
    alpha: float = 1.0,
    size: float | None = None,
    show_two_panels: bool = False,
) -> pn.ggplot:
    """
    Generate a static 2D scatter plot showing data points, archetypes and the polytope they span.

    This function visualizes the archetypes computed via Archetypal Analysis (AA)
    in PCA space, along with the data points. An optional color vector can be used
    to annotate the data points.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing the archetypes in `adata.uns["AA_results"]["Z"]`
        and PCA-reduced data in `adata.obsm["X_pca"]`.
    color : str | None, default `None`
        Column name in `adata.obs` to use for coloring the data points. If None, no coloring is applied.
    alpha : float, default `1.0`
        Opacity of the points in the scatter plot (0.0 to 1.0).
    size : float | None, default `None`
        Size of the points in the scatter plot. If None, uses the default size of the plotting library.
    show_two_panels : bool, default `False`
        If True, the plot will be split into two panels showing the archetypes from different orientations

    Returns
    -------
    pn.ggplot
        A static 2D scatter plot showing the data and archetypes.
    """
    _validate_aa_config(adata)
    _validate_aa_results(adata)
    obsm_key = adata.uns["AA_config"]["obsm_key"]
    n_dimensions = adata.uns["AA_config"]["n_dimensions"]
    X = adata.obsm[obsm_key][:, n_dimensions]
    Z = adata.uns["AA_results"]["Z"]
    color_vec = sc.get.obs_df(adata, color).values.flatten() if color else None
    plot = plot_2D(X=X, Z=Z, color_vec=color_vec, alpha=alpha, size=size, show_two_panels=show_two_panels)
    if color:
        plot += pn.labs(color=color)
    return plot


def plot_2D(
    X: np.ndarray,
    Z: np.ndarray,
    color_vec: np.ndarray | None = None,
    alpha: float = 1.0,
    size: float | None = None,
    show_two_panels: bool = False,
) -> pn.ggplot:
    """
    2D plot of the datapoints in X and the 2D polytope enclosed by the archetypes in Z.

    Parameters
    ----------
    X : np.ndarray
        A 2D array of shape (n_samples, n_features) representing the data points.
    Z : np.ndarray
        A 2D array of shape (n_archetypes, n_features) representing the archetype coordinates.
    color_vec : np.ndarray, default `None`
        A 1D array of shape (n_samples,) containing values for coloring the data points in `X`.
    alpha : float, default `1.0`
        Opacity of the points in the scatter plot (0.0 to 1.0).
    size : float | None, default `None`
        Size of the points in the scatter plot. If None, uses the default size of the plotting library.
    show_two_panels : bool, default `False`
        If True, the plot will be split into two panels showing the archetypes from different orientations

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
        if size is not None:
            plot += pn.geom_point(
                data=data_df, mapping=pn.aes(x="x0", y="value", color="color_vec"), alpha=alpha, size=size
            )
        else:
            plot += pn.geom_point(data=data_df, mapping=pn.aes(x="x0", y="value", color="color_vec"), alpha=alpha)
    else:
        if size is not None:
            plot += pn.geom_point(
                data=data_df, mapping=pn.aes(x="x0", y="value"), color="black", alpha=alpha, size=size
            )
        else:
            plot += pn.geom_point(data=data_df, mapping=pn.aes(x="x0", y="value"), color="black", alpha=alpha)

    plot += pn.geom_point(data=arch_df, mapping=pn.aes(x="x0", y="value"), color="red", size=1)
    plot += pn.geom_path(data=arch_df, mapping=pn.aes(x="x0", y="value"), color="red", size=1)
    plot += pn.geom_label(
        data=arch_df, mapping=pn.aes(x="x0", y="value", label="archetype_label"), color="black", size=12
    )
    plot += pn.facet_wrap(facets="variable", scales="fixed")
    plot += pn.labs(x="First PC", y="Second / Third PC")
    plot += pn.coord_equal()

    return plot


def plot_archetypes_3D(
    adata: anndata.AnnData, color: str | None = None, size: int = 4, alpha: float = 0.5, alpha_hull: float = 0.2
) -> pn.ggplot:
    """
    Create an interactive 3D scatter plot showing data points, archetypes and the polytope they span.

    This function uses the first three principal components from `adata.obsm["X_pca"]`
    and visualizes the archetypes stored in `adata.uns["AA_results"]["Z"]`.
    If a color key is provided, it colors data points by the corresponding values from `adata.obs`.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing the PCA-reduced data in `obsm["X_pca"]` and
        archetypes in `uns["AA_results"]["Z"]`.
    color : str, default `None`
        Name of a column in `adata.obs` to color the data points by.
    size : int, default `4`
        The size of the markers for the data points in `X`.
    alpha : float, default `0.5`
        Opacity of the points in the scatter plot (0.0 to 1.0).
    alpha : float, default `0.2`
        Opacity of the polytope spanned by the archetypes (0.0 to 1.0).

    Returns
    -------
    go.Figure
        A Plotly figure object showing a 3D scatter plot of the data and archetypes.
    """
    _validate_aa_config(adata)
    _validate_aa_results(adata)
    obsm_key = adata.uns["AA_config"]["obsm_key"]
    n_dimensions = adata.uns["AA_config"]["n_dimensions"]
    X = adata.obsm[obsm_key][:, n_dimensions]
    Z = adata.uns["AA_results"]["Z"]
    color_vec = sc.get.obs_df(adata, color).values.flatten() if color else None
    plot = plot_3D(X=X, Z=Z, color_vec=color_vec, size=size, alpha=alpha, alpha_hull=alpha_hull)
    return plot


def plot_3D(
    X: np.ndarray,
    Z: np.ndarray,
    color_vec: np.ndarray | None = None,
    size: int = 4,
    alpha: float = 0.5,
    alpha_hull: float = 0.2,
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
    color_vec : np.ndarray, default `None`
        A 1D array of shape (n_samples,) containing values for coloring the data points in `X`.
    size : int, default `4`
        The size of the markers for the data points in `X`.
    alpha : float, default `0.5`
        Opacity of the points in the scatter plot (0.0 to 1.0).
    color_polyhedron : str, default `green`
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
            opacity=alpha,
        )
    else:
        fig = px.scatter_3d(
            plot_df,
            x="x0",
            y="x1",
            z="x2",
            labels={"x0": "PC 1", "x1": "PC 2", "x2": "PC 3"},
            title="3D polytope",
            opacity=alpha,
        )

    fig.update_traces(marker={"size": size})

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

    if Z_plot.shape[0] > Z_plot.shape[1]:
        # Add the polytope (convex hull) to the plot
        hull = ConvexHull(Z_plot)
        fig.add_trace(
            go.Mesh3d(
                x=Z_plot[:, 0],
                y=Z_plot[:, 1],
                z=Z_plot[:, 2],
                i=hull.simplices[:, 0],
                j=hull.simplices[:, 1],
                k=hull.simplices[:, 2],
                color=color_polyhedron,
                opacity=alpha_hull,
            )
        )

    # Add edges of the polytope to the plot
    for arch_idx in range(Z_plot.shape[0]):
        fig.add_trace(
            go.Scatter3d(
                x=[Z_plot[arch_idx, 0]],
                y=[Z_plot[arch_idx, 1]],
                z=[Z_plot[arch_idx, 2]],
                mode="lines",
                line={"color": color_polyhedron, "width": 5},
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
    meta_enrich: `pd.DataFrame`
        Output of `meta_enrichment()`, a DataFrame where rows are archetypes and columns are metadata categories,
        with values representing normalized enrichment scores.
    meta : str, default `"Meta"`
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
    meta_enrich: `pd.DataFrame`
        Output of `meta_enrichment()`, a DataFrame where rows are archetypes and columns are metadata categories,
        with values representing normalized enrichment scores.
    meta : str, default `"Meta"`
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
    show: bool, default `True`
        Whether to show the plots immediately. If False, the plots are created but not displayed.

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
    specific_processes_arch : `pd.DataFrame`
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


def radarplot_meta_enrichment(meta_enrich: pd.DataFrame, color_map: None | dict = None):
    """
    Parameters
    ----------
    meta_enrich: `pd.DataFrame`
        Output of meta_enrichment(), a pd.DataFrame containing the enrichment of meta categories (columns) for all archetypes (rows).
    color_map: None | dict, default `None`
        A dictionary mapping meta categories to colors. If None, a default color palette is used.

    Returns
    -------
    plt.pyplot.Figure
        Radar plots for all archetypes.
    """
    # prepare data
    meta_enrich = meta_enrich.T.reset_index().rename(columns={"index": "Meta_feature"})
    if color_map:
        color_list = [color_map[feat] for feat in meta_enrich["Meta_feature"]]
    else:
        default_palette = plt.colormaps.get_cmap("Dark2")
        color_list = [default_palette(idx) for idx in range(len(meta_enrich))]
    numeric_meta_enrich = meta_enrich.drop(columns=["Meta_feature"]).astype(float)

    # function to create a radar plot for a given row
    def make_radar(row, title, color):
        # set number of meta categories
        categories = list(numeric_meta_enrich.columns)
        N = len(categories)

        # calculate angles for the radar plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # initialise the radar plot
        ax = plt.subplot(int(np.ceil(len(meta_enrich) / 2)), 2, row + 1, polar=True)

        # put first axis on top:
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # axis labels
        archetype_label = [f"A{i}" for i in range(len(categories))]
        plt.xticks(angles[:-1], archetype_label, color="grey", size=8)

        # values for this radar
        values = numeric_meta_enrich.loc[row].values.flatten().tolist()
        values += values[:1]

        # y-axis handling
        if np.allclose(numeric_meta_enrich.sum(axis=0), 1):
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.0"], color="grey", size=7)
        else:
            raw_ymax = max(values)
            locator = ticker.MaxNLocator(4)
            yticks = locator.tick_values(0, raw_ymax)
            ymax = yticks[-1]

            if ymax < 0.1:
                ytick_labels = [f"{y:.2e}" for y in yticks]
            elif ymax < 1:
                ytick_labels = [f"{y:.2f}" for y in yticks]
            elif ymax < 10:
                ytick_labels = [f"{y:.1f}" for y in yticks]
            else:
                ytick_labels = [f"{int(y)}" for y in yticks]

            ax.set_ylim(0, ymax)
            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_labels, color="grey", size=7)

        ax.set_rlabel_position(0)

        # Draw plot
        ax.plot(angles, values, color=color, linewidth=2, linestyle="solid")
        ax.fill(angles, values, color=color, alpha=0.4)

        # Add title
        plt.title(title, size=11, color=color, y=1.065)

    # Initialize figure
    my_dpi = 96
    fig = plt.figure(figsize=(1000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)

    # Generate plots
    for row, color in zip(range(len(meta_enrich.index)), color_list, strict=False):
        make_radar(
            row=row,
            title=f"Feature: {meta_enrich['Meta_feature'][row]}",
            color=color,
        )

    plt.close(fig)
    return fig
