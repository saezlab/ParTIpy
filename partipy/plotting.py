import colorsys

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
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import chi2

from ._docs import docs
from .paretoti import _validate_aa_config, _validate_aa_results, compute_selection_metrics

DEFAULT_ARCHETYPE_COLORS = {
    0: "#4e79a7",  # muted blue
    1: "#f28e2b",  # warm amber-orange
    2: "#59a14f",  # medium green
    3: "#b07aa1",  # muted violet
    4: "#edc948",  # golden yellow
    5: "#9c755f",  # soft brown
    6: "#bab0ab",  # gray-taupe
    7: "#76b7b2",  # muted teal
}


def generate_distinct_colors(n: int) -> list[str]:
    """Generate `n` distinct hex colors using HSL color space."""
    return [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for h in range(n)
        for r, g, b in [colorsys.hls_to_rgb(h / n, 0.5, 1.0)]
    ]


def _compute_contour_df_2D(bootstrap_df: pd.DataFrame, col_1: str, col_2: str, confidence_level: float = 0.95):
    chi2_val = chi2.ppf(confidence_level, df=2)  # 2 DOF for 2D
    n_points = 100
    theta = np.linspace(0, 2 * np.pi, n_points)
    contour_df_list = []
    for arch_idx in bootstrap_df["archetype"].unique():
        arch_df = bootstrap_df.loc[bootstrap_df["archetype"] == arch_idx, :].copy()

        arch_mtx = arch_df[[col_1, col_2]].values

        arch_mean = arch_mtx.mean(axis=0)
        arch_cov = np.cov(arch_mtx, rowvar=False, bias=True)

        # Eigendecomposition of covariance matrix
        eigenvals, eigenvecs = np.linalg.eigh(arch_cov)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

        # Semi-axes lengths scaled by chi-square value
        a = np.sqrt(chi2_val * eigenvals[0])  # semi-major axis
        b = np.sqrt(chi2_val * eigenvals[1])  # semi-minor axis

        # Generate ellipse points
        ellipse_x = a * np.cos(theta)
        ellipse_y = b * np.sin(theta)

        # Rotate ellipse
        cos_angle = np.cos(np.radians(angle))
        sin_angle = np.sin(np.radians(angle))

        x_rot = ellipse_x * cos_angle - ellipse_y * sin_angle
        y_rot = ellipse_x * sin_angle + ellipse_y * cos_angle

        # Translate to mean
        contour_x = x_rot + arch_mean[0]
        contour_y = y_rot + arch_mean[1]

        # Create dataframe for the contour
        contour_df_list.append(pd.DataFrame({col_1: contour_x, col_2: contour_y, "archetype": arch_idx}))
    contour_df = pd.concat(contour_df_list)
    contour_df["archetype"] = pd.Categorical(contour_df["archetype"])
    return contour_df


def _compute_contour_df_3D(
    bootstrap_df: pd.DataFrame, col_1: str, col_2: str, col_3: str, confidence_level: float = 0.95
):
    chi2_val = chi2.ppf(confidence_level, df=3)  # 3 DOF for 3D

    # Create spherical coordinates for sampling points on unit sphere
    n_theta = 100  # azimuthal angle resolution
    n_phi = 50  # polar angle resolution

    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)

    # Create meshgrid for spherical coordinates
    THETA, PHI = np.meshgrid(theta, phi)

    # Convert to Cartesian coordinates on unit sphere
    x_sphere = np.sin(PHI) * np.cos(THETA)
    y_sphere = np.sin(PHI) * np.sin(THETA)
    z_sphere = np.cos(PHI)

    # Flatten to get list of points
    sphere_points = np.column_stack([x_sphere.flatten(), y_sphere.flatten(), z_sphere.flatten()])

    contour_df_list = []

    for arch_idx in bootstrap_df["archetype"].unique():
        arch_df = bootstrap_df.loc[bootstrap_df["archetype"] == arch_idx, :].copy()
        arch_mtx = arch_df[[col_1, col_2, col_3]].values

        arch_mean = arch_mtx.mean(axis=0)
        arch_cov = np.cov(arch_mtx, rowvar=False, bias=True)

        # Eigendecomposition of covariance matrix
        eigenvals, eigenvecs = np.linalg.eigh(arch_cov)

        # Scale by chi-square value and eigenvalues
        scaling_matrix = eigenvecs @ np.diag(np.sqrt(chi2_val * eigenvals))

        # Transform unit sphere points to ellipsoid
        ellipsoid_points = sphere_points @ scaling_matrix.T + arch_mean

        # Create dataframe for the contour surface
        contour_df_list.append(
            pd.DataFrame(
                {
                    col_1: ellipsoid_points[:, 0],
                    col_2: ellipsoid_points[:, 1],
                    col_3: ellipsoid_points[:, 2],
                    "archetype": arch_idx,
                }
            )
        )

    contour_df = pd.concat(contour_df_list)
    contour_df["archetype"] = pd.Categorical(contour_df["archetype"])
    return contour_df


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


@docs.dedent
def plot_bootstrap_2D(
    adata: anndata.AnnData,
    n_archetypes: int,
    dimensions: list[int] | None = None,
    show_contours: bool = True,
    contours_confidence_level: float = 0.95,
    contours_size: float = 2.0,
    contours_alpha: float = 0.75,
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
    dimensions : list[int] | None, default `None`
        List of 2 dimension indices to plot. If None, uses first 2 dimensions specified in `adata.uns["AA_config"]["n_dimensions"]`.
    show_contours : bool, default `True`
        If True, a multivariate Gaussian distribution is fit per archetype, and a contour line for one confidence level is shown.
    contours_confidence_level: float, default `0.95`
        Which confidence should be used to create the contour line
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
    _validate_aa_config(adata=adata)

    obsm_key = adata.uns["AA_config"]["obsm_key"]
    n_dimensions = adata.uns["AA_config"]["n_dimensions"]

    if dimensions is None:
        dimensions = n_dimensions[:2]

    if len(dimensions) != 2:
        raise ValueError("dimensions must contain exactly 2 dimensions for 2D plotting")

    bootstrap_df = adata.uns["AA_bootstrap"][n_archetypes_str].copy()

    if show_contours:
        contour_df = _compute_contour_df_2D(
            bootstrap_df=bootstrap_df,
            col_1=f"{obsm_key}_{dimensions[0]}",
            col_2=f"{obsm_key}_{dimensions[1]}",
            confidence_level=contours_confidence_level,
        )

    point_args = {"alpha": alpha}
    if size is not None:
        point_args["size"] = size

    p = (
        pn.ggplot(bootstrap_df)
        + pn.geom_point(
            pn.aes(
                x=f"{obsm_key}_{dimensions[0]}",
                y=f"{obsm_key}_{dimensions[1]}",
                color="archetype",
                shape="reference",
            ),
            **point_args,  # type: ignore[arg-type]
        )
        + pn.coord_equal()
        + pn.labs(color="Archetype\nIndex", shape="Reference\nArchetype")
    )

    if show_contours:
        p += pn.geom_path(
            pn.aes(
                x=f"{obsm_key}_{dimensions[0]}",
                y=f"{obsm_key}_{dimensions[1]}",
                color="archetype",
            ),
            data=contour_df,
            linetype="solid",
            size=contours_size,
            alpha=contours_alpha,
        )

    # use default archetype colors if number of archetypes is below 8
    if int(n_archetypes_str) < len(DEFAULT_ARCHETYPE_COLORS):
        p += pn.scale_color_manual(values=DEFAULT_ARCHETYPE_COLORS)

    return p


def plot_bootstrap_3D(
    adata: anndata.AnnData,
    n_archetypes: int,
    dimensions: list[int] | None = None,
    show_contours: bool = True,
    contours_confidence_level: float = 0.95,
    contours_alpha: float = 0.3,
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
    dimensions : list[int] | None, default `None`
        List of 3 dimension indices to plot. If None, uses first 3 dimensions.
    show_contours : bool, default `True`
        Whether to show confidence ellipsoids for each archetype.
    contours_confidence_level : float, default `0.95`
        Confidence level for the ellipsoids (0.0 to 1.0).
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
    if (contours_confidence_level >= 1) or (contours_confidence_level <= 0):
        raise ValueError("contours_confidence_level must be in the interval (0, 1)")
    _validate_aa_config(adata=adata)

    obsm_key = adata.uns["AA_config"]["obsm_key"]
    n_dimensions = adata.uns["AA_config"]["n_dimensions"]

    if dimensions is None:
        dimensions = n_dimensions[:3]

    if len(dimensions) != 3:
        raise ValueError("dimensions must contain exactly 3 dimensions for 3D plotting")

    bootstrap_df = adata.uns["AA_bootstrap"][n_archetypes_str].copy()

    if show_contours:
        contour_df = _compute_contour_df_3D(
            bootstrap_df=bootstrap_df,
            col_1=f"{obsm_key}_{dimensions[0]}",
            col_2=f"{obsm_key}_{dimensions[1]}",
            col_3=f"{obsm_key}_{dimensions[2]}",
            confidence_level=contours_confidence_level,
        )

    unique_archetypes = sorted(bootstrap_df["archetype"].unique())
    if len(unique_archetypes) <= len(DEFAULT_ARCHETYPE_COLORS):
        color_discrete_map = DEFAULT_ARCHETYPE_COLORS
    else:
        palette = generate_distinct_colors(len(unique_archetypes))
        color_discrete_map = {arch: palette[i] for i, arch in enumerate(unique_archetypes)}

    fig = px.scatter_3d(
        bootstrap_df,
        x=f"{obsm_key}_{dimensions[0]}",
        y=f"{obsm_key}_{dimensions[1]}",
        z=f"{obsm_key}_{dimensions[2]}",
        color="archetype",
        symbol="reference",
        title="Archetypes on bootstrapped data",
        hover_data=["iter", "archetype", "reference"],
        opacity=alpha,
        color_discrete_map=color_discrete_map,
    )
    fig.update_traces(marker={"size": size})

    if show_contours:
        # create mesh surface for each archetype
        for arch_idx in contour_df["archetype"].unique():
            arch_contour = contour_df[contour_df["archetype"] == arch_idx]

            fig.add_trace(
                go.Mesh3d(
                    x=arch_contour[f"{obsm_key}_{dimensions[0]}"],
                    y=arch_contour[f"{obsm_key}_{dimensions[1]}"],
                    z=arch_contour[f"{obsm_key}_{dimensions[2]}"],
                    opacity=contours_alpha,
                    color=color_discrete_map[arch_idx],
                    name=f"Contour {arch_idx}",
                    showlegend=True,
                    alphahull=0,  # creates convex hull
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        template="plotly_white",
        scene={
            "xaxis_title": f"{obsm_key}_{dimensions[0]}",
            "yaxis_title": f"{obsm_key}_{dimensions[1]}",
            "zaxis_title": f"{obsm_key}_{dimensions[2]}",
        },
    )

    return fig


def plot_bootstrap_variance(adata: anndata.AnnData, summary_method: str = "median") -> pn.ggplot:
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
    if summary_method not in ["median", "max", "mean"]:
        raise ValueError('summary_method must be either of ["median", "max", "mean"]')
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
    df_summary = full_df.groupby("n_archetypes")["variance_per_archetype"].agg(["median", "max", "mean"]).reset_index()

    p = (
        pn.ggplot()
        + pn.geom_line(
            data=df_summary,
            mapping=pn.aes(x="n_archetypes", y=summary_method),
            linetype="dotted",
            size=1.5,
            alpha=0.5,
            color="grey",
        )
        + pn.geom_point(data=full_df, mapping=pn.aes(x="n_archetypes", y="variance_per_archetype"), alpha=0.5, size=3)
        + pn.labs(x="Number of Archetypes", y="Variance per Archetype", linetype="Variance\nSummary")
        + pn.scale_x_continuous(breaks=list(range(2, max([int(k) for k in adata.uns["AA_bootstrap"].keys()]) + 1)))
        + pn.theme_matplotlib()
        + pn.theme(panel_grid_major=pn.element_line(color="gray", size=0.5, alpha=0.5), figure_size=(6, 3))
    )

    return p


@docs.dedent
def plot_archetypes_2D(
    adata: anndata.AnnData,
    dimensions: list[int] | None = None,
    show_contours: bool = False,
    contours_confidence_level: float = 0.95,
    contours_size: float = 2.0,
    contours_alpha: float = 0.75,
    color: str | None = None,
    alpha: float = 0.5,
    size: float | None = None,
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
    dimensions : list[int] | None, default `None`
        List of two integers specifying the dimensions to plot. If None, uses the first two dimensions.
    show_contours : bool, default `True`
        If True, a multivariate Gaussian distribution is fit per archetype, and a contour line for one confidence level is shown.
    color : str | None, default `None`
        Column name in `adata.obs` to use for coloring the data points. If None, no coloring is applied.
    alpha : float, default `1.0`
        Opacity of the points in the scatter plot (0.0 to 1.0).
    size : float | None, default `None`
        Size of the points in the scatter plot. If None, uses the default size of the plotting library.

    Returns
    -------
    pn.ggplot
        A static 2D scatter plot showing the data and archetypes.
    """
    _validate_aa_config(adata)
    _validate_aa_results(adata)

    if (contours_confidence_level >= 1) or (contours_confidence_level <= 0):
        raise ValueError("contours_confidence_level must be in the interval (0, 1)")

    obsm_key = adata.uns["AA_config"]["obsm_key"]
    n_dimensions = adata.uns["AA_config"]["n_dimensions"]

    if dimensions is None:
        dimensions = n_dimensions[:2]

    if len(dimensions) != 2:
        raise ValueError("dimensions must contain exactly 2 dimensions for 2D plotting")

    if show_contours:
        n_archetypes_str = str(adata.uns["AA_results"]["Z"].shape[0])

        if "AA_bootstrap" not in adata.uns:
            raise ValueError("AA_bootstrap not found in adata.uns. Please run bootstrap_aa() to compute")

        if n_archetypes_str not in adata.uns["AA_bootstrap"].keys():
            raise ValueError(
                f"n_archetypes {n_archetypes_str} not found in adata.uns['AA_bootstrap']. Available keys: {list(adata.uns['AA_bootstrap'].keys())}"
            )
        bootstrap_df = adata.uns["AA_bootstrap"][n_archetypes_str].copy()
        contour_df = _compute_contour_df_2D(
            bootstrap_df=bootstrap_df,
            col_1=f"{obsm_key}_{dimensions[0]}",
            col_2=f"{obsm_key}_{dimensions[1]}",
            confidence_level=contours_confidence_level,
        )

    data_df = pd.DataFrame(
        {
            f"{obsm_key}_{dimensions[0]}": adata.obsm[obsm_key][:, dimensions[0]],
            f"{obsm_key}_{dimensions[1]}": adata.obsm[obsm_key][:, dimensions[1]],
        }
    )
    if color is not None:
        color_vec = sc.get.obs_df(adata, color).values.flatten()
        data_df[color] = np.array(color_vec)

    arch_df = pd.DataFrame(
        {
            f"{obsm_key}_{dimensions[0]}": adata.uns["AA_results"]["Z"][:, dimensions[0]],
            f"{obsm_key}_{dimensions[1]}": adata.uns["AA_results"]["Z"][:, dimensions[1]],
            "archetype": np.arange(adata.uns["AA_results"]["Z"].shape[0]),
        }
    )
    arch_df["archetype"] = pd.Categorical(arch_df["archetype"])
    # reorder such that the polygon can be drawn
    Z = adata.uns["AA_results"]["Z"][:, dimensions].copy()
    order = np.argsort(np.arctan2(Z[:, 1] - np.mean(Z[:, 1]), Z[:, 0] - np.mean(Z[:, 0])))
    arch_df = arch_df.iloc[order].reset_index(drop=True)

    point_args = {"alpha": alpha}
    if size is not None:
        point_args["size"] = size

    p = pn.ggplot() + pn.coord_equal()

    # if we have more than 2 archetypes add the polygon
    if adata.uns["AA_results"]["Z"].shape[0] > 2:
        p += pn.geom_polygon(
            data=arch_df,
            mapping=pn.aes(
                x=f"{obsm_key}_{dimensions[0]}",
                y=f"{obsm_key}_{dimensions[1]}",
            ),
            color="#000080",
            size=1,
            alpha=0.05,
        )

    if color:
        p += pn.geom_point(
            data=data_df,
            mapping=pn.aes(
                x=f"{obsm_key}_{dimensions[0]}",
                y=f"{obsm_key}_{dimensions[1]}",
                color=color,
            ),
            **point_args,  # type: ignore[arg-type]
        )

        if show_contours:
            p += pn.geom_path(
                data=contour_df,
                mapping=pn.aes(
                    x=f"{obsm_key}_{dimensions[0]}",
                    y=f"{obsm_key}_{dimensions[1]}",
                    linetype="archetype",
                ),
                color="#000080",
                size=contours_size,
                alpha=contours_alpha,
            )
            p += pn.scale_linetype_manual(values=dict.fromkeys(contour_df["archetype"].unique(), "solid"))

            p += pn.scale_size_manual(values=dict.fromkeys(contour_df["archetype"].unique(), 1))

        p += pn.geom_point(
            data=arch_df,
            mapping=pn.aes(x=f"{obsm_key}_{dimensions[0]}", y=f"{obsm_key}_{dimensions[1]}", size="archetype"),
        )

        p += pn.geom_label(
            data=arch_df,
            mapping=pn.aes(x=f"{obsm_key}_{dimensions[0]}", y=f"{obsm_key}_{dimensions[1]}", label="archetype"),
            size=12,
        )
        p += pn.guides(size=False, linetype=False)

    else:
        p += pn.geom_point(
            data=data_df,
            mapping=pn.aes(
                x=f"{obsm_key}_{dimensions[0]}",
                y=f"{obsm_key}_{dimensions[1]}",
            ),
            **point_args,  # type: ignore[arg-type]
        )

        if show_contours:
            p += pn.geom_path(
                data=contour_df,
                mapping=pn.aes(
                    x=f"{obsm_key}_{dimensions[0]}",
                    y=f"{obsm_key}_{dimensions[1]}",
                    color="archetype",
                ),
                linetype="solid",
                size=contours_size,
                alpha=contours_alpha,
            )

        p += pn.geom_point(
            data=arch_df,
            mapping=pn.aes(
                x=f"{obsm_key}_{dimensions[0]}",
                y=f"{obsm_key}_{dimensions[1]}",
                color="archetype",
            ),
            size=1,
        )
        p += pn.geom_label(
            data=arch_df,
            mapping=pn.aes(
                x=f"{obsm_key}_{dimensions[0]}", y=f"{obsm_key}_{dimensions[1]}", label="archetype", color="archetype"
            ),
            size=12,
        )

        if adata.uns["AA_results"]["Z"].shape[0] < len(DEFAULT_ARCHETYPE_COLORS):
            p += pn.scale_color_manual(values=DEFAULT_ARCHETYPE_COLORS)

        p += pn.guides(color=False)

    return p


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
    adata: anndata.AnnData,
    dimensions: list[int] | None = None,
    show_contours: bool = False,
    contours_confidence_level: float = 0.95,
    contours_alpha: float = 0.3,
    color: str | None = None,
    size: float = 2.0,
    alpha: float = 0.2,
    alpha_hull: float = 0.2,
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
    size : int, default `2`
        The size of the markers for the data points in `X`.
    alpha : float, default `0.2`
        Opacity of the points in the scatter plot (0.0 to 1.0).
    alpha_hull : float, default `0.2`
        Opacity of the polytope spanned by the archetypes (0.0 to 1.0).

    Returns
    -------
    go.Figure
        A Plotly figure object showing a 3D scatter plot of the data and archetypes.
    """
    color_polyhedron = "#000080"
    color_points = "#000000"

    _validate_aa_config(adata)
    _validate_aa_results(adata)
    Z = adata.uns["AA_results"]["Z"].copy()

    if (contours_confidence_level >= 1) or (contours_confidence_level <= 0):
        raise ValueError("contours_confidence_level must be in the interval (0, 1)")

    obsm_key = adata.uns["AA_config"]["obsm_key"]
    n_dimensions = adata.uns["AA_config"]["n_dimensions"]

    if dimensions is None:
        dimensions = n_dimensions[:3]

    if len(dimensions) != 3:
        raise ValueError("dimensions must contain exactly 3 dimensions for 3D plotting")

    if show_contours:
        n_archetypes_str = str(adata.uns["AA_results"]["Z"].shape[0])

        if "AA_bootstrap" not in adata.uns:
            raise ValueError("AA_bootstrap not found in adata.uns. Please run bootstrap_aa() to compute")

        if n_archetypes_str not in adata.uns["AA_bootstrap"].keys():
            raise ValueError(
                f"n_archetypes {n_archetypes_str} not found in adata.uns['AA_bootstrap']. Available keys: {list(adata.uns['AA_bootstrap'].keys())}"
            )
        bootstrap_df = adata.uns["AA_bootstrap"][n_archetypes_str].copy()
        contour_df = _compute_contour_df_3D(
            bootstrap_df=bootstrap_df,
            col_1=f"{obsm_key}_{dimensions[0]}",
            col_2=f"{obsm_key}_{dimensions[1]}",
            col_3=f"{obsm_key}_{dimensions[2]}",
            confidence_level=contours_confidence_level,
        )

    data_df = pd.DataFrame(
        {
            f"{obsm_key}_{dimensions[0]}": adata.obsm[obsm_key][:, dimensions[0]],
            f"{obsm_key}_{dimensions[1]}": adata.obsm[obsm_key][:, dimensions[1]],
            f"{obsm_key}_{dimensions[2]}": adata.obsm[obsm_key][:, dimensions[2]],
        }
    )
    if color is not None:
        color_vec = sc.get.obs_df(adata, color).values.flatten() if color else None
        data_df["color"] = np.array(color_vec)
    else:
        data_df["color"] = "uniform_color"

    arch_df = pd.DataFrame(
        {
            f"{obsm_key}_{dimensions[0]}": adata.uns["AA_results"]["Z"][:, dimensions[0]],
            f"{obsm_key}_{dimensions[1]}": adata.uns["AA_results"]["Z"][:, dimensions[1]],
            f"{obsm_key}_{dimensions[2]}": adata.uns["AA_results"]["Z"][:, dimensions[2]],
            "archetype": np.arange(adata.uns["AA_results"]["Z"].shape[0]),
        }
    )
    arch_df["archetype"] = pd.Categorical(arch_df["archetype"])

    # take care of the colors
    unique_archetypes = sorted(arch_df["archetype"].unique())
    if len(unique_archetypes) <= len(DEFAULT_ARCHETYPE_COLORS):
        color_discrete_map = DEFAULT_ARCHETYPE_COLORS
    else:
        palette = generate_distinct_colors(len(unique_archetypes))
        color_discrete_map = {arch: palette[i] for i, arch in enumerate(unique_archetypes)}

    # Create the 3D scatter plot
    if color is not None:
        fig = px.scatter_3d(
            data_df,
            x=f"{obsm_key}_{dimensions[0]}",
            y=f"{obsm_key}_{dimensions[1]}",
            z=f"{obsm_key}_{dimensions[2]}",
            title="",
            color="color",
            opacity=alpha,
        )
    else:
        fig = px.scatter_3d(
            data_df,
            x=f"{obsm_key}_{dimensions[0]}",
            y=f"{obsm_key}_{dimensions[1]}",
            z=f"{obsm_key}_{dimensions[2]}",
            title="",
            opacity=alpha,
            color="color",
            color_discrete_sequence=[color_points],
        )

    fig.update_traces(marker={"size": size})

    # Add archetypes to the plot
    archetype_labels = [f"Archetype {i}" for i in range(Z.shape[0])]
    archetype_colors = [color_discrete_map.get(i, color_polyhedron) for i in range(Z.shape[0])]
    fig.add_trace(
        go.Scatter3d(
            x=Z[:, 0],
            y=Z[:, 1],
            z=Z[:, 2],
            mode="markers",
            text=archetype_labels,
            marker={"size": 8, "color": archetype_colors, "symbol": "diamond"},
            hoverinfo="text",
            name="Archetypes",
        )
    )

    # Add the polytope (convex hull) if we have enough archetypes
    if Z.shape[0] > Z.shape[1]:
        try:
            hull = ConvexHull(Z)
            fig.add_trace(
                go.Mesh3d(
                    x=Z[:, 0],
                    y=Z[:, 1],
                    z=Z[:, 2],
                    i=hull.simplices[:, 0],
                    j=hull.simplices[:, 1],
                    k=hull.simplices[:, 2],
                    color=color_polyhedron,
                    opacity=alpha_hull,
                    name="Polytope",
                )
            )
        except QhullError as e:
            print(f"Warning: Could not create convex hull: {e}")

    # Add edges connecting archetypes (optional - you might want to make this configurable)
    for i in range(Z.shape[0]):
        for j in range(i + 1, Z.shape[0]):
            fig.add_trace(
                go.Scatter3d(
                    x=[Z[i, 0], Z[j, 0]],
                    y=[Z[i, 1], Z[j, 1]],
                    z=[Z[i, 2], Z[j, 2]],
                    mode="lines",
                    line={"color": color_polyhedron, "width": 2},
                    showlegend=False,
                    opacity=0.3,
                )
            )

    if show_contours:
        # create mesh surface for each archetype
        for arch_idx in contour_df["archetype"].unique():
            arch_contour = contour_df[contour_df["archetype"] == arch_idx]

            fig.add_trace(
                go.Mesh3d(
                    x=arch_contour[f"{obsm_key}_{dimensions[0]}"],
                    y=arch_contour[f"{obsm_key}_{dimensions[1]}"],
                    z=arch_contour[f"{obsm_key}_{dimensions[2]}"],
                    opacity=contours_alpha,
                    color=color_discrete_map[arch_idx],
                    name=f"Contour {arch_idx}",
                    showlegend=True,
                    alphahull=0,  # Creates convex hull
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        template="plotly_white",
        scene={
            "xaxis_title": f"{obsm_key}_{dimensions[0]}",
            "yaxis_title": f"{obsm_key}_{dimensions[1]}",
            "zaxis_title": f"{obsm_key}_{dimensions[2]}",
        },
    )

    return fig


def barplot_meta_enrichment(meta_enrich: pd.DataFrame, meta: str = "Meta", color_map: None | dict = None):
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
    if not color_map:
        categories = meta_enrich_long["Meta"].unique()
        sorted_categories = sorted(categories)
        colors = hue_pal()(len(sorted_categories))
        color_map = dict(zip(sorted_categories, colors, strict=False))

    # Create plot
    plot = (
        pn.ggplot(
            meta_enrich_long,
            pn.aes(x="factor(archetype)", y="Normalized_Enrichment", fill="Meta"),
        )
        + pn.geom_bar(stat="identity", position="stack")
        + pn.theme_matplotlib()
        + pn.scale_fill_manual(values=color_map)
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
    if not color_map:
        categories = meta_enrich["Meta_feature"].unique()
        sorted_categories = sorted(categories)
        colors = hue_pal()(len(sorted_categories))
        color_map = dict(zip(sorted_categories, colors, strict=False))
    color_list = [color_map[feat] for feat in meta_enrich["Meta_feature"]]
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


def plot_top_genes(archetype_expression, arch_idx, top_n=20):
    """TODO"""
    archetype_expression_long = archetype_expression.reset_index(names="archetype").melt(
        id_vars="archetype", var_name="feature", value_name="expression"
    )
    top_features = (
        archetype_expression_long.loc[archetype_expression_long["archetype"] == arch_idx, :]
        .sort_values("expression")
        .tail(top_n)
        .loc[:, "feature"]
        .tolist()
    )
    plot_df = archetype_expression_long.loc[archetype_expression_long["feature"].isin(top_features), :].copy()

    feature_order = plot_df.loc[plot_df["archetype"] == arch_idx].sort_values("expression")["feature"].tolist()
    archetype_order = list(range(len(plot_df["archetype"].unique())))

    plot_df["archetype"] = pd.Categorical(plot_df["archetype"], categories=archetype_order)
    plot_df["feature"] = pd.Categorical(plot_df["feature"], categories=feature_order)

    p = (
        pn.ggplot(plot_df)
        + pn.geom_col(pn.aes(x="feature", y="expression", fill="archetype"), position=pn.position_dodge())
        + pn.coord_flip()
        + pn.labs(y="Expression", x="Feature", fill="Archetype")
    )

    return p
