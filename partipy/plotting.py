import colorsys
from collections.abc import Mapping
from typing import Any

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
from .paretoti import (
    _ensure_bootstrap_dict,
    _matches,
    _resolve_aa_result,
    _validate_aa_config,
    _validate_aa_results,
    get_aa_bootstrap,
    summarize_aa_metrics,
)
from .schema import ArchetypeConfig, query_configs_by_signature

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


def _config_to_filters(cfg: ArchetypeConfig) -> dict[str, Any]:
    """Convert an `ArchetypeConfig` into filters usable by the AA getter utilities."""
    filters = cfg.model_dump()
    optim_kwargs = filters.get("optim_kwargs")
    if isinstance(optim_kwargs, Mapping):
        filters["optim_kwargs"] = dict(optim_kwargs)
    elif isinstance(optim_kwargs, tuple):
        filters["optim_kwargs"] = dict(optim_kwargs)
    return filters


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


def plot_var_explained(
    adata: anndata.AnnData,
    ymin: None | float = None,
    ymax: None | float = None,
    result_filters: Mapping[str, Any] | None = None,
) -> pn.ggplot:
    """
    Generate an elbow plot of the variance explained by Archetypal Analysis (AA) for a range of archetypes.

    This function creates a plot showing the variance explained by AA models with different numbers of archetypes.
    Cached selection metrics are summarized on demand. Selection metrics must be computed beforehand via
    :func:`~partipy.compute_selection_metrics`.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing cached selection metrics in ``adata.uns["AA_selection_metrics"]``.
    ymin : None | float

    ymax : None | float
        specify y
    result_filters : Mapping[str, Any] | None, default ``None``
        Optional filters applied to ``ArchetypeConfig`` entries when summarizing cached metrics.

    Returns
    -------
    pn.ggplot
        A ggplot object showing the variance explained plot.
    """
    if "AA_selection_metrics" not in adata.uns:
        raise ValueError(
            "No cached selection metrics found in `adata.uns['AA_selection_metrics']`. "
            "Run `compute_selection_metrics` before calling `plot_var_explained`."
        )

    filters = dict(result_filters or {})
    plot_df = summarize_aa_metrics(adata, **filters)
    if ymin:
        assert (ymin >= 0.0) and (ymin < 1.0)
    if ymax:
        assert (ymax > 0.0) and (ymax <= 1.0)
    if ymin and ymax:
        assert ymax > ymin

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


def plot_IC(adata: anndata.AnnData, result_filters: Mapping[str, Any] | None = None) -> pn.ggplot:
    """
    Generate a plot showing an information criteria for a range of archetypes.

    This function creates a plot showing the variance explained by AA models with different numbers of archetypes.
    Cached selection metrics are summarized on demand. Selection metrics must be computed beforehand via
    :func:`~partipy.compute_selection_metrics`.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing cached selection metrics in ``adata.uns["AA_selection_metrics"]``.
    result_filters : Mapping[str, Any] | None, default ``None``
        Optional filters applied to ``ArchetypeConfig`` entries when summarizing cached metrics.

    Returns
    -------
    pn.ggplot
        A ggplot object showing the variance explained plot.
    """
    if "AA_selection_metrics" not in adata.uns:
        raise ValueError(
            "No cached selection metrics found in `adata.uns['AA_selection_metrics']`. "
            "Run `compute_selection_metrics` before calling `plot_IC`."
        )

    filters = dict(result_filters or {})
    plot_df = summarize_aa_metrics(adata, **filters)
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
    dimensions: list[int] | None = None,
    show_contours: bool = True,
    contours_confidence_level: float = 0.95,
    contours_size: float = 2.0,
    contours_alpha: float = 0.75,
    alpha: float = 1.0,
    size: float | None = None,
    result_filters: Mapping[str, Any] | None = None,
) -> pn.ggplot:
    """
    Visualize the distribution and stability of archetypes across bootstrap samples in 2D PCA space.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing the archetype bootstrap data in ``adata.uns["AA_bootstrap"]``.
    dimensions : list[int] | None, default `None`
        List of 2 dimension indices to plot. If None, uses the first two dimensions specified in the AA configuration.
    show_contours : bool, default `True`
        If True, a multivariate Gaussian distribution is fit per archetype, and a contour line for one confidence level is shown.
    contours_confidence_level : float, default `0.95`
        Confidence level for the contour line (0.0 to 1.0).
    alpha : float, default `1.0`
        Opacity of the points in the scatter plot (0.0 to 1.0).
    size : float | None, default `None`
        Size of the points in the scatter plot. If None, uses the default size of the plotting library.
    result_filters : Mapping[str, Any] | None, default `None`
        Filters applied to ``ArchetypeConfig`` entries to select the optimization configuration whose bootstrap
        runs are visualized. If unspecified, there must be only one configuration stored.

    Returns
    -------
    pn.ggplot
        A 2D scatter plot visualizing the bootstrap results for the archetypes.
    """
    _validate_aa_config(adata=adata)

    if not (0 < contours_confidence_level < 1):
        raise ValueError("contours_confidence_level must be in the interval (0, 1)")

    bootstrap_store = _ensure_bootstrap_dict(adata)

    filters = dict(result_filters or {})

    candidate_configs = [cfg for cfg in bootstrap_store.keys() if isinstance(cfg, ArchetypeConfig)]
    if not candidate_configs:
        raise ValueError("No ArchetypeConfig entries found in `adata.uns['AA_bootstrap']`.")

    if filters:
        filtered_configs = [cfg for cfg in candidate_configs if _matches(cfg, filters)]
        if not filtered_configs:
            raise ValueError(
                "No bootstrap entries match the provided filters. "
                "Ensure bootstrap_variance was computed for a configuration matching result_filters."
            )
        reference_cfg = filtered_configs[0]
        matched_configs = query_configs_by_signature(
            candidate_configs,
            reference_cfg,
            ignore_fields=("n_archetypes",),
        )
        if any(cfg not in matched_configs for cfg in filtered_configs):
            raise ValueError(
                "Filters correspond to multiple optimization configurations. "
                "Please provide more specific result_filters (e.g., init, optim, weight)."
            )
    else:
        reference_cfg = candidate_configs[0]
        matched_configs = query_configs_by_signature(
            candidate_configs,
            reference_cfg,
            ignore_fields=("n_archetypes",),
        )
        if len(matched_configs) != len(candidate_configs):
            raise ValueError(
                "Multiple optimization configurations present. Please provide result_filters to select one."
            )

    cfg = reference_cfg
    filters_for_getter = _config_to_filters(cfg)
    bootstrap_df = get_aa_bootstrap(adata, **filters_for_getter).copy()

    obsm_key = cfg.obsm_key
    cfg_dims = tuple(cfg.n_dimensions)

    if dimensions is None:
        if len(cfg_dims) < 2:
            raise ValueError("Need at least two dimensions available to plot in 2D.")
        plot_dims = list(cfg_dims[:2])
    else:
        if len(dimensions) != 2:
            raise ValueError("dimensions must contain exactly 2 dimensions for 2D plotting")
        plot_dims = list(dimensions)

    for dim in plot_dims:
        if dim not in cfg_dims:
            raise ValueError(f"Dimension {dim} not available in archetype result. Available: {cfg_dims}")

    x_col = f"{obsm_key}_{plot_dims[0]}"
    y_col = f"{obsm_key}_{plot_dims[1]}"

    contour_df = None
    if show_contours:
        contour_df = _compute_contour_df_2D(
            bootstrap_df=bootstrap_df,
            col_1=x_col,
            col_2=y_col,
            confidence_level=contours_confidence_level,
        )

    if size is None:
        point_layer = pn.geom_point(
            pn.aes(x=x_col, y=y_col, color="archetype", shape="reference"),
            alpha=alpha,
        )
    else:
        point_layer = pn.geom_point(
            pn.aes(x=x_col, y=y_col, color="archetype", shape="reference"),
            alpha=alpha,
            size=size,
        )

    p = (
        pn.ggplot(bootstrap_df)
        + point_layer
        + pn.coord_equal()
        + pn.labs(color="Archetype\nIndex", shape="Reference\nArchetype")
    )

    if show_contours and contour_df is not None:
        p += pn.geom_path(
            pn.aes(x=x_col, y=y_col, color="archetype"),
            data=contour_df,
            linetype="solid",
            size=contours_size,
            alpha=contours_alpha,
        )

    if cfg.n_archetypes < len(DEFAULT_ARCHETYPE_COLORS):
        p += pn.scale_color_manual(values=DEFAULT_ARCHETYPE_COLORS)

    return p


@docs.dedent
def plot_bootstrap_3D(
    adata: anndata.AnnData,
    dimensions: list[int] | None = None,
    show_contours: bool = True,
    contours_confidence_level: float = 0.95,
    contours_alpha: float = 0.3,
    size: float = 6,
    alpha: float = 0.5,
    result_filters: Mapping[str, Any] | None = None,
) -> go.Figure:
    """
    Interactive 3D visualization of archetypes from bootstrap samples to assess their variability.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing the archetype bootstrap data in ``adata.uns["AA_bootstrap"]``.
    dimensions : list[int] | None, default `None`
        Three dimension indices to plot. If None, uses the first three dimensions specified in the AA configuration.
    show_contours : bool, default `True`
        Whether to show confidence ellipsoids for each archetype.
    contours_confidence_level : float, default `0.95`
        Confidence level for the ellipsoids (0.0 to 1.0).
    size : float, default `6`
        Size of the points in the scatter plot.
    alpha : float, default `0.5`
        Opacity of the points in the scatter plot (0.0 to 1.0).
    result_filters : Mapping[str, Any] | None, default `None`
        Filters applied to ``ArchetypeConfig`` entries to select the optimization configuration whose bootstrap
        runs are visualized. If unspecified, there must be only one configuration stored.

    Returns
    -------
    go.Figure
        A 3D scatter plot visualizing the bootstrap results for the archetypes.
    """
    if not (0 < contours_confidence_level < 1):
        raise ValueError("contours_confidence_level must be in the interval (0, 1)")

    _validate_aa_config(adata=adata)

    bootstrap_store = _ensure_bootstrap_dict(adata)

    filters = dict(result_filters or {})

    candidate_configs = [cfg for cfg in bootstrap_store.keys() if isinstance(cfg, ArchetypeConfig)]
    if not candidate_configs:
        raise ValueError("No ArchetypeConfig entries found in `adata.uns['AA_bootstrap']`.")

    if filters:
        filtered_configs = [cfg for cfg in candidate_configs if _matches(cfg, filters)]
        if not filtered_configs:
            raise ValueError(
                "No bootstrap entries match the provided filters. "
                "Ensure bootstrap_variance was computed for a configuration matching result_filters."
            )
        reference_cfg = filtered_configs[0]
        matched_configs = query_configs_by_signature(
            candidate_configs,
            reference_cfg,
            ignore_fields=("n_archetypes",),
        )
        if any(cfg not in matched_configs for cfg in filtered_configs):
            raise ValueError(
                "Filters correspond to multiple optimization configurations. "
                "Please provide more specific result_filters (e.g., init, optim, weight)."
            )
    else:
        reference_cfg = candidate_configs[0]
        matched_configs = query_configs_by_signature(
            candidate_configs,
            reference_cfg,
            ignore_fields=("n_archetypes",),
        )
        if len(matched_configs) != len(candidate_configs):
            raise ValueError(
                "Multiple optimization configurations present. Please provide result_filters to select one."
            )

    cfg = reference_cfg
    filters_for_getter = _config_to_filters(cfg)
    bootstrap_df = get_aa_bootstrap(adata, **filters_for_getter).copy()

    obsm_key = cfg.obsm_key
    cfg_dims = tuple(cfg.n_dimensions)

    if dimensions is None:
        if len(cfg_dims) < 3:
            raise ValueError("Need at least three dimensions available to plot in 3D.")
        plot_dims = list(cfg_dims[:3])
    else:
        if len(dimensions) != 3:
            raise ValueError("dimensions must contain exactly 3 indices for 3D plotting")
        plot_dims = list(dimensions)

    for dim in plot_dims:
        if dim not in cfg_dims:
            raise ValueError(f"Dimension {dim} not available in archetype result. Available: {cfg_dims}")

    x_col = f"{obsm_key}_{plot_dims[0]}"
    y_col = f"{obsm_key}_{plot_dims[1]}"
    z_col = f"{obsm_key}_{plot_dims[2]}"

    contour_df = None
    if show_contours:
        contour_df = _compute_contour_df_3D(
            bootstrap_df=bootstrap_df,
            col_1=x_col,
            col_2=y_col,
            col_3=z_col,
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
        x=x_col,
        y=y_col,
        z=z_col,
        color="archetype",
        symbol="reference",
        title="Archetypes on bootstrapped data",
        hover_data=["iter", "archetype", "reference"],
        opacity=alpha,
        color_discrete_map=color_discrete_map,
    )
    fig.update_traces(marker={"size": size})

    if show_contours and contour_df is not None:
        for arch_idx in contour_df["archetype"].unique():
            arch_contour = contour_df[contour_df["archetype"] == arch_idx]

            fig.add_trace(
                go.Mesh3d(
                    x=arch_contour[x_col],
                    y=arch_contour[y_col],
                    z=arch_contour[z_col],
                    opacity=contours_alpha,
                    color=color_discrete_map[arch_idx],
                    name=f"Contour {arch_idx}",
                    showlegend=True,
                    alphahull=0,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        template="plotly_white",
        scene={
            "xaxis_title": x_col,
            "yaxis_title": y_col,
            "zaxis_title": z_col,
        },
    )

    return fig


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


def plot_bootstrap_variance(
    adata: anndata.AnnData,
    summary_method: str = "median",
    result_filters: Mapping[str, Any] | None = None,
) -> pn.ggplot:
    """
    Visualize archetype stability as a function of the number of archetypes using cached bootstrap results.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing bootstrap results in ``adata.uns["AA_bootstrap"]``.
    summary_method : str, default ``"median"``
        Summary statistic to display as a dotted line across archetype counts. Must be one of ``{"median", "max", "mean"}``.
    result_filters : Mapping[str, Any] | None, default `None`
        Filters forwarded to ``_resolve_aa_result`` to select the AA configuration for which
        bootstrap results are summarized. If ``None``, a configuration with matching bootstrap data must exist uniquely.

    Returns
    -------
    pn.ggplot
        A plot of per-archetype variance alongside the selected summary statistic across archetype counts.
    """
    if summary_method not in {"median", "max", "mean"}:
        raise ValueError("summary_method must be one of {'median', 'max', 'mean'}")

    bootstrap_store = _ensure_bootstrap_dict(adata)

    filters = dict(result_filters or {})

    matching_items: list[tuple[ArchetypeConfig, pd.DataFrame]] = []
    for cfg, df in bootstrap_store.items():
        if not isinstance(cfg, ArchetypeConfig):
            continue
        if filters and not _matches(cfg, filters):
            continue
        matching_items.append((cfg, df))

    if not matching_items:
        raise ValueError(
            "No bootstrap entries match the provided filters. "
            "Ensure bootstrap_variance was computed for a configuration matching result_filters."
        )

    matching_configs = [cfg for cfg, _ in matching_items]
    reference_cfg = matching_configs[0]
    equivalent_configs = set(
        query_configs_by_signature(matching_configs, reference_cfg, ignore_fields=("n_archetypes",))
    )

    if any(cfg not in equivalent_configs for cfg in matching_configs):
        raise ValueError(
            "Multiple optimization configurations match the provided filters. "
            "Please supply more specific result_filters (e.g., init, optim, weight)."
        )

    df_list = []
    for cfg, df in matching_items:
        if not isinstance(df, pd.DataFrame):
            continue
        copy_df = df.copy()
        copy_df["n_archetypes"] = cfg.n_archetypes
        df_list.append(copy_df[["archetype", "variance_per_archetype", "n_archetypes"]])

    if not df_list:
        raise ValueError(
            "Bootstrap entries found but no variance data available. "
            "Ensure bootstrap_variance results include 'variance_per_archetype'."
        )

    full_df = pd.concat(df_list, axis=0, ignore_index=True)
    full_df = full_df.dropna(subset=["variance_per_archetype"])  # defensive

    df_summary = full_df.groupby("n_archetypes")["variance_per_archetype"].agg(summary_method).reset_index()

    p = (
        pn.ggplot()
        + pn.geom_line(
            data=df_summary,
            mapping=pn.aes(x="n_archetypes", y="variance_per_archetype"),
            linetype="dotted",
            size=1.5,
            alpha=0.5,
            color="grey",
        )
        + pn.geom_point(
            data=full_df,
            mapping=pn.aes(x="n_archetypes", y="variance_per_archetype"),
            alpha=0.5,
            size=3,
        )
        + pn.labs(x="Number of Archetypes", y="Variance per Archetype", linetype="Variance Summary")
        + pn.scale_x_continuous(breaks=sorted(full_df["n_archetypes"].unique()))
        + pn.theme_matplotlib()
        + pn.theme(panel_grid_major=pn.element_line(color="gray", size=0.5, alpha=0.5), figure_size=(6, 3))
    )

    return p


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
    result_filters: Mapping[str, Any] | None = None,
) -> pn.ggplot:
    """
    Generate a static 2D scatter plot showing data points, archetypes and the polytope they span.

    This function visualizes the archetypes computed via Archetypal Analysis (AA)
    in PCA space, along with the data points. An optional color vector can be used
    to annotate the data points.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing archetypal analysis results in `adata.uns["AA_results"]`
        and PCA-reduced data in `adata.obsm[obsm_key]`.
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
    result_filters : Mapping[str, Any] | None, default `None`
        Filters forwarded to ``_resolve_aa_result`` to select a specific cached AA configuration.

    Returns
    -------
    pn.ggplot
        A static 2D scatter plot showing the data and archetypes.
    """
    _validate_aa_config(adata)
    _validate_aa_results(adata)

    if not (0 < contours_confidence_level < 1):
        raise ValueError("contours_confidence_level must be in the interval (0, 1)")

    filters = dict(result_filters or {})
    cfg, payload = _resolve_aa_result(adata, result_filters=filters)

    obsm_key = cfg.obsm_key
    cfg_dims = tuple(cfg.n_dimensions)
    dim_to_idx = {dim: idx for idx, dim in enumerate(cfg_dims)}

    Z_full = payload.get("Z")
    if Z_full is None:
        raise ValueError("Matched AA payload does not contain 'Z'.")

    if dimensions is None:
        if len(cfg_dims) < 2:
            raise ValueError("Need at least three dimensions available to plot in 2D.")
        plot_dims = list(cfg_dims[:2])
    else:
        if len(dimensions) != 2:
            raise ValueError("dimensions must contain exactly 2 dimensions for 2D plotting")
        plot_dims = list(dimensions)

    for dim in plot_dims:
        if dim not in dim_to_idx:
            raise ValueError(f"Dimension {dim} not available in archetype result. Available: {cfg_dims}")

    Z = np.column_stack([Z_full[:, dim_to_idx[dim]] for dim in plot_dims])

    contour_df = None
    if show_contours:
        filters_for_getter = _config_to_filters(cfg)
        try:
            bootstrap_df = get_aa_bootstrap(adata, **filters_for_getter).copy()
        except ValueError as err:
            raise ValueError(
                "AA bootstrap results not found for the selected configuration. "
                "Run `compute_bootstrap_variance` with matching parameters."
            ) from err
        contour_df = _compute_contour_df_2D(
            bootstrap_df=bootstrap_df,
            col_1=f"{obsm_key}_{plot_dims[0]}",
            col_2=f"{obsm_key}_{plot_dims[1]}",
            confidence_level=contours_confidence_level,
        )

    data_df = pd.DataFrame(
        {
            f"{obsm_key}_{plot_dims[0]}": adata.obsm[obsm_key][:, plot_dims[0]],
            f"{obsm_key}_{plot_dims[1]}": adata.obsm[obsm_key][:, plot_dims[1]],
        }
    )
    if color is not None:
        color_vec = sc.get.obs_df(adata, keys=[color])[color].values
        data_df[color] = np.array(color_vec)

    arch_df = pd.DataFrame(
        {
            f"{obsm_key}_{plot_dims[0]}": Z_full[:, dim_to_idx[plot_dims[0]]],
            f"{obsm_key}_{plot_dims[1]}": Z_full[:, dim_to_idx[plot_dims[1]]],
            "archetype": np.arange(Z_full.shape[0]),
        }
    )
    arch_df["archetype"] = pd.Categorical(arch_df["archetype"])
    x_col = f"{obsm_key}_{plot_dims[0]}"
    y_col = f"{obsm_key}_{plot_dims[1]}"
    # reorder such that the polygon can be drawn
    Z = np.column_stack([Z_full[:, dim_to_idx[dim]] for dim in plot_dims])
    order = np.argsort(np.arctan2(Z[:, 1] - np.mean(Z[:, 1]), Z[:, 0] - np.mean(Z[:, 0])))
    arch_df = arch_df.iloc[order].reset_index(drop=True)

    point_args = {"alpha": alpha}
    if size is not None:
        point_args["size"] = size

    p = pn.ggplot() + pn.coord_equal()

    # if we have more than 2 archetypes add the polygon
    if Z_full.shape[0] > 2:
        p += pn.geom_polygon(
            data=arch_df,
            mapping=pn.aes(
                x=x_col,
                y=y_col,
            ),
            color="#000080",
            size=1,
            alpha=0.05,
        )

    if color:
        p += pn.geom_point(
            data=data_df,
            mapping=pn.aes(
                x=x_col,
                y=y_col,
                color=color,
            ),
            **point_args,  # type: ignore[arg-type]
        )

        if show_contours and contour_df is not None:
            p += pn.geom_path(
                data=contour_df,
                mapping=pn.aes(
                    x=x_col,
                    y=y_col,
                    linetype="archetype",
                ),
                color="#000080",
                size=contours_size,
                alpha=contours_alpha,
            )
            unique_archetypes = list(contour_df["archetype"].unique())
            p += pn.scale_linetype_manual(values=dict.fromkeys(unique_archetypes, "solid"))
            p += pn.scale_size_manual(values=dict.fromkeys(unique_archetypes, 1))

        p += pn.geom_point(
            data=arch_df,
            mapping=pn.aes(x=x_col, y=y_col, size="archetype"),
        )

        p += pn.geom_label(
            data=arch_df,
            mapping=pn.aes(x=x_col, y=y_col, label="archetype"),
            size=12,
        )
        p += pn.guides(size=False, linetype=False)

    else:
        p += pn.geom_point(
            data=data_df,
            mapping=pn.aes(
                x=x_col,
                y=y_col,
            ),
            **point_args,  # type: ignore[arg-type]
        )

        if show_contours and contour_df is not None:
            p += pn.geom_path(
                data=contour_df,
                mapping=pn.aes(
                    x=x_col,
                    y=y_col,
                    color="archetype",
                ),
                linetype="solid",
                size=contours_size,
                alpha=contours_alpha,
            )

        p += pn.geom_point(
            data=arch_df,
            mapping=pn.aes(
                x=x_col,
                y=y_col,
                color="archetype",
            ),
            size=1,
        )
        p += pn.geom_label(
            data=arch_df,
            mapping=pn.aes(
                x=x_col,
                y=y_col,
                label="archetype",
                color="archetype",
            ),
            size=12,
        )

        if Z_full.shape[0] < len(DEFAULT_ARCHETYPE_COLORS):
            p += pn.scale_color_manual(values=DEFAULT_ARCHETYPE_COLORS)

        p += pn.guides(color=False)

    return p


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
    result_filters: Mapping[str, Any] | None = None,
) -> go.Figure:
    """
    Create an interactive 3D scatter plot showing data points, archetypes and the polytope they span.

    This function uses the principal components defined in the selected AA configuration and visualizes the
    archetypes stored in `adata.uns["AA_results"]`. If a color key is provided, it colors data points by the
    corresponding values from `adata.obs`.
    """
    color_polyhedron = "#000080"
    color_points = "#000000"

    _validate_aa_config(adata)
    _validate_aa_results(adata)

    if not (0 < contours_confidence_level < 1):
        raise ValueError("contours_confidence_level must be in the interval (0, 1)")

    filters = dict(result_filters or {})
    cfg, payload = _resolve_aa_result(adata, result_filters=filters)

    obsm_key = cfg.obsm_key
    cfg_dims = tuple(cfg.n_dimensions)
    dim_to_idx = {dim: idx for idx, dim in enumerate(cfg_dims)}

    Z_full = payload.get("Z")
    if Z_full is None:
        raise ValueError("Matched AA payload does not contain 'Z'.")

    if dimensions is None:
        if len(cfg_dims) < 3:
            raise ValueError("Need at least three dimensions available to plot in 3D.")
        plot_dims = list(cfg_dims[:3])
    else:
        if len(dimensions) != 3:
            raise ValueError("dimensions must contain exactly 3 dimensions for 3D plotting")
        plot_dims = list(dimensions)

    for dim in plot_dims:
        if dim not in dim_to_idx:
            raise ValueError(f"Dimension {dim} not available in archetype result. Available: {cfg_dims}")

    Z = np.column_stack([Z_full[:, dim_to_idx[dim]] for dim in plot_dims])

    contour_df = None
    if show_contours:
        filters_for_getter = _config_to_filters(cfg)
        try:
            bootstrap_df = get_aa_bootstrap(adata, **filters_for_getter).copy()
        except ValueError as err:
            raise ValueError(
                "AA bootstrap results not found for the selected configuration. "
                "Run `compute_bootstrap_variance` with matching parameters."
            ) from err
        contour_df = _compute_contour_df_3D(
            bootstrap_df=bootstrap_df,
            col_1=f"{obsm_key}_{plot_dims[0]}",
            col_2=f"{obsm_key}_{plot_dims[1]}",
            col_3=f"{obsm_key}_{plot_dims[2]}",
            confidence_level=contours_confidence_level,
        )

    X = adata.obsm[obsm_key]
    data_df = pd.DataFrame({f"{obsm_key}_{dim}": X[:, dim] for dim in plot_dims})
    if color is not None:
        color_vec = sc.get.obs_df(adata, keys=[color])[color].values
        data_df["color"] = np.asarray(color_vec)
    else:
        data_df["color"] = color_points

    arch_df = pd.DataFrame(
        {
            f"{obsm_key}_{plot_dims[0]}": Z[:, 0],
            f"{obsm_key}_{plot_dims[1]}": Z[:, 1],
            f"{obsm_key}_{plot_dims[2]}": Z[:, 2],
            "archetype": np.arange(Z.shape[0]),
        }
    )
    arch_df["archetype"] = pd.Categorical(arch_df["archetype"])

    unique_archetypes = sorted(arch_df["archetype"].unique())
    if len(unique_archetypes) <= len(DEFAULT_ARCHETYPE_COLORS):
        color_discrete_map = DEFAULT_ARCHETYPE_COLORS
    else:
        palette = generate_distinct_colors(len(unique_archetypes))
        color_discrete_map = {arch: palette[i] for i, arch in enumerate(unique_archetypes)}

    scatter_kwargs = dict(  # noqa
        x=f"{obsm_key}_{plot_dims[0]}",
        y=f"{obsm_key}_{plot_dims[1]}",
        z=f"{obsm_key}_{plot_dims[2]}",
        title="",
        opacity=alpha,
    )
    if color is not None:
        fig = px.scatter_3d(data_df, color="color", **scatter_kwargs)
    else:
        fig = px.scatter_3d(
            data_df,
            color="color",
            color_discrete_sequence=[color_points],
            **scatter_kwargs,
        )

    fig.update_traces(marker={"size": size})

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

    if alpha_hull > 0 and Z.shape[0] > Z.shape[1]:
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
        except (QhullError, ValueError):
            pass

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

    if show_contours and contour_df is not None:
        for arch_idx in contour_df["archetype"].unique():
            arch_contour = contour_df[contour_df["archetype"] == arch_idx]
            fig.add_trace(
                go.Mesh3d(
                    x=arch_contour[f"{obsm_key}_{plot_dims[0]}"],
                    y=arch_contour[f"{obsm_key}_{plot_dims[1]}"],
                    z=arch_contour[f"{obsm_key}_{plot_dims[2]}"],
                    opacity=contours_alpha,
                    color=color_discrete_map.get(arch_idx, "#1f78b4"),
                    name=f"Contour {arch_idx}",
                    showlegend=True,
                    alphahull=0,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        template="plotly_white",
        scene={
            "xaxis_title": f"{obsm_key}_{plot_dims[0]}",
            "yaxis_title": f"{obsm_key}_{plot_dims[1]}",
            "zaxis_title": f"{obsm_key}_{plot_dims[2]}",
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
