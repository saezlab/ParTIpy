import math

import numpy as np
import plotnine as pn
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import Tuple
import plotly.express as px
import plotly.graph_objects as go

from .arch import AA
from .const import (
    DEFAULT_INIT,
    DEFAULT_OPTIM
)

####TODO####
# Make functions work with andata object (add our dataframes to uns)
# Break functions up in those adding to andata and plotting
# add/fix t-ratio function
# Function mean archetype variance for different n_archetypes
# Functions to extract results more easily
############

def var_explained_aa(
        X: np.ndarray,
        min_a: int = 2, 
        max_a: int = 10,
        optim: str = DEFAULT_OPTIM, 
        init: str = DEFAULT_INIT
    ) -> Tuple[pn.ggplot, pn.ggplot, pn.ggplot, pd.DataFrame]:
    """
    Computes variance explained by Archetypal Analysis (AA) 
    for a range of archetypes (min_a to max_a) and returns:
    - Three plotnine plots (ggplot)
    - A Pandas DataFrame with variance explained data

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix for AA.
    min_a : int, optional (default=2)
        Minimum number of archetypes to test.
    max_a : int, optional (default=10)
        Maximum number of archetypes to test.
    optim : str, optional (default=DEFAULT_OPTIM)
        optimization function to use.
    init : str, optional (default=DEFAULT_INIT)
        initalization function to use.

    Returns:
    --------
    Tuple[pn.ggplot, pn.ggplot, pn.ggplot, pd.DataFrame]
        p1: ggplot - Variance Explained plot
        p2: ggplot - Distance to projected point plot
        p3: ggplot - Variance explained over (k-1) model
        plot_df: pd.DataFrame - Data for plotting
    """
    
    k_arr = np.arange(min_a, max_a+1)
    results = {}
    
    for k in k_arr:
        A, B, Z, RSS, varexpl = AA(n_archetypes=k, optim=optim, init=init).fit(X).return_all()
        results[k] = {"Z": Z, "A": A, "B": B, "RSS": RSS, "varexpl": varexpl}
       
    varexpl_values = np.array([results[k]["varexpl"] for k in k_arr])

    plot_df = pd.DataFrame(
        {
            "k": k_arr,
            "varexpl": varexpl_values,
            "varexpl_ontop": np.insert(np.diff(varexpl_values), 0, varexpl_values[0]
            ),
        }
    )
    
    diag_df = pd.concat([plot_df.head(1), plot_df.tail(1)])
    diag_df.loc[diag_df["k"] == 1, "varexpl"] = 0

    offset_vec = plot_df[["k", "varexpl"]].iloc[0].values
    proj_vec = (plot_df[["k", "varexpl"]].values - offset_vec)[-1, :][:, None]
    proj_mtx = proj_vec @ np.linalg.inv(proj_vec.T @ proj_vec) @ proj_vec.T
    proj_val = (proj_mtx @ (plot_df[["k", "varexpl"]].values - offset_vec).T).T + offset_vec
    proj_df = pd.DataFrame(proj_val, columns=["k", "varexpl"])
    plot_df["dist_to_projected"] = np.linalg.norm(plot_df[["k", "varexpl"]].values - proj_df[["k", "varexpl"]].values, axis=1)

    p1 = (
        pn.ggplot(plot_df)
        + pn.geom_line(data=plot_df, mapping=pn.aes(x="k", y="varexpl"), color="black")
        + pn.geom_point(data=plot_df, mapping=pn.aes(x="k", y="varexpl"), color="black")
        + pn.geom_line(data=diag_df, mapping=pn.aes(x="k", y="varexpl"), color="gray")
        + pn.labs(x="Number of Archetypes (k)", y="Variance Explained")
        + pn.lims(y=[0, 1])
        + pn.scale_x_continuous(breaks=np.arange(min_a, max_a + 1))
        + pn.theme_matplotlib()
    )

    p2 = (
        pn.ggplot()
        + pn.geom_col(data=plot_df, mapping=pn.aes(x="k", y="dist_to_projected"))
        + pn.scale_x_continuous(breaks=np.arange(min_a, max_a + 1))
        + pn.labs(x="Number of Archetypes (k)", y="Distance to Projected Point")
        + pn.theme_matplotlib()
    )

    p3 = (
        pn.ggplot(plot_df)
        + pn.geom_point(pn.aes(x="k", y="varexpl_ontop"), color="black")
        + pn.geom_line(pn.aes(x="k", y="varexpl_ontop"), color="black")
        + pn.labs(
            x="Number of Archetypes (k)", y="Variance Explained on Top of (k-1) Model"
        )
        + pn.scale_x_continuous(breaks=np.arange(min_a, max_a + 1))
        + pn.lims(y=(0, None))
        + pn.theme_matplotlib()
    )

    return p1, p2, p3, plot_df

def bootstrap_AA(
        X: np.ndarray, 
        n_bootstrap: int,
        n_archetypes: int, 
        optim: str = DEFAULT_OPTIM, 
        init: str = DEFAULT_INIT, 
        seed: int = 42, 
        plot: bool = True,
        **kwargs
    ) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Computes archetypes on bootstrap samples, aligns them with reference archetypes,  
    and returns the results along with an interactive 3D scatter plot.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_bootstrap : int
        Number of bootstrap samples.
    n_archetypes : int
        Number of archetypes.
    optim : str, optional (default=DEFAULT_OPTIM)
        optimization function to use.
    init : str, optional (default=DEFAULT_INIT)
        initalization function to use.
    seed : int, optional (default=42)
        Random seed for reproducibility.
    plot: bool, optional (default=True)
        If the 3D plot will be returned or not.

    Returns:
    --------
    Tuple[pd.DataFrame, go.Figure]
        bootstrap_df: DataFrame containing bootstrap archetype results.
        fig: Interactive 3D plot of the results. Only when plot=True
    """

    n_samples, n_features = X.shape
    rng = np.random.default_rng(seed)
    
    ref_Z = AA(n_archetypes=n_archetypes, optim = optim, init = init).fit(X).Z
    
    idx_bootstrap = rng.choice(n_samples, size=(n_bootstrap, n_samples), replace=True)
    Z_list = [
        AA(n_archetypes=n_archetypes, optim=optim, init=init).fit(X[idx, :]).Z 
        for idx in idx_bootstrap
    ]

    Z_list = [
        align_archetypes(ref_arch=ref_Z.copy(), query_arch=query_Z.copy())
        for query_Z in Z_list
    ]

    Z_stack = np.stack(Z_list)
    var_per_archetype = Z_stack.var(axis=0).mean(axis=1)
    mean_variance = var_per_archetype.mean()

    bootstrap_data = [
        pd.DataFrame(Z, columns=[f"pc_{i}" for i in range(n_features)])
        .assign(archetype=np.arange(n_archetypes), iter=i + 1) 
        for i, Z in enumerate(Z_list)
        ]
    bootstrap_df = pd.concat(bootstrap_data)
 
    df = pd.DataFrame(ref_Z, columns=[f"pc_{i}" for i in range(n_features)])
    df["archetype"] = np.arange(n_archetypes)
    df["iter"] = 0

    bootstrap_df = pd.concat((bootstrap_df, df), axis=0)
    bootstrap_df["reference"] = bootstrap_df["iter"] == 0
    bootstrap_df["archetype"] = pd.Categorical(bootstrap_df["archetype"])

    bootstrap_df["variance_per_archetype"] = bootstrap_df["archetype"].map(dict(zip(np.arange(n_archetypes), var_per_archetype)))
    bootstrap_df["mean_variance"] = mean_variance

    if plot:
        fig = px.scatter_3d(
            bootstrap_df,
            x='pc_0', 
            y='pc_1', 
            z='pc_2',
            color='archetype', 
            symbol="reference",
            labels={
                "pc_0": "PC 1",
                "pc_1": "PC 2",
                "pc_2": "PC 3",
            },
            title="Archetypes on bootstrapepd data",
            size_max=kwargs.pop("size_max", 10),
            hover_data=["iter", "archetype", "reference"],
            **kwargs,
        )
        fig.update_layout(template="none")

        return bootstrap_df, fig
    return bootstrap_df

def plot_2D(
        X: np.ndarray, 
        Z: np.ndarray, 
        color_vec: np.ndarray=None, 
        **kwargs
    ) -> pn.ggplot:
    """
    2D plot of the datapoints in X and the 2D polytope enclosed by the archetypes in Z.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    Z : np.ndarray
        Input archetype matrix of shape (n_samples, n_features).
    color_vec : np.ndarray (optional)
        Values for coloring the datapoints from X.

    Returns:
    --------
    p1: pn.ggplot
        2D plot of X and polytope enclosed by Z
    """
    if X.shape[1] < 2 or Z.shape[1] < 2:
        raise ValueError("Both X and Z must have at least 2 columns (PCs).")
    
    X_plot, Z_plot = X.copy(), Z.copy()

    X_plot, Z_plot = X_plot[:, :2], Z_plot[:, :2]

    plot_df = pd.DataFrame(X_plot, columns=["x0", "x1"])
    order = np.argsort(np.arctan2(Z_plot[:, 1] - np.mean(Z_plot[:, 1]), 
                                  Z_plot[:, 0] - np.mean(Z_plot[:, 0])))
    
    arch_df = pd.DataFrame(Z_plot, columns=["x0", "x1"])
    arch_df = arch_df.iloc[order].reset_index(drop=True)
    arch_df = pd.concat([arch_df, arch_df.iloc[:1]], ignore_index=True)

    p1 = pn.ggplot()

    if color_vec is not None:
        if len(color_vec) != len(plot_df):
            raise ValueError("color_vec must have the same length as X.")
        plot_df["color_vec"] = np.array(color_vec)  
        p1 += pn.geom_point(data=plot_df, mapping=pn.aes(x="x0", y="x1", color="color_vec"), **kwargs)
    else:
        p1 += pn.geom_point(data=plot_df, mapping=pn.aes(x="x0", y="x1"), color="black", **kwargs)

    p1 += pn.geom_point(data=arch_df, mapping=pn.aes(x="x0", y="x1"), color="red", size=1)
    p1 += pn.geom_path(data=arch_df, mapping=pn.aes(x="x0", y="x1"), color="red", size=1)

    p1 += pn.labs(x="PC 1", y="PC 2")
    p1 += pn.theme_matplotlib()

    return p1

def plot_3D(
        X: np.ndarray, 
        Z: np.ndarray, 
        color_vec: np.ndarray=None, 
        marker_size: int = 4, 
        color_polyhedron: str ="green", 
        **kwargs
    ) -> go.Figure:
    """
    3D plot of the datapoints in X and the 3D polytope enclosed by the archetypes in Z.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    Z : np.ndarray
        Input archetype matrix of shape (n_samples, n_features).
    color_vec : np.ndarray (optional)
        Values for coloring the datapoints from X.
    marker_size: int (optional)
        Size of the dots in the scatterplot from data X
    color_polyhedron: str
        Color of the polyhedron from the archetypes Z

    Returns:
    --------
    fig: go.Figuret
        3D plot of X and polytope enclosed by Z
    """

    if X.shape[1] < 3 or Z.shape[1] < 3:
        raise ValueError("Both X and Z must have at least 3 columns (PCs).")

    X_plot, Z_plot = X[:, :3], Z[:, :3]

    plot_df = pd.DataFrame(X_plot, columns=["x0", "x1", "x2"])
    plot_df["marker_size"] = np.repeat(marker_size, X_plot.shape[0])

    kwargs.pop("color_polyhedron", None)

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
            size_max=kwargs.pop("size_max", 10),
            **kwargs,
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
            size_max=kwargs.pop("size_max", 10),
            **kwargs,
        )

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
            opacity=0.1
        )
    )

    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])
        fig.add_trace(
            go.Scatter3d(
                x=Z_plot[simplex, 0],
                y=Z_plot[simplex, 1],
                z=Z_plot[simplex, 2],
                mode="lines",
                line=dict(color=color_polyhedron, width=4),
                showlegend=False,
            )
        )

    fig.update_layout(template="none")
    return fig

def align_archetypes(
    ref_arch: np.ndarray, 
    query_arch: np.ndarray
    ) -> np.ndarray:
    """
    Arranges the archetypes of the query after the order of archetypes in the reference
    Parameters:
    -----------
    ref_arch: np.ndarray
        Reference archtypes of shape (n_samples, n_features).
    query_arch: np.ndarray
        Query archtypes of shape (n_samples, n_features).

    Returns:
    --------
    query_arch: np.ndarray
        Sorted query archetypes
    """
    euclidean_d = cdist(ref_arch, query_arch.copy(), metric="euclidean")
    ref_idx, query_idx = linear_sum_assignment(euclidean_d)
    return query_arch[query_idx, :]

# def bootstrap_variance_k_arr(X, n_bootstrap, k_arr, delta=0, seed=42, **kwargs):
#     assert k_arr.min() > 1
#     bootstrap_var = np.array(
#         [
#             bootstrap_variance_single_k(
#                 X, n_bootstrap=n_bootstrap, k=k, delta=delta, seed=seed, **kwargs
#             )
#             for k in k_arr
#         ]
#     )
#     plot_df = pd.DataFrame({"k": k_arr, "var": bootstrap_var})
#     p = (
#         pn.ggplot(plot_df, pn.aes(x="k", y="var"))
#         + pn.geom_point(color="blue")
#         + pn.geom_line(color="blue")
#         + pn.labs(x="Number of Archetypes", y="Mean Variance in Archetype Position")
#     )
#     return p


# t-ratio
# def simplex_volume(simplex_points):
#     pivot_point = simplex_points[0, :]
#     k = len(pivot_point)
#     return np.abs(np.linalg.det((simplex_points - pivot_point[None, ])[1:, :])) / math.factorial(k)

# def compute_t_ratio(X, Z):
#     D, k = X.shape[1], Z.shape[0] # number of PCs, number of archetypes
#     if k < D + 1:
#         convhull_volume = ConvexHull(project_on_polytope(X, Z)[0].T).volume
#         polytope_volume = ConvexHull(project_on_polytope(Z, Z)[0].T).volume
#     elif k == D + 1:
#         convhull_volume = ConvexHull(X.T).volume
#         polytope_volume = simplex_volume(Z.T)
#     elif k > D + 1:
#         convhull_volume = ConvexHull(X.T).volume
#         polytope_volume = ConvexHull(Z.T).volume
#     return polytope_volume / convhull_volume


# def project_on_polytope(X, Z):
#     D, k = X.shape[0], Z.shape[1]
#     assert k < (D + 1) and k > 1
#     if k == 2:
#         proj_vec = (Z[:, 1] - Z[:, 0])[:, None]
#     else:
#         proj_vec = (Z.T - Z[:, 0])[1:].T
#     proj_coord = (
#         np.linalg.inv(proj_vec.T @ proj_vec) @ proj_vec.T @ (X - Z[:, 0][:, None])
#     )
#     proj_X = proj_vec @ proj_coord + Z[:, 0][:, None]
#     # proj_mtx = proj_vec@np.linalg.inv(proj_vec.T@proj_vec)@proj_vec.T
#     return proj_coord, proj_X



# def compute_t_ratio_vitali(X, Z):
#     # adapted from: https://github.com/vitkl/ParetoTI/blob/510990630da589101c6a8313571c96f7544879da/R/fit_pch.R#L247
#     # NOTE: I am not fully convinced that this makes sense, since we do not consider all dimensions, but only (k-1) dimensions.
#     # This is especially harmful if the dimensions are not ordered by variance explained
#     k = Z.shape[1]
#     convhull_volume = ConvexHull(X[0 : (k - 1), :].T).volume
#     polytope_volume = simplex_volume(Z[0 : (k - 1), :].T)
#     return polytope_volume / convhull_volume

# Appendix

# not sure if this ratio of archeytpe over data variance is useful in any way
# def compute_var_ratio_vitali(X, Z):
#    # adapted from: https://github.com/vitkl/ParetoTI/blob/510990630da589101c6a8313571c96f7544879da/R/fit_pch.R#L1178
#    data_var = X.var(axis=1)
#    arch_var = Z.var(axis=1)
#    return arch_var / data_var

# see https://github.com/vitkl/ParetoTI/blob/510990630da589101c6a8313571c96f7544879da/R/fit_pch.R#L257
# archetypes = Z.A.T
# print(archetypes.shape)
#
## create random matrix where each row sums to 1
# n_additional = 100
# np.random.seed(42)
# rand_mtx = np.random.rand(n_additional, archetypes.shape[0])
# rand_mtx /= rand_mtx.sum(axis=1)[:, None]
#
# additional_archetypes = (rand_mtx @ archetypes)
# print(additional_archetypes.shape)
#
# stacked_archetypes = np.row_stack([archetypes, additional_archetypes])
# print(stacked_archetypes.shape)
#
## https://github.com/vitkl/ParetoTI/blob/510990630da589101c6a8313571c96f7544879da/R/fit_pch.R#L879C39-L879C46
## Vitali used the "FA" argument, but this is turned on by default I think in the scipy version (FA - report total area and volume),
## see http://www.qhull.org/html/qhull.htm
# convhull_archetypes = ConvexHull(stacked_archetypes, qhull_options="QJ")
# print(convhull_archetypes.volume)
