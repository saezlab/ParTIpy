import math

import numpy as np
import plotnine as pn
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment

from .arch import AA


def hvg_vst_method(adata):
    from scipy.sparse import issparse
    from scipy.interpolate import UnivariateSpline

    if "std_variance" in adata.var.columns and "std_variance_rank" in adata.var.columns:
        print("It seems like hvg_vst has been used before; aborting")
        return None
    df = pd.DataFrame(index=adata.var_names)
    df["mean"] = adata.X.mean(axis=0).A1 if issparse(adata.X) else adata.X.mean(axis=0)
    df = df.loc[df["mean"] > 0, :].copy()
    adata_subset = adata[:, df.index]
    df["variance"] = (
        adata_subset.X.todense().var(axis=0).A1
        if issparse(adata_subset.X)
        else adata_subset.X.var(axis=0)
    )
    df["dispersion"] = df["variance"] / df["mean"]
    df["log10_mean"] = np.log10(df["mean"])
    df["log10_variance"] = np.log10(df["variance"])
    df = df[np.isfinite(df["variance"])]
    df = df[df["variance"] > 0]

    x, y = df["log10_mean"].values, df["log10_variance"].values
    order = np.argsort(x)
    x, y = x[order], y[order]
    cs = UnivariateSpline(x, y)

    df["log10_predicted_variance"] = cs(df["log10_mean"])
    df["predicted_variance"] = 10 ** df["log10_predicted_variance"]

    tmp_mtx = (
        adata_subset.X.copy().T.todense().astype(np.float64)
        if issparse(adata_subset.X)
        else adata_subset.X.copy().T.astype(np.float64)
    )
    tmp_mtx -= df["mean"].to_numpy()[:, None]
    tmp_mtx /= (df["predicted_variance"].to_numpy() ** 0.5)[:, None]
    clip = adata_subset.shape[0] ** 0.5
    tmp_mtx[tmp_mtx > clip] = clip
    df["std_variance"] = tmp_mtx.std(axis=1)
    adata.var = adata_subset.var.join(df, how="left").copy()
    adata.var["std_variance_rank"] = (
        (adata.var["std_variance"] * -1).argsort().argsort()
    )


def get_var_explained_pca(X, random_state=42, max_comp=np.inf):
    from sklearn.decomposition import PCA

    pca = PCA(random_state=random_state)
    pca.fit(X)
    mean_vec = X.mean(axis=0)
    total_variance = np.sum((X - mean_vec) ** 2) / X.shape[0]
    assert np.isclose(
        total_variance, np.sum((X.shape[0] - 1) / X.shape[0] * pca.explained_variance_)
    )
    # note that n_samples - 1 degrees of freedom is used, which I don't like
    pca.explained_variance_ = (X.shape[0] - 1) / X.shape[0] * pca.explained_variance_
    plot_df = pd.DataFrame(
        {
            "component": np.arange(1, len(pca.explained_variance_) + 1),
            "Variance Explained": pca.explained_variance_,
            "Fraction of Variance Explained": (
                pca.explained_variance_ / total_variance
            ),
            "Cumulative Fraction of Variance Explained": np.cumsum(
                pca.explained_variance_
            )
            / total_variance,
        }
    )
    plot_df = plot_df.melt(id_vars="component", var_name="var_type", value_name="value")
    plot_df = plot_df.query("var_type != 'Variance Explained'").copy()
    plot_df["var_type"] = pd.Categorical(
        plot_df["var_type"],
        categories=[
            "Fraction of Variance Explained",
            "Cumulative Fraction of Variance Explained",
        ],
    )
    plot_df = plot_df.loc[plot_df["component"] < max_comp, :].copy()
    p = (
        pn.ggplot(plot_df)
        + pn.geom_point(pn.aes(x="component", y="value"), color="blue")
        + pn.labs(x="Component", y="")
        + pn.facet_wrap("~var_type", scales="free_y", ncol=1)
        + pn.geom_hline(yintercept=0, color="grey")
    )
    return p


def get_var_explained_aa(X, k_arr=np.arange(2, 10)):
    # see also Supplementary Figure 3 in "Inferring biological tasks using Pareto analysis of high-dimensional data"
    # see also Vitali's code: https://github.com/vitkl/ParetoTI/blob/510990630da589101c6a8313571c96f7544879da/R/plot_arc_var.R#L31
    results = {}
    for k in k_arr:
        A, B, Z, RSS, varexpl = AA(n_archetypes=k).fit(X).return_all()
        results[k] = {"Z": Z, "A": A, "B": B, "RSS": RSS, "varexpl": varexpl}
    varexpl = np.array([results[k]["varexpl"] for k in k_arr])
    plot_df = pd.DataFrame(
        {
            "k": k_arr,
            "varexpl": np.array([results[k]["varexpl"] for k in k_arr]),
            "varexpl_ontop": np.concatenate(
                (varexpl[0][None], varexpl[1:] - varexpl[:-1])
            ),
        }
    )
    diag_df = pd.concat([plot_df.head(1), plot_df.tail(1)])
    diag_df.loc[diag_df["k"] == 1, "varexpl"] = 0

    # note the projection won't look orthogonal on the plot, because the scale of the axis is so different
    # just set pn.coord_equal() to see it
    offset_vec = plot_df.values[:, :2][0,]
    proj_vec = (plot_df.values[:, :2] - offset_vec)[-1, :][:, None]
    proj_mtx = proj_vec @ np.linalg.inv(proj_vec.T @ proj_vec) @ proj_vec.T
    proj_val = (
        proj_mtx @ (plot_df.values[:, :2] - plot_df.values[:, :2][0,]).T
    ).T + offset_vec
    proj_df = pd.DataFrame(proj_val, columns=["k", "varexpl"])
    dist = ((plot_df.values[:, :2] - proj_df.values[:, :2]) ** 2).sum(axis=1) ** 0.5
    dist_df = pd.DataFrame({"k": k_arr, "dist": dist})

    p1 = (
        pn.ggplot(plot_df)
        + pn.geom_line(data=plot_df, mapping=pn.aes(x="k", y="varexpl"), color="blue")
        + pn.geom_point(data=plot_df, mapping=pn.aes(x="k", y="varexpl"), color="blue")
        + pn.geom_point(data=proj_df, mapping=pn.aes(x="k", y="varexpl"), color="white")
        + pn.geom_line(data=diag_df, mapping=pn.aes(x="k", y="varexpl"), color="white")
        + pn.labs(x="k (Number of Archetypes)", y="Variance Explained")
        + pn.lims(y=[0, 1])
        + pn.scale_x_continuous(breaks=np.arange(k_arr.min(), k_arr.max() + 1))
    )

    p2 = (
        pn.ggplot()
        + pn.geom_point(data=dist_df, mapping=pn.aes(x="k", y="dist"), color="blue")
        + pn.geom_segment(
            data=dist_df,
            mapping=pn.aes(x="k", y=0, xend="k", yend="dist"),
            color="blue",
        )
        + pn.scale_x_continuous(breaks=np.arange(k_arr.min(), k_arr.max() + 1))
        + pn.labs(x="Component (Number of Archetypes)", y="Distance to Projected Point")
    )

    p3 = (
        pn.ggplot(plot_df)
        + pn.geom_point(pn.aes(x="k", y="varexpl_ontop"), color="grey")
        + pn.geom_line(pn.aes(x="k", y="varexpl_ontop"), color="grey")
        + pn.labs(
            x="k (Number of Archetypes)", y="Variance Explained on Top of (k-1) Model"
        )
        + pn.scale_x_continuous(breaks=np.arange(k_arr.min(), k_arr.max() + 1))
        + pn.lims(y=(0, None))
    )

    return p1, p2, p3


def bootstrap_variance_single_k(X, n_bootstrap, k, seed=42, **kwargs):
    # see Vitali's function here: ?

    # compute the reference archetypes on the full dataset
    ref_Z = AA(n_archetypes=k).fit(X).Z
    n_samples, n_features = X.shape

    # compute archetypes on bootstrap samples
    Z_stack = []
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        idx_bootstrap = rng.choice(np.arange(n_samples), size=n_samples, replace=True)
        X_bootstrap = X[idx_bootstrap, :].copy()
        Z_stack.append(AA(n_archetypes=k).fit(X_bootstrap).Z)

    # align the archetypes to the reference archetype
    Z_stack = np.stack(
        [align_archetypes(ref_arch=ref_Z, query_arch=query_Z) for query_Z in Z_stack]
    )

    # stack the results from each bootstrap sample and compute the variance across all bootstrap samples
    # resultings in [features x archetypes]
    var_per_feature_per_arch = Z_stack.var(axis=0)

    # compute mean per feature and mean per archetype
    # var_per_feature = var_per_feature_per_arch.mean(axis=0)
    var_per_archetype = var_per_feature_per_arch.mean(axis=1)

    # normalize the mean per feature based on the total features variance in the data
    # TODO: not sure why one should do that?
    # total_feature_var = X.var(axis=0).sum()
    # var_per_feature_norm = var_per_feature / total_feature_var

    return var_per_archetype.mean()


def bootstrap_variance_k_arr(X, n_bootstrap, k_arr, delta=0, seed=42, **kwargs):
    assert k_arr.min() > 1
    bootstrap_var = np.array(
        [
            bootstrap_variance_single_k(
                X, n_bootstrap=n_bootstrap, k=k, delta=delta, seed=seed, **kwargs
            )
            for k in k_arr
        ]
    )
    plot_df = pd.DataFrame({"k": k_arr, "var": bootstrap_var})
    p = (
        pn.ggplot(plot_df, pn.aes(x="k", y="var"))
        + pn.geom_point(color="blue")
        + pn.geom_line(color="blue")
        + pn.labs(x="Number of Archetypes", y="Mean Variance in Archetype Position")
    )
    return p


def compute_dist_mtx(mtx_1, mtx_2):
    AB = np.dot(mtx_1, mtx_2.T)
    AA = np.sum(np.square(mtx_1), axis=1)
    BB = np.sum(np.square(mtx_2), axis=1)
    dist_mtx = (BB - 2 * AB).T + AA
    dist_mtx[np.isclose(dist_mtx, 0)] = (
        0  # avoid problems if we get small negative numbers due to numerical inaccuracies
    )
    dist_mtx = np.sqrt(dist_mtx)
    return dist_mtx


def align_archetypes(ref_arch, query_arch):
    # not sure if copy here is needed, compute_dist_mtx should not modify the matrices
    euclidean_d = compute_dist_mtx(ref_arch, query_arch.copy()).T
    ref_idx, query_idx = linear_sum_assignment(euclidean_d)
    return query_arch[query_idx, :]


def simplex_volume(simplex_points):
    pivot_point = simplex_points[0, :]
    k = len(pivot_point)
    return np.abs(
        np.linalg.det((simplex_points - pivot_point[None,])[1:, :])
    ) / math.factorial(k)


def compute_t_ratio_vitali(X, Z):
    # adapted from: https://github.com/vitkl/ParetoTI/blob/510990630da589101c6a8313571c96f7544879da/R/fit_pch.R#L247
    # NOTE: I am not fully convinced that this makes sense, since we do not consider all dimensions, but only (k-1) dimensions.
    # This is especially harmful if the dimensions are not ordered by variance explained
    k = Z.shape[1]
    convhull_volume = ConvexHull(X[0 : (k - 1), :].T).volume
    polytope_volume = simplex_volume(Z[0 : (k - 1), :].T)
    return polytope_volume / convhull_volume


def compute_t_ratio(X, Z):
    D, k = X.shape[0], Z.shape[1]  # number of PCs, number of archetypes
    if k < D + 1:
        convhull_volume = ConvexHull(project_on_polytope(X.T, Z.T)[0].T).volume
        polytope_volume = ConvexHull(project_on_polytope(Z.T, Z.T)[0].T).volume
    elif k == D + 1:
        convhull_volume = ConvexHull(X.T).volume
        polytope_volume = simplex_volume(Z.T)
    elif k > D + 1:
        convhull_volume = ConvexHull(X.T).volume
        polytope_volume = ConvexHull(Z.T).volume
    return polytope_volume / convhull_volume


def project_on_polytope(X, Z):
    D, k = X.shape[0], Z.shape[1]
    assert k < (D + 1) and k > 1
    if k == 2:
        proj_vec = (Z[:, 1] - Z[:, 0])[:, None]
    else:
        proj_vec = (Z.T - Z[:, 0])[1:].T
    proj_coord = (
        np.linalg.inv(proj_vec.T @ proj_vec) @ proj_vec.T @ (X - Z[:, 0][:, None])
    )
    proj_X = proj_vec @ proj_coord + +Z[:, 0][:, None]
    # proj_mtx = proj_vec@np.linalg.inv(proj_vec.T@proj_vec)@proj_vec.T
    return proj_coord, proj_X


def plot_2D(X, Z, color_vec, opacity=0.4):
    import plotly.graph_objects as go
    import plotly.express as px

    D, k = X.shape[0], Z.shape[1]
    if D != 2:
        raise ValueError("D should be 2")
    if k not in [2, 3, 4]:
        raise ValueError("k should be in [2, 3, 4]")
    plot_df = pd.DataFrame(X.T, columns=[f"x{i}" for i in range(X.T.shape[1])])
    plot_df["marker_size"] = np.repeat(4, X.shape[1])
    if color_vec is not None:
        plot_df["color"] = color_vec
        fig = px.scatter(
            plot_df, x="x0", y="x1", color="color", size="marker_size", size_max=10
        )
    else:
        fig = px.scatter(plot_df, x="x0", y="x1", size="marker_size", size_max=10)
    Z_plot = Z.copy()
    order = np.argsort(
        np.arctan2(
            Z_plot[1, :] - np.mean(Z_plot[1, :]), Z_plot[0, :] - np.mean(Z_plot[0, :])
        )
    )
    Z_plot = Z_plot[:, order]
    Z_plot = np.column_stack((Z_plot, Z_plot[:, 0]))
    fig.add_trace(
        go.Scatter(x=Z_plot[0,], y=Z_plot[1,], fill="toself", opacity=opacity)
    )

    fig.update_layout(template="plotly_dark")
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.show()


def plot_3D(X, Z, color_vec=None, opacity=0.2):
    import plotly.graph_objects as go
    import plotly.express as px

    X = X.T
    Z = Z.T
    D, k = X.shape[0], Z.shape[1]
    if D != 3:
        raise ValueError("D should be 3")
    if k not in [3, 4]:
        raise ValueError("k should be in [3, 4]")
    plot_df = pd.DataFrame(X.T, columns=[f"x{i}" for i in range(X.T.shape[1])])
    plot_df["marker_size"] = np.repeat(4, X.shape[1])
    if color_vec is not None:
        plot_df["color"] = color_vec
        fig = px.scatter_3d(
            plot_df,
            x="x0",
            y="x1",
            z="x2",
            color="color",
            size="marker_size",
            size_max=10,
        )
    else:
        fig = px.scatter_3d(
            plot_df, x="x0", y="x1", z="x2", size="marker_size", size_max=10
        )

    vertices = [[Z[0, i], Z[1, i], Z[2, i]] for i in range(k)]
    if k == 3:
        triangles = [[0, 1, 2]]
        fig.add_trace(
            go.Mesh3d(
                x=[vertex[0] for vertex in vertices],
                y=[vertex[1] for vertex in vertices],
                z=[vertex[2] for vertex in vertices],
                i=[triangle[0] for triangle in triangles],
                j=[triangle[1] for triangle in triangles],
                k=[triangle[2] for triangle in triangles],
                color="green",
                opacity=opacity,
            )
        )
    elif k == 4:
        tetrahedron = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        fig.add_trace(
            go.Mesh3d(
                x=[vertex[0] for vertex in vertices],
                y=[vertex[1] for vertex in vertices],
                z=[vertex[2] for vertex in vertices],
                i=[tet[0] for tet in tetrahedron],
                j=[tet[1] for tet in tetrahedron],
                k=[tet[2] for tet in tetrahedron],
                color="green",
                opacity=opacity,
            )
        )

    Z_plot = Z.copy()
    order = np.argsort(
        np.arctan2(
            Z_plot[1, :] - np.mean(Z_plot[1, :]), Z_plot[0, :] - np.mean(Z_plot[0, :])
        )
    )
    Z_plot = Z_plot[:, order]
    Z_plot = np.column_stack((Z_plot, Z_plot[:, 0]))
    fig.add_trace(
        go.Scatter3d(
            x=Z_plot[0, :],
            y=Z_plot[1, :],
            z=Z_plot[2, :],
            mode="lines",
            line=dict(color="green", width=4),
        )
    )
    fig.update_layout(template="plotly_dark")
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.show()


def bootstrap_PCHA(X, n_bootstrap, k, seed=42, **kwargs):
    # compute the reference archetypes on the full dataset
    ref_Z = AA(n_archetypes=k).fit(X).Z
    n_samples, n_features = X.shape

    # compute archetypes on bootstrap samples
    Z_list = []
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        idx_bootstrap = rng.choice(np.arange(n_samples), size=n_samples, replace=True)
        X_bootstrap = X[idx_bootstrap, :].copy()
        Z_list.append(AA(n_archetypes=k).fit(X_bootstrap).Z)

    # align the archetypes to the reference archetype
    Z_list = [
        align_archetypes(ref_arch=ref_Z.copy(), query_arch=query_Z.copy())
        for query_Z in Z_list
    ]

    bootstrap_df = []
    for idx in range(len(Z_list)):
        df = pd.DataFrame(Z_list[idx], columns=[f"pc_{i}" for i in range(n_features)])
        df["archetype"] = np.arange(k)
        df["iter"] = idx + 1  # iter=0 is reserved for the reference
        bootstrap_df.append(df)
    bootstrap_df = pd.concat(bootstrap_df)
    bootstrap_df["archetype"] = pd.Categorical(bootstrap_df["archetype"])

    df = pd.DataFrame(ref_Z, columns=[f"pc_{i}" for i in range(n_features)])
    df["archetype"] = np.arange(k)
    df["iter"] = 0

    bootstrap_df = pd.concat((bootstrap_df, df), axis=0)
    bootstrap_df["reference"] = bootstrap_df["iter"] == 0
    bootstrap_df["archetype"] = pd.Categorical(bootstrap_df["archetype"])

    # bootstrap_df["archetype_permuted"] = np.random.choice(bootstrap_df["archetype"], size=bootstrap_df.shape[0])
    # bootstrap_df["archetype_permuted"] = pd.Categorical(bootstrap_df["archetype_permuted"])
    return bootstrap_df


def plot_3D_bootstrap(
    bootstrap_df, color_var="archetype", opacity=0.5, iter_filter=None, jitter=0
):
    import plotly.express as px

    bootstrap_df["marker_size"] = np.repeat(4, bootstrap_df.shape[0])
    if iter_filter is not None:
        bootstrap_df = bootstrap_df.loc[
            np.isin(bootstrap_df["iter"], iter_filter), :
        ].copy()
    if jitter > 0:
        bootstrap_df[["pc_" + str(i) for i in range(3)]] += np.random.normal(
            loc=0,
            scale=jitter,
            size=bootstrap_df[["pc_" + str(i) for i in range(3)]].shape,
        )
    fig = px.scatter_3d(
        bootstrap_df,
        x="pc_0",
        y="pc_1",
        z="pc_2",
        color=color_var,
        size="marker_size",
        size_max=10,
        opacity=opacity,
        # add upon hover the iteration number
        hover_data=["iter", "archetype", "reference"],
    )
    fig.update_layout(template="plotly_dark")
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.show()


def plot_2D_bootstrap(
    bootstrap_df, color_var="archetype", opacity=0.5, iter_filter=None, jitter=0
):
    import plotly.express as px

    bootstrap_df["marker_size"] = np.repeat(1, bootstrap_df.shape[0])
    if iter_filter is not None:
        bootstrap_df = bootstrap_df.loc[
            np.isin(bootstrap_df["iter"], iter_filter), :
        ].copy()
    if jitter > 0:
        bootstrap_df[["pc_" + str(i) for i in range(3)]] += np.random.normal(
            loc=0,
            scale=jitter,
            size=bootstrap_df[["pc_" + str(i) for i in range(3)]].shape,
        )
    fig = px.scatter(
        bootstrap_df,
        x="pc_0",
        y="pc_1",
        color=color_var,
        size="marker_size",
        size_max=10,
        opacity=opacity,
        hover_data=["iter", "archetype", "reference"],
    )
    fig.update_layout(template="plotly_dark")
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.show()


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
