import anndata
import numpy as np
import pandas as pd
import plotnine as pn
from sklearn.decomposition import PCA
from tqdm import tqdm


def compute_shuffled_pca(
    adata: anndata.AnnData,
    layer_key: None | str = None,
    n_components: int = 50,
    n_shuffle: int = 50,
    seed: int = 42,
    save_to_anndata: bool = True,
    **pca_kwargs,
) -> None | pd.DataFrame:
    """
    Compute PCA on the data and compare it to shuffled data to determine significant components.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing the data to analyze.
    layer_key : str, default `None`
        Key for the layer in adata to use for PCA. If None, uses adata.X.
    n_components : int, default `50`
        Number of PCA components to compute.
    n_shuffle : int, default `50`
        Number of times to shuffle the data for comparison.
    seed : int, default `42`
        Random seed for reproducibility.
    save_to_anndata : bool, default `True`
        If True, save the results to adata.uns["AA_pca"]. If False, return a DataFrame.
    **pca_kwargs : keyword arguments
        Additional keyword arguments to pass to the PCA constructor.

    Returns
    -------
    None or pd.DataFrame
        If save_to_anndata is True, saves the results to adata.uns["AA_pca"]. Otherwise, returns a DataFrame with the results.
    """
    if layer_key is None:
        X = adata.X.copy()
    else:
        X = adata.layers[layer_key].copy()
    X -= X.mean(axis=0, keepdims=True)
    n_samples, n_features = X.shape
    # total_variance = np.sum(X**2) / n_samples # same for X_perm, since permutation corresponds to mulitplication with orthogonal matrix
    pca_unshuffled = PCA(n_components=n_components, **pca_kwargs).fit(X.copy())
    variance_unshuffled = pca_unshuffled.explained_variance_

    variance_shuffled = np.zeros((n_shuffle, n_components), dtype=np.float32)

    rng = np.random.default_rng(seed)
    seeds = rng.choice(a=int(1e9), size=n_shuffle, replace=False)

    pbar = tqdm(iterable=enumerate(seeds), total=n_shuffle)
    for shuffle_iter, seed in pbar:
        rng_inner = np.random.default_rng(seed=seed)
        X_perm = np.column_stack([rng_inner.permutation(X[:, col_idx]) for col_idx in range(n_features)])
        pca_shuffled = PCA(n_components=n_components, **pca_kwargs).fit(X_perm.copy())
        variance_shuffled[shuffle_iter, :] = pca_shuffled.explained_variance_

    variance_shuffled_mean = variance_shuffled.mean(axis=0)
    variance_shuffled_std = variance_shuffled.mean(axis=0)

    df = pd.DataFrame(
        {
            "component": np.arange(n_components),
            "variance_unshuffled": variance_unshuffled,
            "variance_shuffled_mean": variance_shuffled_mean,
            "variance_shuffled_std": variance_shuffled_std,
            "variance_shuffled_mean+std": variance_shuffled_mean + variance_shuffled_std,
        }
    )
    df["included"] = df["variance_unshuffled"] > df["variance_shuffled_mean+std"]

    if save_to_anndata:
        adata.uns["AA_pca"] = df
        return None
    else:
        return df


def plot_shuffled_pca(adata: anndata.AnnData) -> pn.ggplot:
    """
    Plot the results of the shuffled PCA analysis stored in adata.uns["AA_pca"].

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing the shuffled PCA results in adata.uns["AA_pca"].

    Returns
    -------
    p : plotnine.ggplot
        A ggplot object visualizing the variance explained by the PCA components.
    """
    if "AA_pca" not in adata.uns:
        raise ValueError("No shuffled PCA results found in adata.uns['AA_pca']. Please run compute_shuffled_pca first.")
    df = adata.uns["AA_pca"]
    max_component = df.query("included")["component"].max()
    p = (
        pn.ggplot(df)
        + pn.geom_vline(xintercept=max_component, linetype="dashed", color="grey", alpha=0.75, size=1.0)
        + pn.geom_line(pn.aes(x="component", y="variance_shuffled_mean+std", color='"Shuffled"'), linetype="solid")
        + pn.geom_point(pn.aes(x="component", y="variance_shuffled_mean+std", color='"Shuffled"'), size=1.5)
        + pn.geom_point(pn.aes(x="component", y="variance_unshuffled", color='"Unshuffled"'), size=1.5)
        + pn.geom_line(pn.aes(x="component", y="variance_unshuffled", color='"Unshuffled"'), size=0.5)
        + pn.scale_color_manual(values={"Unshuffled": "black", "Shuffled": "grey"}, name="Data Type")
        + pn.labs(x="PC Component", y="Variance Explained", title=f"0-{max_component} PC Components above Noise")
        + pn.theme(legend_position="right")
    )
    return p
