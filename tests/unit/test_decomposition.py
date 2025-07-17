import anndata
import numpy as np
import partipy as pt
import plotnine as pn
import pytest
from partipy.simulate import simulate_archetypes

np.random.seed(42)


def _simulate_adata(n_samples, n_dimensions, n_archetypes, seed: int = 42):
    X, A, Z = simulate_archetypes(
        n_samples=n_samples, n_archetypes=n_archetypes, n_dimensions=n_dimensions, noise_std=0.0, seed=seed
    )
    adata = anndata.AnnData(X)
    return adata


@pytest.mark.github_actions
@pytest.mark.parametrize("n_archetypes", list(range(2, 6)))
def test_that_shuffled_pca_identifies_correct_dimensions(
    n_archetypes: int,
) -> None:
    adata = _simulate_adata(n_samples=2_000, n_dimensions=50, n_archetypes=n_archetypes)
    pt.compute_shuffled_pca(adata, n_components=50)
    assert adata.uns["AA_pca"]["included"].sum() == n_archetypes - 1

    p = pt.plot_shuffled_pca(adata)
    assert isinstance(p, pn.ggplot), "Expected a plotnine ggplot object"
