import types
import warnings

import numpy as np
import pandas as pd
import partipy as pt
import plotly.graph_objects as go
import plotnine as pn
import pytest
import scanpy as sc
from partipy.plotting import (
    barplot_enrichment_comparison,
    barplot_functional_enrichment,
    barplot_meta_enrichment,
    heatmap_meta_enrichment,
    plot_2D,
    plot_3D,
    plot_archetypes_2D,
    plot_archetypes_3D,
    plot_bootstrap_2D,
    plot_bootstrap_3D,
    plot_bootstrap_multiple_k,
    plot_IC,
    plot_var_explained,
    radarplot_meta_enrichment,
)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="plotly")


def _mock_adata():
    adata = sc.AnnData(X=np.random.rand(1000, 50))
    sc.pp.pca(adata)
    pt.set_obsm(adata, "X_pca", 4)
    pt.bootstrap_aa(adata=adata, n_bootstrap=10, n_archetypes=3)
    pt.bootstrap_aa_multiple_k(adata=adata, n_bootstrap=10, n_archetypes_list=range(3, 4))
    pt.compute_archetypes(adata, n_archetypes=4)
    return adata


def _mock_enrich_table():
    df = pd.DataFrame(
        {
            "Process": [f"Process_{i}" for i in range(10)],
            **{int(i): np.random.rand(10) for i in range(4)},
            "specificity": np.random.rand(10),
        }
    )
    return df


def _mock_meta_enrich_table():
    df = pd.DataFrame(
        {
            "Meta1": np.random.rand(10),
            "Meta2": np.random.rand(10),
            "Meta3": np.random.rand(10),
        }
    )
    df = df.div(df.sum(axis=1), axis=0)
    return df


mock_adata = _mock_adata()
mock_enrich_table = {0: _mock_enrich_table(), 1: _mock_enrich_table()}
mock_meta_enrich_table = _mock_meta_enrich_table()

### plot_2D ###


@pytest.mark.github_actions
def test_plot_2D(mock_adata=mock_adata):
    p = plot_2D(mock_adata.X, mock_adata.uns["AA_results"]["Z"])
    assert isinstance(p, pn.ggplot), "Expected a plotnine ggplot object"


### plot_3D ###


@pytest.mark.github_actions
def test_plot_3D(mock_adata=mock_adata):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        p = plot_3D(mock_adata.X, mock_adata.uns["AA_results"]["Z"])
        assert isinstance(p, go.Figure), "Expected a plotly graph_objects Figure"


### plot_archetypes_2D ###


@pytest.mark.github_actions
def test_plot_archetypes_2D(mock_adata=mock_adata):
    p = plot_archetypes_2D(mock_adata)
    assert isinstance(p, pn.ggplot), "Expected a plotnine ggplot object"


### plot_archetypes_3D ###


@pytest.mark.github_actions
def test_plot_archetypes_3D(mock_adata=mock_adata):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        p = plot_archetypes_3D(mock_adata)
        assert isinstance(p, go.Figure), "Expected a plotly graph_objects Figure"


### plot_bootstrap_2D ###


@pytest.mark.github_actions
def test_plot_bootstrap_2D(mock_adata=mock_adata):
    p = plot_bootstrap_2D(mock_adata)
    assert isinstance(p, pn.ggplot), "Expected a plotnine ggplot object"


### plot_bootstrap_3D ###


@pytest.mark.github_actions
def test_plot_bootstrap_3D(mock_adata=mock_adata):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        p = plot_bootstrap_3D(mock_adata)
        assert isinstance(p, go.Figure), "Expected a plotly graph_objects Figure"


### plot_bootstrap_multiple_k ###


@pytest.mark.github_actions
def test_plot_bootstrap_multiple_k(mock_adata=mock_adata):
    p = plot_bootstrap_multiple_k(mock_adata)
    assert isinstance(p, pn.ggplot), "Expected a plotnine ggplot object"


### plot_IC ###


@pytest.mark.github_actions
def test_plot_IC(mock_adata=mock_adata):
    p = plot_IC(mock_adata)
    assert isinstance(p, pn.ggplot), "Expected a plotnine ggplot object"


### plot_var_explained ###


@pytest.mark.github_actions
def test_plot_var_explained(mock_adata=mock_adata):
    p = plot_var_explained(mock_adata)
    assert isinstance(p, pn.ggplot), "Expected a plotnine ggplot object"


### barplot_enrichment_comparison ###


@pytest.mark.github_actions
def test_barplot_enrichment_comparison(mock_enrich_table=mock_enrich_table):
    p = barplot_enrichment_comparison(mock_enrich_table[0])
    assert isinstance(p, pn.ggplot), "Expected a plotnine ggplot object"


### barplot_functional_enrichment ###


@pytest.mark.github_actions
def test_barplot_functional_enrichment(mock_enrich_table=mock_enrich_table):
    p = barplot_functional_enrichment(mock_enrich_table, show=False)
    assert p is not None, "Object is None"


### barplot_meta_enrichment ###


@pytest.mark.github_actions
def test_barplot_meta_enrichment(mock_meta_enrich_table=mock_meta_enrich_table):
    p = barplot_meta_enrichment(mock_meta_enrich_table)
    assert isinstance(p, pn.ggplot), "Expected a plotnine ggplot object"


### heatmap_meta_enrichment ###


@pytest.mark.github_actions
def test_heatmap_meta_enrichment(mock_meta_enrich_table=mock_meta_enrich_table):
    p = heatmap_meta_enrichment(mock_meta_enrich_table)
    assert isinstance(p, pn.ggplot), "Expected a plotnine ggplot object"


### radarplot_meta_enrichment ###


@pytest.mark.github_actions
def test_radarplot_meta_enrichment(mock_meta_enrich_table=mock_meta_enrich_table):
    p = radarplot_meta_enrichment(mock_meta_enrich_table)
    assert isinstance(p, types.ModuleType), "Expected a ModuleType"
