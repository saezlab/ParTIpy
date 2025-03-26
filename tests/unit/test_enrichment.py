import numpy as np
import pytest
import scanpy as sc
from partipy.enrichment import calculate_weights


def test_calculate_weights_anndata():
    """Test that it works correctly with sc.AnnData."""
    adata = sc.AnnData(np.random.rand(10, 5))
    adata.obsm["X_pca"] = np.random.rand(10, 5)
    adata.uns["archetypal_analysis"] = {"Z": np.random.rand(3, 5)}
    adata.uns["n_pcs"] = 5
    calculate_weights(adata)
    assert "cell_weights" in adata.obsm
    assert adata.obsm["cell_weights"].shape == (10, 3)


def test_calculate_weights_anndata_manual():
    """Test that it works correctly with sc.AnnData and manual mode."""
    adata = sc.AnnData(np.random.rand(10, 5))
    adata.obsm["X_pca"] = np.random.rand(10, 5)
    adata.uns["archetypal_analysis"] = {"Z": np.random.rand(3, 5)}
    adata.uns["n_pcs"] = 5
    length_scale = 1.0
    calculate_weights(adata, mode="manual", length_scale=length_scale)
    assert "cell_weights" in adata.obsm
    assert adata.obsm["cell_weights"].shape == (10, 3)


def test_calculate_weights_array():
    """Test that it works correctly with np.ndarray."""
    X = np.random.rand(10, 5)
    Z = np.random.rand(3, 5)
    weights = calculate_weights(X, Z)
    assert weights is not None
    assert weights.shape == (10, 3)


def test_calculate_weights_array_manual():
    """Test that it works correctly with np.ndarray and manual mode."""
    X = np.random.rand(10, 5)
    Z = np.random.rand(3, 5)
    length_scale = 1.0
    weights = calculate_weights(X, Z, mode="manual", length_scale=length_scale)
    assert weights is not None
    assert weights.shape == (10, 3)


def test_calculate_weights_missing_archetypes():
    """Test that it raises an error when archetypes are missing in the sc.AnnData object."""
    adata = sc.AnnData(np.random.rand(10, 5))
    adata.obsm["X_pca"] = np.random.rand(10, 5)
    with pytest.raises(ValueError):
        calculate_weights(adata)


def test_calculate_weights_missing_Z():
    """Test that it raises an error when archetypes are missing as input."""
    X = np.random.rand(10, 5)
    with pytest.raises(ValueError):
        calculate_weights(X)


## TODO ##
# Something to "rediscover" ground truth
##########

#### weighted_expr #######
