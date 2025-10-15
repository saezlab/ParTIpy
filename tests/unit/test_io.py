from typing import Any

import anndata as ad
import numpy as np
import pytest
from partipy.io import ensure_archetype_config_keys, read_h5ad, serialize_archetype_caches, write_h5ad
from partipy.schema import (
    DEFAULT_INIT,
    DEFAULT_MAX_ITER,
    DEFAULT_OPTIM,
    DEFAULT_REL_TOL,
    DEFAULT_WEIGHT,
    ArchetypeConfig,
)

CONFIG_PREFIX = "ArchetypeConfig::"


def _make_config(n_archetypes: int = 3) -> ArchetypeConfig:
    return ArchetypeConfig(
        obsm_key="X_pca",
        n_dimensions=(0, 1),
        n_archetypes=n_archetypes,
        init=DEFAULT_INIT,
        optim=DEFAULT_OPTIM,
        weight=DEFAULT_WEIGHT,
        max_iter=DEFAULT_MAX_ITER,
        rel_tol=DEFAULT_REL_TOL,
        early_stopping=True,
        coreset_algorithm=None,
        coreset_fraction=0.1,
        coreset_size=None,
        delta=0.0,
        seed=42,
        optim_kwargs={},
    )


def test_serialize_and_restore_archetype_caches() -> None:
    adata = ad.AnnData(np.ones((3, 2)))
    cfg = _make_config()
    adata.uns["AA_results"] = {cfg: {"Z": np.ones((3, 2))}}

    originals = serialize_archetype_caches(adata)

    assert "AA_results" in originals
    serialized = adata.uns["AA_results"]
    (key,) = serialized.keys()
    assert isinstance(key, str)
    assert key.startswith("ArchetypeConfig::")

    ensure_archetype_config_keys(adata)
    ensured = adata.uns["AA_results"]
    (restored_key,) = ensured.keys()
    assert isinstance(restored_key, ArchetypeConfig)
    assert ensured[restored_key]["Z"].shape == (3, 2)


@pytest.mark.parametrize(
    "uns_key", ["AA_selection_metrics", "AA_cell_weights", "AA_bootstrap", "t_ratio", "AA_permutation"]
)
def test_ensure_archetype_config_keys_handles_multiple_stores(uns_key: str) -> None:
    adata = ad.AnnData(np.ones((2, 2)))
    cfg = _make_config(4)
    value: float | dict[str, Any]
    if uns_key == "t_ratio":
        value = 0.5
    elif uns_key == "AA_permutation":
        value = {"t_ratio": np.array([0.1]), "rss": np.array([0.2])}
    else:
        value = {"payload": 1}
    adata.uns[uns_key] = {cfg: value}
    serialize_archetype_caches(adata, uns_keys=(uns_key,))
    ensure_archetype_config_keys(adata, uns_keys=(uns_key,))
    ensured = adata.uns[uns_key]
    (key,) = ensured.keys()
    assert isinstance(key, ArchetypeConfig)
    restored = ensured[key]
    if uns_key == "t_ratio":
        assert restored == pytest.approx(0.5)
    elif uns_key == "AA_permutation":
        np.testing.assert_array_equal(restored["t_ratio"], np.array([0.1]))
        np.testing.assert_array_equal(restored["rss"], np.array([0.2]))
    else:
        assert restored["payload"] == 1


def test_write_and_read_h5ad_roundtrip(tmp_path) -> None:
    adata = ad.AnnData(np.arange(6).reshape(3, 2).astype(float))
    adata.obsm["X_pca"] = np.arange(6).reshape(3, 2).astype(float)
    cfg = _make_config()
    adata.uns["AA_results"] = {cfg: {"Z": np.ones((3, 2))}}

    path = tmp_path / "roundtrip.h5ad"
    write_h5ad(adata, path)

    # In-memory object should have been restored after writing
    (in_memory_key,) = adata.uns["AA_results"].keys()
    assert isinstance(in_memory_key, ArchetypeConfig)

    reloaded = read_h5ad(path)
    (reloaded_key,) = reloaded.uns["AA_results"].keys()
    assert isinstance(reloaded_key, ArchetypeConfig)
    np.testing.assert_array_equal(reloaded.uns["AA_results"][reloaded_key]["Z"], np.ones((3, 2)))


def test_ensure_handles_legacy_string_keys() -> None:
    adata = ad.AnnData(np.ones((1, 1)))
    cfg = _make_config()
    legacy_key = f"{CONFIG_PREFIX}{cfg.model_dump_json()}"
    adata.uns["AA_results"] = {legacy_key: {"Z": np.ones((3, 2))}}

    ensure_archetype_config_keys(adata, uns_keys=("AA_results",))
    (restored_key,) = adata.uns["AA_results"].keys()
    assert isinstance(restored_key, ArchetypeConfig)
