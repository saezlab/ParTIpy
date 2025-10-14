from typing import Any

import anndata
import numpy as np
import pandas as pd
import partipy as pt
import pytest
import scanpy as sc
from partipy.paretoti import (
    _align_archetypes,
    _validate_aa_config,
    _validate_aa_results,
    compute_archetypes,
    compute_bootstrap_variance,
    compute_selection_metrics,
    set_obsm,
)
from partipy.simulate import simulate_archetypes


class _DummyConfig:
    """Lightweight stand-in for ArchetypeConfig used to mock cached entries."""

    def __init__(self, **fields):
        self._fields = dict(fields)
        for key, value in self._fields.items():
            setattr(self, key, value)

    def __hash__(self) -> int:
        # Freeze mapping fields for hashing; assume values are hashable or tuples/lists of hashables.
        frozen_items = []
        for key, value in self._fields.items():
            if isinstance(value, dict):
                frozen_items.append((key, tuple(sorted(value.items()))))
            elif isinstance(value, list):
                frozen_items.append((key, tuple(value)))
            else:
                frozen_items.append((key, value))
        return hash(tuple(sorted(frozen_items)))

    def _signature(self, *, ignore_fields=("n_archetypes",)) -> tuple[tuple[str, Any], ...]:
        ignore = set(ignore_fields)
        items = []
        for key, value in self._fields.items():
            if key in ignore:
                continue
            if isinstance(value, dict):
                items.append((key, tuple(sorted(value.items()))))
            elif isinstance(value, list):
                items.append((key, tuple(value)))
            else:
                items.append((key, value))
        return tuple(sorted(items))


def _make_dummy_config(n_archetypes: int, **overrides):
    base = {
        "obsm_key": "X_pca",
        "n_dimensions": (0,),
        "n_archetypes": n_archetypes,
        "init": "plus_plus",
        "optim": "projected_gradients",
        "weight": None,
        "max_iter": 500,
        "rel_tol": 1e-4,
        "early_stopping": True,
        "coreset_algorithm": None,
        "coreset_fraction": None,
        "coreset_size": None,
        "delta": 0.0,
        "seed": 42,
        "optim_kwargs": {},
    }
    base.update(overrides)
    return _DummyConfig(**base)


def _simulate_adata(n_samples, n_dimensions, n_archetypes, n_pcs, noise_std=0.0):
    X, A, Z = simulate_archetypes(
        n_samples=n_samples, n_archetypes=n_archetypes, n_dimensions=n_dimensions, noise_std=noise_std
    )
    adata = anndata.AnnData(X)
    adata.obsm["X_pca"] = sc.pp.pca(X, n_comps=n_pcs)
    adata.uns["AA_config"] = {
        "obsm_key": "X_pca",
        "n_dimensions": list(range(n_pcs)),
    }
    adata.uns["AA_results"] = {"Z": Z[:, :n_pcs]}
    return adata


@pytest.fixture
def _mock_adata():
    adata = anndata.AnnData(X=np.random.rand(1000, 50))
    sc.pp.pca(adata)
    pt.set_obsm(adata, "X_pca", 4)
    return adata


### _align_archetypes ###


@pytest.mark.github_actions
def test_align_archetypes_correct_alignment_same_archetypes():
    """Test if _align_archetypes correctly aligns identical archetypes.

    Verifies:
    - Archetypes are aligned with themselves.
    """
    ref = np.random.rand(10, 5)
    ref_aligned = _align_archetypes(ref, np.random.permutation(ref))
    assert np.array_equal(ref, ref_aligned), "Archetypes should be aligned correctly."


@pytest.mark.github_actions
def test_align_archetypes_correct_alignment_similar_archetypes():
    """Test if _align_archetypes correctly aligns similar archetypes.

    Verifies:
    - Archetypes are aligned with similar ones.
    """
    ref = np.random.rand(10, 5)
    similar_archetypes = ref + np.random.uniform(-0.01, 0.01, ref.shape)
    aligned_archetypes = _align_archetypes(ref, np.random.permutation(similar_archetypes))
    assert np.allclose(ref, aligned_archetypes, atol=0.01), "Archetypes should be aligned correctly."


### _validate_aa_config ###


@pytest.mark.github_actions
def test_validate_aa_config_input_validation(_mock_adata):
    """Test if input validation for _validate_aa_config works as intended.

    Verifies:
    - Raises ValueError when adata.uns['aa_config'] is not found.
    - Raises ValueError when adata.uns['aa_config'] is not a dictionary.
    - Raises ValueError when required keys are missing.
    - Raises ValueError when obsm_key is invalid.
    - Raises ValueError when n_dimension exceeds available dimensions.
    """
    adata = _mock_adata.copy()
    # Test invalid n_dimensions
    adata.uns["AA_config"]["n_dimensions"] = list(range(100))
    with pytest.raises(ValueError):
        _validate_aa_config(adata)

    # Test invalid obsm_key
    adata.uns["AA_config"]["obsm_key"] = "invalid_key"
    with pytest.raises(ValueError):
        _validate_aa_config(adata)

    # Test missing keys
    del adata.uns["AA_config"]["obsm_key"]
    del adata.uns["AA_config"]["n_dimensions"]
    with pytest.raises(ValueError):
        _validate_aa_config(adata)

    # Test missing aa_config
    del adata.uns["AA_config"]
    with pytest.raises(ValueError):
        _validate_aa_config(adata)


### _validate_aa_results ###


@pytest.mark.github_actions
def test_validate_aa_results_input_validation(_mock_adata):
    """Test if input validation for _validate_aa_results works as intended.

    Verifies:
    - Raises ValueError when adata.uns['AA_results'] is not found.
    """
    adata = _mock_adata.copy()
    # Test missing AA_results
    with pytest.raises(ValueError):
        _validate_aa_results(adata)


### bootstrap_aa ###


@pytest.mark.github_actions
def test_bootstrap_aa_output_correct_shape(_mock_adata):
    """Test if bootstrap_aa correctly updates adata.uns["AA_bootstrap"].

    Verifies:
    - Correct saving in AnnData object.
    - Correct types of AA_bootstrap.
    - Correct number of elements in AA_bootstrap.
    - Correct number and names of columns.
    - Correct number of rows.
    """
    adata = _mock_adata.copy()
    compute_bootstrap_variance(adata=adata, n_bootstrap=10, n_archetypes_list=[3, 4, 5])

    # Check existence of AA_bootstrap
    assert "AA_bootstrap" in adata.uns, "AA_bootstrap does not exist in adata.uns"

    # Check type of AA_bootstrap
    assert isinstance(adata.uns["AA_bootstrap"], dict), "AA_bootstrap is not a dictionary"

    # Check length of AA_bootstrap
    assert len(adata.uns["AA_bootstrap"]) == 3, "AA_bootstrap does not have the correct number of elements"

    # Check column names
    expected_columns = [
        "X_pca_0",
        "X_pca_1",
        "X_pca_2",
        "X_pca_3",
        "archetype",
        "iter",
        "reference",
        "mean_variance",
        "variance_per_archetype",
    ]
    bootstrap_df = pt.get_aa_bootstrap(adata, n_archetypes=3)
    assert all(col in bootstrap_df.columns for col in expected_columns), (
        "AA_bootstrap does not have the correct column names"
    )

    # Check shape
    assert bootstrap_df.shape == (33, len(expected_columns)), "AA_bootstrap does not have the correct shape"


@pytest.mark.github_actions
def test_bootstrap_aa_with_noisy_and_non_noisy_data():
    """Test the effect of noise on bootstrap results.

    Verifies:
    - Mean variance is higher for noisy data (adata03) compared to non-noisy data (adata0).
    - Maximum variance per archetype is higher for noisy data compared to non-noisy data.
    """
    adata0 = _simulate_adata(1000, 10, 5, 4, noise_std=0)
    adata03 = _simulate_adata(1000, 10, 5, 4, noise_std=0.3)

    compute_bootstrap_variance(adata0, n_bootstrap=10, n_archetypes_list=5)
    compute_bootstrap_variance(adata03, n_bootstrap=10, n_archetypes_list=5)

    # Compare mean variance
    mean_variance_adata0 = pt.get_aa_bootstrap(adata0, n_archetypes=5)["mean_variance"].mean()
    mean_variance_adata03 = pt.get_aa_bootstrap(adata03, n_archetypes=5)["mean_variance"].mean()
    assert mean_variance_adata0 < mean_variance_adata03, "Mean variance should be higher for noisy data."

    # Compare variance per archetype
    max_variance_adata0 = pt.get_aa_bootstrap(adata0, n_archetypes=5)["variance_per_archetype"].max()
    min_variance_adata03 = pt.get_aa_bootstrap(adata03, n_archetypes=5)["variance_per_archetype"].min()
    assert max_variance_adata0 < min_variance_adata03, (
        "Max variance of non-noisy data should be less than min variance of noisy data."
    )


### getter utilities ###


@pytest.mark.github_actions
def test_get_aa_metrics_filtering_behavior():
    adata = anndata.AnnData(np.empty((0, 0)))
    cfg_base = _make_dummy_config(3)
    cfg_other = _make_dummy_config(4)

    metrics_base = pd.DataFrame({"n_archetypes": [3], "IC": [1.0]})
    metrics_other = pd.DataFrame({"n_archetypes": [4], "IC": [2.0]})

    adata.uns["AA_selection_metrics"] = {cfg_base: metrics_base, cfg_other: metrics_other}

    with pytest.raises(ValueError):
        pt.get_aa_metrics(adata)

    df = pt.get_aa_metrics(adata, n_archetypes=3)
    assert df.equals(metrics_base)

    cfg, df_with_cfg = pt.get_aa_metrics(adata, return_config=True, n_archetypes=4)
    assert cfg.n_archetypes == 4
    assert df_with_cfg.equals(metrics_other)

    with pytest.raises(ValueError):
        pt.get_aa_metrics(adata, n_archetypes=5)

    del adata.uns["AA_selection_metrics"]
    with pytest.raises(ValueError):
        pt.get_aa_metrics(adata)


@pytest.mark.github_actions
def test_get_aa_bootstrap_filtering_behavior():
    adata = anndata.AnnData(np.empty((0, 0)))
    cfg_base = _make_dummy_config(2)
    cfg_other = _make_dummy_config(3)

    bootstrap_base = pd.DataFrame({"archetype": [0], "iter": [0]})
    bootstrap_other = pd.DataFrame({"archetype": [1], "iter": [1]})

    adata.uns["AA_bootstrap"] = {cfg_base: bootstrap_base, cfg_other: bootstrap_other}

    with pytest.raises(ValueError):
        pt.get_aa_bootstrap(adata)

    df = pt.get_aa_bootstrap(adata, n_archetypes=2)
    assert df.equals(bootstrap_base)

    cfg, df_with_cfg = pt.get_aa_bootstrap(adata, return_config=True, n_archetypes=3)
    assert cfg.n_archetypes == 3
    assert df_with_cfg.equals(bootstrap_other)

    with pytest.raises(ValueError):
        pt.get_aa_bootstrap(adata, n_archetypes=4)

    del adata.uns["AA_bootstrap"]
    with pytest.raises(ValueError):
        pt.get_aa_bootstrap(adata)


@pytest.mark.github_actions
def test_summarize_aa_metrics_requires_matching_configs():
    adata = anndata.AnnData(np.empty((0, 0)))
    cfg_base = _make_dummy_config(2)
    cfg_other = _make_dummy_config(3)
    cfg_mismatch = _make_dummy_config(4, init="uniform")

    adata.uns["AA_selection_metrics"] = {
        cfg_base: pd.DataFrame({"k": [2], "varexpl": [0.8]}),
        cfg_other: pd.DataFrame({"k": [3], "varexpl": [0.85]}),
        cfg_mismatch: pd.DataFrame({"k": [4], "varexpl": [0.9]}),
    }

    summary = pt.summarize_aa_metrics(adata, init="plus_plus")
    assert set(summary["k"]) == {2, 3}

    with pytest.raises(ValueError):
        pt.summarize_aa_metrics(adata)


### compute_archetypes ###


@pytest.mark.github_actions
def test_compute_archetypes_output_shape(_mock_adata):
    """Test the output shape of compute_archetypes.

    Verifies:
    - The shape of the archetypes matrix `Z` matches `(n_archetypes, n_features)`.
    """
    adata = _mock_adata.copy()
    compute_archetypes(adata, n_archetypes=3)
    _, payload = pt.get_aa_result(adata, return_config=True)
    Z = payload["Z"]
    assert Z.shape == (3, len(adata.uns["AA_config"]["n_dimensions"])), "Archetypes matrix `Z` has incorrect shape."


@pytest.mark.github_actions
def test_compute_archetypes_archetypes_only_parameter(_mock_adata):
    """Test the `archetypes_only` parameter of compute_archetypes.

    Verifies:
    - Only the archetypes matrix `Z` is saved when `archetypes_only=True`.
    - All outputs are saved when `archetypes_only=False`.
    """
    # Test `archetypes_only=True`
    adata = _mock_adata.copy()
    compute_archetypes(adata, n_archetypes=3, archetypes_only=True)
    _, payload = pt.get_aa_result(adata, return_config=True)
    assert "Z" in payload, "Archetypes matrix `Z` not saved."
    assert "A" not in payload, "Unexpected key `A` in results."

    # Test `archetypes_only=False`
    compute_archetypes(adata, n_archetypes=3, archetypes_only=False, force_recompute=True)
    _, payload = pt.get_aa_result(adata, return_config=True)
    for key in ["A", "B", "Z", "RSS", "varexpl"]:
        assert key in payload, f"Missing key {key!r} in results when `archetypes_only=False`."


@pytest.mark.github_actions
def test_compute_archetypes_reproducibility(_mock_adata):
    """Test reproducibility of compute_archetypes.

    Verifies:
    - Results are consistent when the same `seed` is used.
    """
    adata = _mock_adata.copy()

    compute_archetypes(adata, n_archetypes=3, seed=42)
    Z1 = pt.get_aa_result(adata)["Z"].copy()

    compute_archetypes(adata, n_archetypes=3, seed=42, force_recompute=True)
    Z2 = pt.get_aa_result(adata)["Z"].copy()

    assert np.allclose(Z1, Z2), "Results are not reproducible with the same seed."


@pytest.mark.github_actions
def test_compute_archetypes_custom_parameters(_mock_adata):
    """Test custom parameters for compute_archetypes.

    Verifies:
    - The function correctly uses custom values for `init`, `optim`, `max_iter`, and `rel_tol`.
    """
    adata = _mock_adata.copy()

    compute_archetypes(adata, n_archetypes=3, init="uniform", optim="frank_wolfe", max_iter=100, rel_tol=1e-4)
    # No explicit assertions; this test ensures no errors are raised with custom parameters.


### set_obsm ###


@pytest.mark.github_actions
def test_set_obsm_input_validation(_mock_adata):
    """Test if input validation for set_obsm works as intended.

    Verifies:
    - Raises ValueError when obs layer does not exist in adata.
    - Raises ValueError when the specified number of dimensions is larger than available dimensions.
    """
    adata = _mock_adata.copy()
    # Test invalid obs layer
    with pytest.raises(ValueError):
        set_obsm(adata, "invalid_layer", 3)

    # Test invalid number of dimensions
    with pytest.raises(ValueError):
        set_obsm(adata, "X_pca", 51)


@pytest.mark.github_actions
def test_set_obsm_output_correct_shape(_mock_adata):
    """Test if set_obsm correctly updates adata.uns["AA_config"].

    Verifies:
    - Correct passing of obsm key.
    - Correct passing of number of dimensions.
    """
    adata = _mock_adata.copy()
    del adata.uns["AA_config"]
    set_obsm(adata, "X_pca", 7)

    # Check obsm key
    assert adata.uns["AA_config"]["obsm_key"] == "X_pca", "obsm key should be X_pca"

    # Check n_dimensions
    assert adata.uns["AA_config"]["n_dimensions"] == list(range(7)), "n_dimensions should be list(range(7))"


### var_explained_aa ###


@pytest.mark.github_actions
def test_var_explained_aa_input_validation(_mock_adata):
    """Test if input validation for var_explained_aa works as intended.

    Verifies:
    - Raises ValueError when min_k < 2.
    - Raises ValueError when max_k <= min_k.
    """
    adata = _mock_adata.copy()
    # Test invalid min_k
    with pytest.raises(ValueError):
        compute_selection_metrics(adata, n_archetypes_list=[1])

    # Test invalid max_k
    with pytest.raises(ValueError):
        compute_selection_metrics(adata, n_archetypes_list=[2, 1])


@pytest.mark.github_actions
def test_model_selection_metrics_aa_on_simulated_data():
    """Test if var_explained_aa works as intended on simulated data.

    Verifies:
    - dist_to_projected is highest at the number of archetypes used to generate the data.
    - IC is lowest at the number of archetypes used to generate the data.
    """
    sim_adata = _simulate_adata(n_samples=1000, n_dimensions=50, n_archetypes=5, n_pcs=6)
    compute_selection_metrics(sim_adata, n_archetypes_list=list(range(2, 11)))

    metrics_summary = pt.summarize_aa_metrics(sim_adata)
    assert metrics_summary.sort_values("IC").iloc[0]["k"] == 5, "Expected lowest IC at 5 archetypes"
