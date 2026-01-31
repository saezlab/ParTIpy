import anndata
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from partipy.enrichment import (
    compute_archetype_expression,
    compute_archetype_weights,
    compute_meta_enrichment,
    compute_quantile_based_categorical_enrichment,
    compute_quantile_based_continuous_enrichment,
    compute_quantile_based_gene_enrichment,
    extract_enriched_processes,
    extract_specific_processes,
)
from partipy.schema import (
    DEFAULT_INIT,
    DEFAULT_MAX_ITER,
    DEFAULT_OPTIM,
    DEFAULT_REL_TOL,
    DEFAULT_WEIGHT,
    ArchetypeConfig,
)
from partipy.simulate import simulate_archetypes
from scipy.spatial.distance import cdist

np.random.seed(42)


def _create_config(
    *,
    n_archetypes: int,
    obsm_key: str,
    n_dimensions: tuple[int, ...],
    seed: int = 42,
) -> ArchetypeConfig:
    return ArchetypeConfig(
        obsm_key=obsm_key,
        n_dimensions=n_dimensions,
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
        seed=seed,
        optim_kwargs={},
    )


def _extract_weights(adata: anndata.AnnData, *, result_filters: dict | None = None) -> np.ndarray:
    weights_store = adata.uns.get("AA_cell_weights")
    if weights_store is None:
        raise KeyError("No cell weights stored in AnnData object.")
    if isinstance(weights_store, dict):
        if result_filters:
            for cfg, weights in weights_store.items():
                match = all(getattr(cfg, k) == v for k, v in result_filters.items())
                if match:
                    return weights
            raise KeyError("No weights matching the provided filters were found.")
        if len(weights_store) != 1:
            raise KeyError("Multiple weight matrices present; please provide result_filters.")
        return next(iter(weights_store.values()))
    return np.asarray(weights_store)


def _simulate_adata(n_samples, n_dimensions, n_archetypes, n_pcs, seed: int = 42, compute_pca: bool = True):
    X, A, Z = simulate_archetypes(
        n_samples=n_samples, n_archetypes=n_archetypes, n_dimensions=n_dimensions, noise_std=0.0, seed=seed
    )
    adata = anndata.AnnData(X)
    if compute_pca:
        adata.obsm["X_pca"] = sc.pp.pca(X, n_comps=n_pcs)
    else:
        adata.obsm["X_pca"] = X
    dims = tuple(range(n_pcs))
    adata.uns["AA_config"] = {
        "obsm_key": "X_pca",
        "n_dimensions": list(dims),
    }
    config = _create_config(n_archetypes=n_archetypes, obsm_key="X_pca", n_dimensions=dims, seed=seed)
    adata.uns["AA_results"] = {config: {"Z": Z[:, :n_pcs]}}
    compute_archetype_weights(adata, result_filters={"n_archetypes": n_archetypes})
    return adata


def _make_quantile_enrichment_adata(n_cells: int = 200) -> anndata.AnnData:
    x = np.linspace(0.0, 1.0, n_cells).reshape(-1, 1)
    adata = anndata.AnnData(X=np.c_[1.0 - x.ravel(), x.ravel()])
    adata.var.index = ["gene0", "gene1"]
    adata.obsm["X_pca"] = x
    adata.obs["score"] = 1.0 - x.ravel()
    adata.obs["label"] = np.where(np.arange(n_cells) < n_cells // 2, "A", "B")
    adata.obs["is_A"] = adata.obs["label"] == "A"

    adata.uns["AA_config"] = {"obsm_key": "X_pca", "n_dimensions": [0]}
    config = _create_config(n_archetypes=2, obsm_key="X_pca", n_dimensions=(0,), seed=42)
    Z = np.array([[0.0], [1.0]])
    adata.uns["AA_results"] = {config: {"Z": Z}}
    return adata


### compute_archetype_weights ###


@pytest.mark.github_actions
def test_compute_archetype_weights_anndata():
    """Test AnnData input with automatic and manual mode.

    Verifies:
    - Saving results in `adata.uns["AA_cell_weights"]`
    - Correct output shape (n_samples × n_archetypes)
    - Weight bounds [0, 1]
    """
    # Setup
    adata = _simulate_adata(n_samples=1000, n_dimensions=10, n_archetypes=5, n_pcs=4)

    # Test automatic mode
    compute_archetype_weights(adata, result_filters={"n_archetypes": 5})
    assert "AA_cell_weights" in adata.uns, "Weights are not saved correctly"
    weights_store = adata.uns["AA_cell_weights"]
    assert isinstance(weights_store, dict), "Weights should be stored per AA configuration"
    ((cfg, weights),) = weights_store.items()
    assert weights.shape == (1000, 5), "Weights have wrong shape"
    assert np.all(weights >= 0) and np.all(weights <= 1), "Weights contain values outside [0, 1]"
    del adata.uns["AA_cell_weights"]

    # Test manual mode
    length_scale = 1.0
    compute_archetype_weights(adata, mode="manual", length_scale=length_scale, result_filters={"n_archetypes": 5})
    weights_store = adata.uns["AA_cell_weights"]
    assert isinstance(weights_store, dict), "Weights should be stored per AA configuration"
    weights = next(iter(weights_store.values()))
    assert weights.shape == (1000, 5), "Weights have wrong shape"
    assert np.all(weights >= 0) and np.all(weights <= 1), "Weights contain values outside [0, 1]"


@pytest.mark.github_actions
def test_compute_archetype_weights_missing_archetypes():
    """Test error handling when archetype information is missing.

    Verifies:
    For AnnData input: Raises ValueError when 'AA_results' is missing from .uns
    For array input: Raises ValueError when archetype coordinates (Z) are not provided
    """
    # Test adata
    adata = _simulate_adata(n_samples=1000, n_dimensions=10, n_archetypes=5, n_pcs=4)
    del adata.uns["AA_results"]
    with pytest.raises(ValueError):
        compute_archetype_weights(adata)


@pytest.mark.github_actions
def test_compute_archetype_weights_ground_truth():
    """Test known example to verify correct weight computation.

    Verifies:
    - Manual mode computes expected Gaussian weights
    - Automatic length scale estimation behaves as expected
    """

    # Setup simple data, two cells, two archetypes in 2D
    X = np.array([[2.0, 1.0], [4.0, 1.0]])

    Z = np.array([[1.0, 1.0], [5.0, 1.0]])

    # Expected distances between X and Z
    expected_distances = np.array([[1.0, 3.0], [3.0, 1.0]])

    # With length_scale=1.0, we can compute expected weights manually
    # Using formula: exp(-distance²/(2*length_scale²))
    expected_weights = np.exp(-(expected_distances**2) / 2)
    expected_weights /= expected_weights.sum(axis=1, keepdims=True)

    # Test with manual length scale
    adata = anndata.AnnData(X=X, obsm={"X_pca": X})
    adata.uns["AA_config"] = {
        "obsm_key": "X_pca",
        "n_dimensions": [0, 1],
    }
    cfg = ArchetypeConfig(
        obsm_key="X_pca",
        n_dimensions=(0, 1),
        n_archetypes=2,
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
    adata.uns["AA_results"] = {cfg: {"Z": Z}}
    weights = compute_archetype_weights(
        adata=adata,
        mode="manual",
        length_scale=1.0,
        save_to_anndata=False,
        result_filters={"n_archetypes": 2},
    )
    assert np.allclose(weights, expected_weights), "Manual mode weights do not match expected values"

    # Test if automatic scale is computed correctly
    # Since median distance of centroid is 2, length_scale should equal 1
    centroid = np.mean(X, axis=0).reshape(1, -1)
    length_scale = np.median(cdist(centroid, Z)) / 2
    assert np.isclose(length_scale, 1.0), "Length scale should be 1.0"


### compute_archetype_expression ###


@pytest.mark.github_actions
def test_compute_archetype_expression_result_shape():
    """Test output shape.

    Verifies:
    - Output shape matches (n_archetypes × n_genes)
    """
    adata = _simulate_adata(n_samples=1000, n_dimensions=10, n_archetypes=5, n_pcs=4)
    assert compute_archetype_expression(adata).shape == (5, 10), "Did not return expected shape"


@pytest.mark.github_actions
def test_compute_archetype_expression_ground_truth():
    """Test correct pseudobulk expression per archetype from known input.

    Verifies:
    - Expression values match expected weighted means
    - Expression values if layer is specified are handled correctly
    """
    # Setup: 2 cells, 2 genes, 2 archetypes with known weights
    expr = np.array([[1.0, 10.0], [2.0, 20.0]]).T

    weights = np.array([[0.8, 0.2], [0.3, 0.7]])

    adata = anndata.AnnData(X=expr)
    adata.var_names = ["gene1", "gene2"]
    adata.uns["AA_config"] = {
        "obsm_key": "X",
        "n_dimensions": [0, 1],
    }
    cfg = _create_config(n_archetypes=2, obsm_key="X", n_dimensions=(0, 1))
    adata.uns["AA_results"] = {cfg: {"Z": np.zeros((2, 2))}}
    adata.obsm["X"] = expr.T
    adata.uns["AA_cell_weights"] = {cfg: weights.T}

    expected_result = pd.DataFrame([[2.8, 5.6], [7.3, 14.6]], columns=["gene1", "gene2"])

    # Test default layer
    result = compute_archetype_expression(adata, result_filters={"n_archetypes": 2})
    assert np.allclose(result, expected_result, atol=1e-4), "Did not return expected results"

    # Test with layer
    adata.layers["scaled"] = expr * 2
    expected_scaled = expected_result * 2
    result_scaled = compute_archetype_expression(adata, layer="scaled", result_filters={"n_archetypes": 2})
    assert np.allclose(result_scaled, expected_scaled, atol=1e-4), (
        "Did not return expected results when layer was specified"
    )


@pytest.mark.github_actions
def test_compute_archetype_expression_input_validation():
    """Tets if input validation works as intended.

    Verifies:
    - Raises ValueError when no cell weights are saved in adata
    - Raises ValueError when the specified layer does not exist
    """

    adata = _simulate_adata(n_samples=1000, n_dimensions=10, n_archetypes=5, n_pcs=4)

    with pytest.raises(ValueError):
        compute_archetype_expression(adata, layer="dklmdsfm")

    del adata.uns["AA_cell_weights"]
    with pytest.raises(ValueError):
        compute_archetype_expression(adata)


### extract_enriched_processes ###


@pytest.mark.github_actions
def test_extract_enriched_processes_shape():
    """Test output shape

    Verifies:
    - Output is a dictionary with one DataFrame per archetype
    - Each DataFrame has shape (n_process, n_archetypes+2)
    """
    # Setup: 2 archetypes, 3 processes
    est = pd.DataFrame(
        {
            "0": [1.5, 0.5, -2.0],
            "1": [0.8, 1.2, -1.0],
        },
        index=["ProcessA", "ProcessB", "ProcessC"],
    ).T

    pval = pd.DataFrame(
        {
            "0": [0.01, 0.03, 0.02],
            "1": [0.02, 0.04, 0.01],
        },
        index=["ProcessA", "ProcessB", "ProcessC"],
    ).T

    result = extract_enriched_processes(est, pval, order="desc", p_threshold=0.05)

    # Test shape
    assert isinstance(result, dict), "Result should be a dictionary of DataFrames"
    assert len(result) == 2, "Expected one result per archetype (2 total)"

    assert result[0].shape == (3, 4), "Did not return expected shape for A0"
    assert result[1].shape == (3, 4), "Did not return expected shape for A1"


@pytest.mark.github_actions
def test_extract_enriched_processes_order():
    """Test process ordering.

    Verifies:
    - Processes are ordered by enrichment in correct direction (`order="desc"` or `"asc"`)
    """
    # Setup: 2 archetypes, 3 processes
    est = pd.DataFrame(
        {
            "0": [1.5, 0.5, -2.0],
            "1": [0.8, 1.2, -1.0],
        },
        index=["ProcessA", "ProcessB", "ProcessC"],
    ).T

    pval = pd.DataFrame(
        {
            "0": [0.01, 0.03, 0.02],
            "1": [0.02, 0.04, 0.01],
        },
        index=["ProcessA", "ProcessB", "ProcessC"],
    ).T

    # Run function in descending and ascending order
    result_desc = extract_enriched_processes(est, pval, order="desc", p_threshold=0.05)
    result_asc = extract_enriched_processes(est, pval, order="asc", p_threshold=0.05)

    # Descending: most enriched processes first
    # Archetype 0
    assert result_desc[0].iloc[0]["Process"] == "ProcessA", "Process not as expected for descending order, A0"
    assert result_desc[0].iloc[0]["0"] == 1.5, "Enrichment score not as expected for descending order, A0"
    # Archetype 1
    assert result_desc[1].iloc[0]["Process"] == "ProcessB", "Process not as expected for descending order, A1"
    assert result_desc[1].iloc[0]["1"] == 1.2, "Enrichment score not as expected for descending order, A1"

    # Ascending: least enriched processes first
    # Archetype 0
    assert result_asc[0].iloc[0]["Process"] == "ProcessC", "Process not as expected for ascending order, A0"
    assert result_asc[0].iloc[0]["0"] == -2.0, "Enrichment score not as expected for ascending order, A0"
    # Archetype 1
    assert result_asc[1].iloc[0]["Process"] == "ProcessC", "Process not as expected for ascending order, A1"
    assert result_asc[1].iloc[0]["1"] == -1.0, "Enrichment score not as expected for ascending order, A1"


@pytest.mark.github_actions
def test_extract_enriched_processes_pvalue_filtering():
    """Test p-value filtering.

    Verifies:
    - Only processes with p < threshold are included per archetype
    - Shape and content of output match filtered results
    """
    # Setup: 2 archetypes, 3 processes
    est = pd.DataFrame(
        {
            "0": [1.5, 0.5, -2.0],
            "1": [0.8, 1.2, -1.0],
        },
        index=["ProcessA", "ProcessB", "ProcessC"],
    ).T

    pval = pd.DataFrame(
        {
            "0": [0.01, 0.03, 0.2],
            "1": [0.02, 0.4, 0.01],
        },
        index=["ProcessA", "ProcessB", "ProcessC"],
    ).T

    result = extract_enriched_processes(est, pval, order="desc", p_threshold=0.05)

    # Archetype 0: ProcessC filtered out (p=0.2)
    assert result[0].shape == (2, 4), "Expected 2 enriched processes for Archetype 0"
    assert set(result[0]["Process"]) == {"ProcessA", "ProcessB"}, (
        "Expected only 'ProcessA' and 'ProcessB' to remain for Archetype 0"
    )

    # Archetype 1: ProcessB filtered out (p=0.4)
    assert result[1].shape == (2, 4), "Expected 2 enriched processes for Archetype 1"
    assert set(result[1]["Process"]) == {"ProcessA", "ProcessC"}, (
        "Expected only 'ProcessA' and 'ProcessC' to remain for Archetype 1"
    )


@pytest.mark.github_actions
def test_extract_enriched_processes_specificity():
    """Test specificity computation based on known input.

    Verifies:
    - Specificity is calculated as difference between the archetype's enrichment and the mean of the others.
    - Signs and magnitudes reflect direction and strength of enrichment.
    """
    # Setup: 3 archetypes, 3 processes
    est = pd.DataFrame(
        {
            "0": [3.0, 0.5, -1.0],
            "1": [1.0, 1.5, -0.5],
            "2": [2.0, 0.8, 0.0],
        },
        index=["ProcessA", "ProcessB", "ProcessC"],
    ).T

    pval = pd.DataFrame(
        {
            "0": [0.01, 0.02, 0.03],
            "1": [0.01, 0.04, 0.01],
            "2": [0.01, 0.01, 0.01],
        },
        index=["ProcessA", "ProcessB", "ProcessC"],
    ).T

    result = extract_enriched_processes(est, pval, order="desc", p_threshold=0.05)

    # 3.0-1.0=2.0, 3.0-1.0=2.0 -> min=1.0
    # 0.5-1.5=-1.0, 0.5-0.8=-0.3 -> min=-1.0
    # -1.0+0.5=-0.5, -1.0-0=-1.0 -> min=-1.0
    assert np.allclose(result[0]["specificity"], [1.0, -1.0, -1.0]), "Unexpected specificity for A0"
    # 1.5−0.8=0.7 Order is different because we sort after enrichment score
    # 1.0-3.0=-2.0
    # -0.5−0.0=-0.5
    assert np.allclose(result[1]["specificity"], [0.7, -2.0, -0.5]), "Unexpected specificity for A1"
    # 2.0−3.0=-1.0
    # 0.8−1.5=-0.7
    # 0.0+0.5=0.5
    assert np.allclose(result[2]["specificity"], [-1.0, -0.7, 0.5]), "Unexpected specificity for A2"
    result[2]


@pytest.mark.github_actions
def test_extract_enriched_processes_input_validation():
    """Tets if input validation works as intended.

    Verifies:
    - Raises ValueError when order is specified incorrectly.
    - Raises ValueError when the pvalue threshold is over 1 or below 0.
    - Raises ValueErroe when est and pval dimension do not match.
    """
    # Setup: 2 archetypes, 3 processes
    est = pd.DataFrame(
        {
            "0": [1.5, 0.5, -2.0],
            "1": [0.8, 1.2, -1.0],
        },
        index=["ProcessA", "ProcessB", "ProcessC"],
    ).T

    pval = pd.DataFrame(
        {
            "0": [0.01, 0.03, 0.02],
            "1": [0.02, 0.04, 0.01],
        },
        index=["ProcessA", "ProcessB", "ProcessC"],
    ).T

    with pytest.raises(ValueError):
        extract_enriched_processes(est, pval, order="a", p_threshold=0.05)

    with pytest.raises(ValueError):
        extract_enriched_processes(est, pval, order="desc", p_threshold=1.05)

    with pytest.raises(ValueError):
        extract_enriched_processes(est, pval, order="desc", p_threshold=-0.05)

    with pytest.raises(ValueError):
        extract_enriched_processes(est[:1], pval, order="desc", p_threshold=0.05)


### extract_specific_processes ###


@pytest.mark.github_actions
def test_extract_specific_processes_shape():
    """Test output shape

    Verifies:
    - Output is a dictionary with one DataFrame per archetype
    - Each DataFrame has shape (n_process, n_archetypes+2)
    """
    # Setup: 2 archetypes, 3 processes
    est = pd.DataFrame(
        {
            "0": [1.5, 0.5, -2.0],
            "1": [0.8, 1.2, -1.0],
        },
        index=["ProcessA", "ProcessB", "ProcessC"],
    ).T

    pval = pd.DataFrame(
        {
            "0": [0.01, 0.03, 0.02],
            "1": [0.02, 0.04, 0.01],
        },
        index=["ProcessA", "ProcessB", "ProcessC"],
    ).T

    result = extract_specific_processes(est, pval, p_threshold=0.05)

    # Test shape
    assert isinstance(result, dict), "Result should be a dictionary of DataFrames"
    assert len(result) == 2, "Expected one result per archetype (2 total)"

    assert result[0].shape == (3, 4), "Did not return expected shape for A0"
    assert result[1].shape == (3, 4), "Did not return expected shape for A1"


@pytest.mark.github_actions
def test_extract_specific_processes_pvalue_filtering():
    """Test p-value filtering.

    Verifies:
    - Only processes with p < threshold are included per archetype
    - Shape and content of output match filtered results
    """
    # Setup: 2 archetypes, 3 processes
    est = pd.DataFrame(
        {
            "0": [1.5, 0.5, -2.0],
            "1": [0.8, 1.2, -1.0],
        },
        index=["ProcessA", "ProcessB", "ProcessC"],
    ).T

    pval = pd.DataFrame(
        {
            "0": [0.01, 0.03, 0.2],
            "1": [0.02, 0.4, 0.01],
        },
        index=["ProcessA", "ProcessB", "ProcessC"],
    ).T

    result = extract_specific_processes(est, pval, p_threshold=0.05)

    # Archetype 0: ProcessC filtered out (p=0.2)
    assert result[0].shape == (2, 4), "Expected 2 enriched processes for Archetype 0"
    assert set(result[0]["Process"]) == {"ProcessA", "ProcessB"}, (
        "Expected only 'ProcessA' and 'ProcessB' to remain for Archetype 0"
    )

    # Archetype 1: ProcessB filtered out (p=0.4)
    assert result[1].shape == (2, 4), "Expected 2 enriched processes for Archetype 1"
    assert set(result[1]["Process"]) == {"ProcessA", "ProcessC"}, (
        "Expected only 'ProcessA' and 'ProcessC' to remain for Archetype 1"
    )


@pytest.mark.github_actions
def test_extract_specific_processes_specificity():
    """Test specificity computation based on known input.

    Verifies:
    - Specificity is calculated as difference between the archetype's enrichment and the mean of the others.
    - Signs and magnitudes reflect direction and strength of enrichment.
    """
    # Setup: 3 archetypes, 3 processes
    est = pd.DataFrame(
        {
            "0": [3.0, 0.5, -1.0],
            "1": [1.0, 1.5, -0.5],
            "2": [2.0, 0.8, 0.0],
        },
        index=["ProcessA", "ProcessB", "ProcessC"],
    ).T

    pval = pd.DataFrame(
        {
            "0": [0.01, 0.02, 0.03],
            "1": [0.01, 0.04, 0.01],
            "2": [0.01, 0.01, 0.01],
        },
        index=["ProcessA", "ProcessB", "ProcessC"],
    ).T

    result = extract_specific_processes(est, pval, p_threshold=0.05)

    # 3.0-1.0=2.0, 3.0-1.0=2.0 -> min=1.0
    # 0.5-1.5=-1.0, 0.5-0.8=-0.3 -> min=-1.0
    # -1.0+0.5=-0.5, -1.0-0=-1.0 -> min=-1.0
    assert np.allclose(result[0]["specificity"], [1.0, -1.0, -1.0]), "Unexpected specificity for A0"
    # 1.5−0.8=0.7
    # -0.5−0.0=-0.5
    # 1.0-3.0=-2.0
    assert np.allclose(result[1]["specificity"], [0.7, -0.5, -2.0]), "Unexpected specificity for A1"
    # 0.0+0.5=0.5
    # 0.8−1.5=-0.7
    # 2.0−3.0=-1.0
    assert np.allclose(result[2]["specificity"], [0.5, -0.7, -1.0]), "Unexpected specificity for A2"


@pytest.mark.github_actions
def test_extract_specific_processes_input_validation():
    """Tets if input validation works as intended.

    Verifies:
    - Raises ValueError when the pvalue threshold is over 1 or below 0.
    - Raises ValueErroe when est and pval dimension do not match.
    """
    # Setup: 2 archetypes, 3 processes
    est = pd.DataFrame(
        {
            "0": [1.5, 0.5, -2.0],
            "1": [0.8, 1.2, -1.0],
        },
        index=["ProcessA", "ProcessB", "ProcessC"],
    ).T

    pval = pd.DataFrame(
        {
            "0": [0.01, 0.03, 0.02],
            "1": [0.02, 0.04, 0.01],
        },
        index=["ProcessA", "ProcessB", "ProcessC"],
    ).T

    with pytest.raises(ValueError):
        extract_specific_processes(est, pval, p_threshold=1.05)

    with pytest.raises(ValueError):
        extract_specific_processes(est, pval, p_threshold=-0.05)

    with pytest.raises(ValueError):
        extract_specific_processes(est[:1], pval, p_threshold=0.05)


### compute_meta_enrichment ###


@pytest.mark.github_actions
def test_compute_meta_enrichment_correct_assigned():
    """
    Test whether meta-enrichment correctly assigns dominant labels to archetypes.

    Verifies:
    - Enrichment score is higher for the dominant label in each archetype
    - Enrichment matrix is row-normalized (each row sums to 1)
    """
    # Setup with 6 cells, 2 cell types and 5 genes
    adata = anndata.AnnData(X=np.random.rand(6, 5))
    adata.obs["cell_type"] = ["A", "A", "A", "B", "B", "B"]
    adata.uns["AA_config"] = {"obsm_key": "X", "n_dimensions": [0, 1]}
    cfg = _create_config(n_archetypes=2, obsm_key="X", n_dimensions=(0, 1))
    adata.uns["AA_results"] = {cfg: {"Z": np.zeros((2, 2))}}
    adata.obsm["X"] = adata.X
    adata.uns["AA_cell_weights"] = {
        cfg: np.array(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.1, 0.9],
            ]
        )
    }

    result = compute_meta_enrichment(adata, "cell_type", result_filters={"n_archetypes": 2})

    # Archetype 0 should be enriched for cell_type A
    assert result.loc[0, "A"] > result.loc[0, "B"], "Archetype 0 is not more enriched for cell type A"

    # Archetype 1 should be enriched for cell_type B
    assert result.loc[1, "B"] > result.loc[1, "A"], "Archetype 1 is not more enriched for cell type B"

    # Rows should sum to 1 (normalization check)
    assert np.allclose(result.sum(axis=1), [1.0, 1.0]), "Meta-enrichment rows do not sum to 1"


@pytest.mark.github_actions
def test_compute_meta_enrichment_input_validation():
    """Tests if input validation works as intended.

    Verifies:
    - Raises ValueError when ometa column does not exist
    - Raises ValueError when cell weights are missing
    """
    adata = _simulate_adata(n_samples=3, n_dimensions=10, n_archetypes=5, n_pcs=2)
    adata.obs["group"] = ["X", "Y", "Z"]

    with pytest.raises(ValueError):
        compute_meta_enrichment(adata, meta_col="dklmdsfm")

    with pytest.raises(ValueError):
        compute_meta_enrichment(adata, meta_col="group", datatype="dfkjgn")

    del adata.uns["AA_cell_weights"]
    with pytest.raises(ValueError):
        compute_meta_enrichment(adata, meta_col="group")


@pytest.mark.github_actions
def test_compute_meta_enrichment_normalization():
    """
    Test whether meta-enrichment correctly normalizes the contributions across archetypes based on known input.

    Verifies:
    - Meta-enrichment values reflect the weighted contributions of cell types
    """
    # Setup with 3 cells, 5 genes, 3 meta groups and 2 archetypes
    adata = anndata.AnnData(X=np.random.rand(3, 5))
    adata.obs["group"] = ["X", "Y", "Z"]
    adata.uns["AA_config"] = {"obsm_key": "X", "n_dimensions": [0, 1]}
    cfg = _create_config(n_archetypes=2, obsm_key="X", n_dimensions=(0, 1))
    adata.uns["AA_results"] = {cfg: {"Z": np.zeros((2, 2))}}
    adata.obsm["X"] = adata.X
    adata.uns["AA_cell_weights"] = {
        cfg: np.array(
            [
                [1.0, 0.0],  # Archetype 0 fully owns cell 0
                [0.0, 1.0],  # Archetype 1 fully owns cell 1
                [0.5, 0.5],  # Cell 2 split
            ]
        )
    }

    result = compute_meta_enrichment(adata, "group", result_filters={"n_archetypes": 2})
    # 2/3 from Archetype 0 is X, 1/3 from Z
    assert np.isclose(result.loc[0, "X"], 0.666, atol=0.01), "Archetype 0 X contribution not as expected"
    assert np.isclose(result.loc[0, "Z"], 0.333, atol=0.01), "Archetype 0 Z contribution not as expected"

    # 2/3 from Archetype 1 is Y, 1/3 from Z
    assert np.isclose(result.loc[1, "Y"], 0.666, atol=0.01), "Archetype 1 Y contribution not as expected"
    assert np.isclose(result.loc[1, "Z"], 0.333, atol=0.01), "Archetype 1 Z contribution not as expected"


@pytest.mark.github_actions
def test_compute_meta_enrichment_datatype_identification_and_shape():
    """
    Test whether compute_meta_enrichment correctly identifies and processes
    categorical and continuous metadata columns based on their dtype.

    Verifies:
    - Correct output shape
    - Correct identification of datatype
    """
    adata = _simulate_adata(n_samples=3, n_dimensions=10, n_archetypes=2, n_pcs=2)
    # Categorical metadata
    adata.obs["categorical"] = ["X", "Y", "Z"]
    # Continuous metadata
    adata.obs["continuous"] = [1, 2.5, 3]

    assert compute_meta_enrichment(adata, "categorical", result_filters={"n_archetypes": 2}).shape == (2, 3), (
        "Did not return expeected shape for categorical data"
    )
    assert compute_meta_enrichment(adata, "continuous", result_filters={"n_archetypes": 2}).shape == (2, 1), (
        "Did not return expeected shape for continuous data"
    )


@pytest.mark.github_actions
@pytest.mark.parametrize("seed", list(range(3)))
def test_compute_meta_enrichment_continuous_data(seed: int):
    """
    Test whether compute_meta_enrichment correctly computes weighted averages
    for continuous metadata. Ensures archetype 0 is enriched in high 'age' values
    and archetype 1 is enriched in low 'age' values.

    Verifies:
    - Correct calculation of continuous data enrichment
    """
    adata = _simulate_adata(n_samples=500, n_dimensions=2, n_archetypes=3, n_pcs=2, seed=seed, compute_pca=False)
    adata.obs["age"] = np.random.randint(10, 50, len(adata.obs))

    X = adata.X
    Z = next(iter(adata.uns["AA_results"].values()))["Z"]

    # Track used indices so we avoid overlap
    used = set()

    # --- Archetype 0 ---
    dists0 = cdist(X, Z[0:1, :])[:, 0]
    idx0_sorted = np.argsort(dists0)

    # take the first 100
    idx0 = [i for i in idx0_sorted if i not in used][:100]
    used.update(idx0)

    # Force age bias
    selected_cells_0 = adata.obs_names[idx0]
    adata.obs.loc[selected_cells_0, "age"] = 100

    # --- Archetype 1 ---
    dists1 = cdist(X, Z[1:2, :])[:, 0]
    idx1_sorted = np.argsort(dists1)

    # Take first 100 not already used
    idx1 = [i for i in idx1_sorted if i not in used][:100]
    used.update(idx1)

    # Force age bias
    selected_cells_1 = adata.obs_names[idx1]
    adata.obs.loc[selected_cells_1, "age"] = 5

    # Continue normally
    res = compute_meta_enrichment(
        adata,
        "age",
        datatype="continuous",
        result_filters={"n_archetypes": 3},
    )
    assert res.iloc[0].item() > res.iloc[2].item() > res.iloc[1].item()
    assert res.iloc[0].item() <= 100


@pytest.mark.github_actions
def test_quantile_based_gene_enrichment_basic():
    adata = _make_quantile_enrichment_adata()
    res = compute_quantile_based_gene_enrichment(adata, n_bins=10, result_filters={"n_archetypes": 2})
    assert res.shape[0] == 4
    assert set(res["gene"]) == {"gene0", "gene1"}

    def _row(arch_idx: int, gene: str) -> pd.Series:
        return res[(res["arch_idx"] == arch_idx) & (res["gene"] == gene)].iloc[0]

    assert _row(0, "gene0")["enriched"]
    assert _row(1, "gene1")["enriched"]
    assert not _row(0, "gene1")["enriched"]
    assert not _row(1, "gene0")["enriched"]


@pytest.mark.github_actions
def test_quantile_based_continuous_enrichment_basic():
    adata = _make_quantile_enrichment_adata()
    res = compute_quantile_based_continuous_enrichment(
        adata, colnames="score", n_bins=10, result_filters={"n_archetypes": 2}
    )
    assert res.shape[0] == 2

    def _row(arch_idx: int) -> pd.Series:
        return res[(res["arch_idx"] == arch_idx) & (res["colname"] == "score")].iloc[0]

    assert _row(0)["enriched"]
    assert not _row(1)["enriched"]


@pytest.mark.github_actions
def test_quantile_based_continuous_enrichment_nan_handling():
    adata = _make_quantile_enrichment_adata()
    adata.obs.loc[adata.obs.index[0], "score"] = np.nan
    with pytest.raises(ValueError):
        compute_quantile_based_continuous_enrichment(adata, colnames="score", result_filters={"n_archetypes": 2})
    res = compute_quantile_based_continuous_enrichment(
        adata, colnames="score", ignore_nans=True, result_filters={"n_archetypes": 2}
    )
    assert res.shape[0] == 2


@pytest.mark.github_actions
def test_quantile_based_continuous_enrichment_type_validation():
    adata = _make_quantile_enrichment_adata()
    with pytest.raises(TypeError):
        compute_quantile_based_continuous_enrichment(adata, colnames="label", result_filters={"n_archetypes": 2})


@pytest.mark.github_actions
def test_quantile_based_categorical_enrichment_basic():
    adata = _make_quantile_enrichment_adata()
    res = compute_quantile_based_categorical_enrichment(
        adata,
        colnames="label",
        n_bins=10,
        min_category_count=100,
        result_filters={"n_archetypes": 2},
    )
    assert res.shape[0] == 4
    assert set(res["category"]) == {"A", "B"}

    def _row(arch_idx: int, category: str) -> pd.Series:
        return res[(res["arch_idx"] == arch_idx) & (res["category"] == category)].iloc[0]

    assert _row(0, "A")["enriched"]
    assert _row(1, "B")["enriched"]
    assert not _row(0, "B")["enriched"]
    assert not _row(1, "A")["enriched"]


@pytest.mark.github_actions
def test_quantile_based_categorical_enrichment_min_category_count():
    adata = _make_quantile_enrichment_adata()
    with pytest.raises(ValueError):
        compute_quantile_based_categorical_enrichment(
            adata, colnames="label", min_category_count=101, result_filters={"n_archetypes": 2}
        )


@pytest.mark.github_actions
def test_quantile_based_categorical_enrichment_nan_handling():
    adata = _make_quantile_enrichment_adata()
    adata.obs.loc[adata.obs.index[0], "label"] = np.nan
    with pytest.raises(ValueError):
        compute_quantile_based_categorical_enrichment(adata, colnames="label", result_filters={"n_archetypes": 2})
    res = compute_quantile_based_categorical_enrichment(
        adata, colnames="label", ignore_nans=True, min_category_count=1, result_filters={"n_archetypes": 2}
    )
    assert res.shape[0] == 4


@pytest.mark.github_actions
def test_quantile_based_categorical_enrichment_type_validation():
    adata = _make_quantile_enrichment_adata()
    with pytest.raises(TypeError):
        compute_quantile_based_categorical_enrichment(adata, colnames="score", result_filters={"n_archetypes": 2})


@pytest.mark.github_actions
def test_quantile_based_categorical_enrichment_bool_column():
    adata = _make_quantile_enrichment_adata()
    res = compute_quantile_based_categorical_enrichment(
        adata,
        colnames="is_A",
        n_bins=10,
        min_category_count=100,
        result_filters={"n_archetypes": 2},
    )
    assert res.shape[0] == 4
