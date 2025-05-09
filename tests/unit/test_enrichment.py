import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from partipy.enrichment import (
    compute_archetype_expression,
    compute_archetype_weights,
    compute_meta_enrichment,
    extract_enriched_processes,
    extract_specific_processes,
)
from partipy.simulate import simulate_archetypes
from scipy.spatial.distance import cdist

np.random.seed(42)


def _simulate_adata(n_samples, n_dimensions, n_archetypes, n_pcs):
    X, A, Z = simulate_archetypes(
        n_samples=n_samples, n_archetypes=n_archetypes, n_dimensions=n_dimensions, noise_std=0.0
    )
    adata = sc.AnnData(X)
    adata.obsm["X_pca"] = sc.pp.pca(X, n_comps=n_pcs)
    adata.uns["aa_config"] = {
        "obsm_key": "X_pca",
        "n_dimension": n_pcs,
    }
    adata.uns["AA_results"] = {"Z": Z[:, :n_pcs]}
    compute_archetype_weights(adata)
    return adata


### compute_archetype_weights ###


@pytest.mark.github_actions
def test_compute_archetype_weights_anndata():
    """Test AnnData input with automatic and manual mode.

    Verifies:
    - Saving results in `adata.obsm["cell_weights"]`
    - Correct output shape (n_samples × n_archetypes)
    - Weight bounds [0, 1]
    """
    # Setup
    adata = _simulate_adata(n_samples=1000, n_dimensions=10, n_archetypes=5, n_pcs=4)

    # Test automatic mode
    compute_archetype_weights(adata)
    assert "cell_weights" in adata.obsm, "Weights are not saved correctly"
    assert adata.obsm["cell_weights"].shape == (1000, 5), "Weights have wrong shape"
    assert np.all(adata.obsm["cell_weights"] >= 0) and np.all(adata.obsm["cell_weights"] <= 1), (
        "Weights are not saved correctly"
    )
    del adata.obsm["cell_weights"]

    # Test manual mode
    length_scale = 1.0
    compute_archetype_weights(adata, mode="manual", length_scale=length_scale)
    assert "cell_weights" in adata.obsm, "Weights are not saved correctly"
    assert adata.obsm["cell_weights"].shape == (1000, 5), "Weights have wrong shape"
    assert np.all(adata.obsm["cell_weights"] >= 0) and np.all(adata.obsm["cell_weights"] <= 1), (
        "Weights are not saved correctly"
    )


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

    # Test with manual length scale
    adata = sc.AnnData(X=X, obsm={"X_pca": X})
    adata.uns["AA_results"] = {"Z": Z}
    adata.uns["aa_config"] = {
        "obsm_key": "X_pca",
        "n_dimension": 2,
    }
    weights = compute_archetype_weights(adata=adata, mode="manual", length_scale=1.0, save_to_anndata=False)
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
    expr = np.array([[1.0, 10.0], [2.0, 20.0]])

    weights = np.array([[0.8, 0.2], [0.3, 0.7]])

    adata = sc.AnnData(X=expr)
    adata.obsm["cell_weights"] = weights
    adata.var_names = ["gene1", "gene2"]

    # Expected pseudobulk calculation:
    # Archetype 1:
    # gene1: (0.8*1 + 0.3*2)/(0.8+0.3) ≈ 1.2727
    # gene2: (0.8*10 + 0.3*20)/(0.8+0.3) ≈ 12.7273

    # Archetype 2:
    # gene1: (0.2*1 + 0.7*2)/(0.2+0.7) ≈ 1.7778
    # gene2: (0.2*10 + 0.7*20)/(0.2+0.7) ≈ 17.7778

    expected_result = pd.DataFrame([[1.2727, 12.7273], [1.7778, 17.7778]], columns=["gene1", "gene2"])

    # Test default layer
    result = compute_archetype_expression(adata)
    assert np.allclose(result, expected_result, atol=1e-4), "Did not return expected results"

    # Test with layer
    adata.layers["scaled"] = expr * 2
    expected_scaled = expected_result * 2
    result_scaled = compute_archetype_expression(adata, layer="scaled")
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

    del adata.obsm["cell_weights"]
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
    adata = sc.AnnData(X=np.random.rand(6, 5))
    adata.obs["cell_type"] = ["A", "A", "A", "B", "B", "B"]
    adata.obsm["cell_weights"] = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.1, 0.9],
        ]
    )

    result = compute_meta_enrichment(adata, "cell_type")

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

    del adata.obsm["cell_weights"]
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
    adata = sc.AnnData(X=np.random.rand(3, 5))
    adata.obs["group"] = ["X", "Y", "Z"]

    adata.obsm["cell_weights"] = np.array(
        [
            [1.0, 0.0],  # Archetype 0 fully owns cell 0
            [0.0, 1.0],  # Archetype 1 fully owns cell 1
            [0.5, 0.5],  # Cell 2 split
        ]
    )

    result = compute_meta_enrichment(adata, "group")
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

    assert compute_meta_enrichment(adata, "categorical").shape == (2, 3), (
        "Did not return expeected shape for categorical data"
    )
    assert compute_meta_enrichment(adata, "continuous").shape == (2, 1), (
        "Did not return expeected shape for continuous data"
    )


@pytest.mark.github_actions
def test_compute_meta_enrichment_continuous_data():
    """
    Test whether compute_meta_enrichment correctly computes weighted averages
    for continuous metadata. Ensures archetype 0 is enriched in high 'age' values
    and archetype 1 is enriched in low 'age' values.

    Verifies:
    - Correct calculation of continuous data enrichment
    """
    adata = _simulate_adata(n_samples=300, n_dimensions=10, n_archetypes=3, n_pcs=2)

    # Assign random ages initially
    adata.obs["age"] = np.random.randint(5, 70, len(adata.obs))

    # Force age bias
    selected_cells = adata.obs_names[adata.obsm["cell_weights"][:, 0] > 0.25]
    adata.obs.loc[selected_cells, "age"] = 70
    selected_cells = adata.obs_names[adata.obsm["cell_weights"][:, 1] > 0.25]
    adata.obs.loc[selected_cells, "age"] = 5

    res = compute_meta_enrichment(adata, "age", datatype="continuous")
    assert res.iloc[0].item() > res.iloc[2].item() > res.iloc[1].item()
