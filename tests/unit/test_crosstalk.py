import numpy as np
import pandas as pd
import pytest
from partipy.crosstalk import (
    get_archetype_crosstalk,
    get_specific_genes_per_archetype,
)


# ------------------------------------------------------------------
# Test data fixtures
# ------------------------------------------------------------------
@pytest.fixture()
def toy_expr_df() -> pd.DataFrame:
    """3 archetypes x 5 genes synthetic expression matrix (ParTIpy convention).

    **Orientation:** rows = archetypes (0,1,2), columns = genes (G1..G5).

    Patterns:
    - G1 high in archetype 0.
    - G2 high in archetype 1.
    - G3 moderate in archetype 1 & 2.
    - G4 high in archetype 2 only.
    - G5 uniform (non-specific baseline).
    """
    return pd.DataFrame(
        {
            "G1": [5.0, 1.0, 0.0],
            "G2": [1.0, 4.0, 0.0],
            "G3": [0.0, 2.0, 3.0],
            "G4": [0.0, 0.0, 5.0],
            "G5": [1.0, 1.0, 1.0],
        },
        index=[0, 1, 2],
    )


@pytest.fixture()
def toy_lr_resource() -> pd.DataFrame:
    """Simple ligand-receptor table across toy genes.

    We'll designate:
    - G1 ligand for receptor G2
    - G2 ligand for receptor G3
    - G1 ligand for receptor G3 (multi-target)
    - G4 ligand for receptor G1 (nonspecific gene -> may be filtered out)
    """
    return pd.DataFrame(
        {
            "ligand": ["G1", "G2", "G1", "G4"],
            "receptor": ["G2", "G3", "G3", "G1"],
            "source_db": ["mock", "mock", "mock", "mock"],
        }
    )


@pytest.fixture()
def specificity_expected(toy_expr_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Hand-compute expected specificity metrics per docs.

    Returns dict keyed by archetype index; values DataFrame indexed by gene.
    Columns: z_score, max_z_score_others, specificity_score, gene
    """
    # Convert to legacy orientation (genes rows; archetypes cols) to reuse a simple formula.
    expr_genes_rows = toy_expr_df.T  # genes rows; archetypes cols
    gene_ids = [str(gene) for gene in expr_genes_rows.index.to_list()]
    archetype_ids = [int(idx) for idx in expr_genes_rows.columns.to_list()]

    expected_dict: dict[int, pd.DataFrame] = {}
    for arch_idx in archetype_ids:
        other_arch_ids = [other for other in archetype_ids if other != arch_idx]
        z_this_arch = expr_genes_rows[arch_idx].to_numpy(dtype=float)
        z_other_arches = expr_genes_rows[other_arch_ids].to_numpy(dtype=float)
        max_other = np.nanmax(z_other_arches, axis=1)
        # broadcast diff: (n_genes, n_others) -> min across others
        spec = np.nanmin((z_this_arch[:, None] - z_other_arches), axis=1)
        expected_df = pd.DataFrame(
            {
                "z_score": z_this_arch,
                "max_z_score_others": max_other,
                "specificity_score": spec,
                "gene": gene_ids,
            },
            index=gene_ids,
        ).sort_values("specificity_score", ascending=False)
        expected_dict[arch_idx] = expected_df
    return expected_dict


# ------------------------------------------------------------------
# Utility asserts
# ------------------------------------------------------------------
@pytest.mark.github_actions
def _assert_df_equal_numeric(
    observed_df: pd.DataFrame,
    expected_df: pd.DataFrame,
    cols,
    atol: float = 1e-8,
):
    """Assert numeric equality for selected columns after aligning by gene."""
    obs_aligned = observed_df.set_index("gene").loc[:, cols].sort_index()
    exp_aligned = expected_df.set_index("gene").loc[:, cols].sort_index()
    assert obs_aligned.shape == exp_aligned.shape, f"Shape mismatch: {obs_aligned.shape} vs {exp_aligned.shape}"
    for col_name in cols:
        assert np.allclose(
            obs_aligned[col_name].values,
            exp_aligned[col_name].values,
            atol=atol,
            equal_nan=True,
        ), f"Column {col_name} mismatch"


# ------------------------------------------------------------------
# Tests for get_specific_genes_per_archetype
# ------------------------------------------------------------------
@pytest.mark.github_actions
def test_specific_genes_basic(toy_expr_df, specificity_expected):
    results_dict = get_specific_genes_per_archetype(toy_expr_df, min_score=-np.inf, max_score=np.inf)

    # --- orientation / key check ---
    assert set(results_dict.keys()) == set(toy_expr_df.index), (
        "Return keys should match archetype rows. If this assertion fails, "
        "check whether the input DataFrame was transposed inside the function."
    )

    # Compare numeric contents for each archetype
    for arch_idx in toy_expr_df.index:
        observed = results_dict[arch_idx].copy()
        expected = specificity_expected[arch_idx].copy()
        # columns expected
        for col_name in ["z_score", "max_z_score_others", "specificity_score", "gene"]:
            assert col_name in observed.columns, f"Missing column {col_name} in archetype {arch_idx}"
        # Values: because function sorts descending, align by gene
        _assert_df_equal_numeric(
            observed,
            expected,
            ["z_score", "max_z_score_others", "specificity_score"],
        )


@pytest.mark.github_actions
@pytest.mark.parametrize(
    "min_score,max_score,expected_counts",
    [
        (0.0, np.inf, {0: 2, 1: 2, 2: 3}),  # inclusive >=0; G5 counts as 0 specificity
        (-np.inf, 0.0, {0: 4, 1: 4, 2: 3}),  # <=0 inclusive (0 counts; see table)
        (2.5, np.inf, {0: 1, 1: 1, 2: 1}),  # genes with specificity >=2.5 (G1, G2, G4)
    ],
)
def test_specific_genes_filtering(toy_expr_df, specificity_expected, min_score, max_score, expected_counts):
    results_dict = get_specific_genes_per_archetype(toy_expr_df, min_score=min_score, max_score=max_score)
    for arch_idx, expected_count in expected_counts.items():
        assert len(results_dict[arch_idx]) == expected_count, f"Unexpected filtered row count for archetype {arch_idx}"
        if expected_count > 0:
            # ensure thresholds enforced
            assert (results_dict[arch_idx]["specificity_score"] >= min_score - 1e-12).all()
            assert (results_dict[arch_idx]["specificity_score"] <= max_score + 1e-12).all()


@pytest.mark.github_actions
def test_specific_genes_sorted_desc(toy_expr_df):
    results_dict = get_specific_genes_per_archetype(toy_expr_df, min_score=-np.inf)
    for arch_idx in toy_expr_df.index:
        scores = results_dict[arch_idx]["specificity_score"].values
        assert np.all(scores[:-1] >= scores[1:]), f"Scores not sorted descending for archetype {arch_idx}"


def _make_specific_genes_for_crosstalk(toy_expr_df):
    """Helper returns *unfiltered* specificity dict so that all genes are present."""
    return get_specific_genes_per_archetype(toy_expr_df, min_score=-np.inf, max_score=np.inf)


# ------------------------------------------------------------------
# Tests for get_archetype_crosstalk
# ------------------------------------------------------------------
@pytest.mark.github_actions
def test_crosstalk_basic(toy_expr_df, toy_lr_resource):
    specificity_dict = _make_specific_genes_for_crosstalk(toy_expr_df)
    crosstalk_dict = get_archetype_crosstalk(specificity_dict, toy_lr_resource)

    archetype_ids = toy_expr_df.index.to_list()
    # nested dict shape
    assert set(crosstalk_dict.keys()) == set(archetype_ids)
    for arch_idx in archetype_ids:
        assert set(crosstalk_dict[arch_idx].keys()) == set(archetype_ids)

    # pick one sender/receiver pair and inspect values
    df_sender0_receiver1 = crosstalk_dict[0][1]
    row = df_sender0_receiver1.loc[
        (df_sender0_receiver1["ligand"] == "G1") & (df_sender0_receiver1["receptor"] == "G2")
    ]
    assert not row.empty, "Expected G1->G2 interaction missing in 0->1 archetype crosstalk"

    # ligand_z_score from archetype 0's specificity table, receptor_z_score from archetype 1's
    lig_z = specificity_dict[0].set_index("gene").loc["G1", "z_score"]
    rec_z = specificity_dict[1].set_index("gene").loc["G2", "z_score"]
    assert np.isclose(row["ligand_z_score"].item(), lig_z)
    assert np.isclose(row["receptor_z_score"].item(), rec_z)


@pytest.mark.github_actions
def test_crosstalk_self_edges_included(toy_expr_df, toy_lr_resource):
    specificity_dict = _make_specific_genes_for_crosstalk(toy_expr_df)
    crosstalk_dict = get_archetype_crosstalk(specificity_dict, toy_lr_resource)

    # self 0->0 should include any LR pair where both ligand & receptor present in archetype 0's table
    df_self = crosstalk_dict[0][0]
    assert ((df_self["ligand"] == "G1") & (df_self["receptor"] == "G2")).any(), "Self-edge missing G1->G2"
