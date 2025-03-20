__version__ = "0.0.1"

from .arch import AA
from .enrichment import (
    barplot_meta_enrichment,
    calculate_weights,
    extract_top_processes,
    extract_top_specific_processes,
    heatmap_meta_enrichment,
    meta_enrichment,
    plot_enrichment_comparison,
    plot_functional_enrichment,
    radarplot_meta_enrichment,
    weighted_expr,
)
from .paretoti_funcs import (
    align_archetypes,
    bootstrap_aa,
    compute_AA,
    compute_t_ratio,
    plot_2D,
    plot_2D_adata,
    plot_3D,
    plot_bootstrap_aa,
    plot_projected_dist,
    plot_var_explained_aa,
    plot_var_on_top,
    project_on_affine_subspace,
    set_dimension,
    t_ratio_significance,
    var_explained_aa,
)

# what is exposed when running 'from ParTIpy import *'
__all__ = [
    # From .enrichment
    "calculate_weights",
    "weighted_expr",
    "extract_top_processes",
    "extract_top_specific_processes",
    "meta_enrichment",
    "barplot_meta_enrichment",
    "heatmap_meta_enrichment",
    "radarplot_meta_enrichment",
    "plot_functional_enrichment",
    "plot_enrichment_comparison",
    # From .paretoti_funcs
    "set_dimension",
    "var_explained_aa",
    "plot_var_explained_aa",
    "plot_projected_dist",
    "plot_var_on_top",
    "bootstrap_aa",
    "plot_bootstrap_aa",
    "project_on_affine_subspace",
    "compute_t_ratio",
    "t_ratio_significance",
    "plot_2D",
    "plot_2D_adata",
    "plot_3D",
    "align_archetypes",
    "compute_AA",
    # From .arch
    "AA",
]
