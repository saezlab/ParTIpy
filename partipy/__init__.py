__version__ = "0.0.1"

from .arch import AA
from .enrichment import (
    compute_archetype_expression,
    compute_archetype_weights,
    compute_meta_enrichment,
    extract_enriched_processes,
    extract_specific_processes,
)
from .paretoti import (
    bootstrap_aa,
    bootstrap_aa_multiple_k,
    compute_archetypes,
    compute_t_ratio,
    set_obsm,
    t_ratio_significance,
    t_ratio_significance_shuffled,
    var_explained_aa,
)
from .plotting import (
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
from .simulate import simulate_archetypes
