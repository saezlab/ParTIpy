__version__ = "0.0.1"

from .arch import AA
from .decomposition import (
    compute_shuffled_pca,
    plot_shuffled_pca,
)
from .enrichment import (
    compute_archetype_expression,
    compute_archetype_weights,
    compute_meta_enrichment,
    compute_quantile_based_categorical_enrichment,
    compute_quantile_based_continuous_enrichment,
    compute_quantile_based_gene_enrichment,
    extract_enriched_processes,
    extract_specific_processes,
)
from .io import ensure_archetype_config_keys, read_h5ad, write_h5ad
from .paretoti import (
    compute_archetypes,
    compute_bootstrap_variance,
    compute_selection_metrics,
    compute_t_ratio,
    delete_aa_result,
    get_aa_bootstrap,
    get_aa_cell_weights,
    get_aa_metrics,
    get_aa_result,
    set_obsm,
    summarize_aa_metrics,
    t_ratio_significance,
)
from .plotting import (
    barplot_enrichment_comparison,
    barplot_functional_enrichment,
    barplot_meta_enrichment,
    heatmap_meta_enrichment,
    plot_2D,
    plot_archetypes_2D,
    plot_archetypes_3D,
    plot_bootstrap_2D,
    plot_bootstrap_3D,
    plot_bootstrap_variance,
    plot_IC,
    plot_top_genes,
    plot_var_explained,
    radarplot_meta_enrichment,
)
from .simulate import simulate_archetypes
