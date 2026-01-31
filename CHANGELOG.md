# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## 0.1.0

### Added
- Added optimizer aliases (`"PCHA"` for `"projected_gradients"` and `"FW"` for `"frank_wolfe"`) across the public API, including caching filters and documentation, plus regression tests to ensure both names yield identical results.
- Added quantile-based continuous enrichment for gene expression and numeric `adata.obs` columns with ParTI-style binning, max-in-bin0 criteria, and optional NaN ignoring for obs columns.
- Added quantile-based categorical enrichment for `adata.obs` labels with ParTI-style binning, max-in-bin0 filtering, hypergeometric over-representation testing, configurable background contrast, and minimum category count filtering.

### Changed
- Quantile-based continuous and categorical enrichment now report raw p-values without FDR correction.

## 0.0.6

### Added
- `partipy.write_h5ad` and `partipy.read_h5ad` helpers that make the `AnnData.uns` caches HDF5-compatible by serializing and restoring `ArchetypeConfig` keys.
- Automatic restoration of cached dictionaries keyed by `ArchetypeConfig` when using the public accessors, enabling use of `.h5ad` files saved with the helper utilities.

## 0.0.5

### Added
- Public accessor layer for cached archetypal analysis artifacts (`get_aa_result`, `get_aa_cell_weights`, `get_aa_metrics`, `get_aa_bootstrap`, `summarize_aa_metrics`) with consistent filtering semantics.
- Comprehensive documentation on caching and retrieval flows, including the new `docs/notebooks/data_access.ipynb` tutorial and updates to other notebooks.
- New bootstrap and selection-metric plotting enhancements that rely on the unified accessors.

### Changed
- Reworked AA caching to remove the eagerly stored `adata.uns['AA_metrics_df']`, generating summaries on demand instead using `summarize_aa_metrics`
- Refactored t-ratio significance testing and AA result handling to better reuse cached runs and ensure typing/mypy compliance.
- Updated plotting APIs (`plot_var_explained`, `plot_IC`, `plot_bootstrap_*`, `plot_archetypes_*`) to require precomputed caches and use the new result filters.
- Streamlined schema defaults and test fixtures after the accessor refactor.
- Multiple unit-test adjustments to align with the new caching workflow.

## 0.0.4

First release alpha version of `partipy`
