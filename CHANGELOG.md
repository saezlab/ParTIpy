# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## 0.0.4

First release alpha version of `partipy`

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
