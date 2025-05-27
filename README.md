# ParTIpy: Pareto Task Inference in Python
<!-- badges: start -->
[![codecov](https://codecov.io/gh/saezlab/partipy/branch/main/graph/badge.svg)](https://codecov.io/gh/saezlab/partipy)
[![Documentation Status](https://readthedocs.org/projects/partipy/badge/?version=latest)](https://partipy.readthedocs.io/en/latest/?badge=latest)
[![GitHub issues](https://img.shields.io/github/issues/saezlab/partipy.svg)](https://github.com/saezlab/partipy/issues/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/saezlab/ParTIpy/main.svg)](https://results.pre-commit.ci/latest/github/saezlab/ParTIpy/main)
<!-- badges: end -->
`partipy` provides a scalable and user-friendly implementation of the Pareto Task Inference (ParTI) framework [1,2] for analyzing functional trade-offs in single-cell and spatial omics data.

ParTI models gene expression variability within a cell type by capturing functional trade-offs - e.g., glycolysis vs. gluconeogenesis. The framework posits that cells lie along Pareto fronts, where improving one biological task inherently compromises another, forming a functional landscape represented as a polytope. Vertices of this polytope correspond to specialist cells optimized for distinct tasks, while generalists occupy interior regions balancing multiple functions.

To infer this structure, *Archetypal Analysis (AA)* models each cell as a convex combination of extremal points, called archetypes. These archetypes are constrained to lie within the convex hull of the data, ensuring interpretability and biological plausibility. In contrast to clustering methods that impose hard boundaries, AA preserves the continuous nature of gene expression variability and reveals functional trade-offs without artificial discretization.

`partipy` integrates with the scverse ecosystem, supports AnnData, and employs coreset-based optimization for scalability to millions of cells.

[1] Hart et al., *Nat Methods* (2015). https://doi.org/10.1038/nmeth.3254

[2] Adler et al., *Cell Systems* (2019). https://doi.org/10.1016/j.cels.2018.12.008

## Documentation

For further information and example tutorials, please check our [documentation](https://partipy.readthedocs.io).

## Installation

Since `partipy` is still in the beta stage and updated frequently, we recommend installing it directly from GitHub:

```
pip install git+https://github.com/saezlab/partipy.git
```

Alternatively, `partipy` can be installed from PyPI:

```
pip install partipy
```

## Questions & Issues

If you have any questions or issues, do not hesitate to open an [issue](https://github.com/saezlab/ParTIpy/issues).

## Citation

TBD
