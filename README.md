# ParTIpy: Pareto Task Inference in Python <img src="https://raw.githubusercontent.com/saezlab/ParTIpy/main/docs/_static/images/logo_blue.png?raw=true" align="right" height="125">
<!-- badges: start -->
[![codecov](https://codecov.io/gh/saezlab/partipy/branch/main/graph/badge.svg)](https://codecov.io/gh/saezlab/partipy)
[![Documentation Status](https://readthedocs.org/projects/partipy/badge/?version=latest)](https://partipy.readthedocs.io/en/latest/?badge=latest)
[![GitHub issues](https://img.shields.io/github/issues/saezlab/partipy.svg)](https://github.com/saezlab/partipy/issues/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/saezlab/ParTIpy/main.svg)](https://results.pre-commit.ci/latest/github/saezlab/ParTIpy/main)
<!-- badges: end -->

`partipy` (Pareto Task Inference in Python) provides a scalable and user-friendly implementation of the Pareto Task Inference (ParTI) framework ([1](https://doi.org/10.1016/j.cels.2018.12.008), [2](https://doi.org/10.1038/nmeth.3254), [3](https://doi.org/10.1016/j.cels.2018.12.008), [4](https://doi.org/10.1016/j.cels.2018.12.008)) for analyzing functional trade-offs in biological data, particularly in high-throughput single-cell and spatial omics data.

ParTI models gene expression variability within a cell type by capturing functional trade-offs - e.g., glycolysis vs. gluconeogenesis. The framework posits that cells lie along Pareto fronts, where improving one biological task inherently compromises another, forming a functional landscape represented as a polytope. Vertices of this polytope correspond to specialist cells optimized for distinct tasks, while generalists occupy interior regions balancing multiple functions.

To infer this structure, **archetypal analysis** models each cell as a convex combination of extremal points, called archetypes. These archetypes are constrained to lie within the convex hull of the data, ensuring interpretability and biological plausibility. In contrast to clustering methods that impose hard boundaries, archetypal analysis preserves the continuous nature of gene expression variability and reveals functional trade-offs without artificial discretization.

`partipy` integrates with the scverse ecosystem, supports AnnData, and employs coreset-based optimization for scalability to millions of cells.

## Documentation

For detailed information and example tutorials, please refer to our [documentation](https://partipy.readthedocs.io). Key resources include:

- [Quickstart Vignette](https://partipy.readthedocs.io/en/latest/notebooks/quickstart.html)
- [Archetypal Analysis Vignette](https://partipy.readthedocs.io/en/latest/notebooks/archetypal_analysis.html)

For a deeper dive into the mathematical foundations of archetypal analysis and the implementation of various initialization and optimization algorithms, see the [methods section](https://github.com/saezlab/ParTIpy/blob/main/docs/methods/methods.pdf).

## Installation

You need to have Python 3.10 or newer installed on your system.

There are several alternative options to install `partipy`:

1. Install the latest stable release from [PyPI](https://pypi.org/project/partipy/) with minimal dependencies:

```
pip install partipy
```

2. Install the latest stable full release from [PyPI](https://pypi.org/project/partipy/) with extra dependencies:

```
pip install partipy[extra]
```

3. Install the latest development version:

```
pip install git+https://github.com/saezlab/partipy.git
```

## Release Notes

See the [changelog](https://partipy.readthedocs.io/en/latest/changelog.html).

## Questions & Issues

If you have any questions or issues, do not hesitate to open an [issue](https://github.com/saezlab/ParTIpy/issues).

## Workflow Overview

<img src="https://raw.githubusercontent.com/saezlab/ParTIpy/main/docs/_static/images/partipy_overview.png" alt="ParTIpy Overview" width="600"/>

## Citation

TBD
