ParTIpy (Pareto Task Inference in Python)
===================================

**partipy** offers an efficient and user-friendly implementation of the **Pareto Task Inference (ParTI)** framework for single-cell analysis within the scverse ecosystem.

**ParTI** models gene expression variability within a cell type by capturing **functional trade-offs**—e.g., glycolysis vs. gluconeogenesis. The framework assumes that cells exist along **Pareto fronts**, where improving one task necessarily compromises another, forming a functional landscape represented as a polytope. Its vertices correspond to specialist cells optimized for distinct tasks, while generalists balance multiple functions near the center.

To infer this structure, **Archetypal Analysis (AA)** identifies the polytope's vertices (**archetypes**) as extremal points in gene expression space. Each cell is expressed as a convex combination of these archetypes, preserving the continuous nature of biological variation.

Compared to traditional clustering, **AA** avoids artificial discretization and provides an interpretable representation of functional trade-offs, linking gene expression patterns to biological processes.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Main

   installation
   release_notes

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Vignettes

   notebooks/quickstart
   notebooks/cross_condition_lupus
   notebooks/crosstalk
