ParTIpy (Pareto Task Inference in Python)
===================================

**partipy**  provides a scalable and user-friendly implementation of the Pareto Task Inference (ParTI) framework for analyzing functional trade-offs in single-cell and spatial omics data, fully integrated within the scverse ecosystem.

**ParTI** models gene expression variability within a cell type by capturing functional trade-offs—for example, glycolysis versus gluconeogenesis. The framework posits that cells lie along Pareto fronts, where improving one biological task inherently compromises another. This forms a functional landscape represented as a polytope, whose vertices correspond to specialist cells optimized for distinct tasks, while generalists reside in the interior, balancing multiple functions.

To infer this structure, Archetypal Analysis (AA) represents each cell as a convex combination of archetypes, which are constrained to lie within the convex hull of the data. This approach preserves the continuous nature of gene expression variability and ensures interpretability and biological plausibility. In contrast to clustering-based methods that impose hard boundaries, AA captures functional diversity without artificial discretization, providing a principled, task-oriented decomposition of gene expression patterns.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Main

   installation
   release_notes
   api

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Vignettes

   notebooks/quickstart
   notebooks/cross_condition_lupus
   notebooks/spatial
   notebooks/crosstalk
   notebooks/archetypal_analysis
