from docrep import DocstringProcessor

from .const import (
    CORESET_ALGS,
    DEFAULT_INIT,
    DEFAULT_MAX_ITER,
    DEFAULT_OBSM_KEY,
    DEFAULT_OPTIM,
    DEFAULT_REL_TOL,
    DEFAULT_SEED,
    DEFAULT_WEIGHT,
    INIT_ALGS,
    OPTIM_ALGS,
    WEIGHT_ALGS,
)

_verbose = """\
verbose : bool, default `False`
    Whether to display progress messages and additional execution details."""

_seed = f"""\
seed : int, default `{DEFAULT_SEED}`
    Random seed to use for reproducible results."""

_optim = f"""\
optim : {{{", ".join(f'`"{alg}"`' for alg in OPTIM_ALGS)}}}, default `"{DEFAULT_OPTIM}"`
    Optimization algorithm to use. Options are:

    - "projected_gradients": Projected gradient descent (PCHA) :cite:`morupArchetypalAnalysisMachine2012`.
    - "frank_wolfe": Frank-Wolfe algorithm :cite:`bauckhageArchetypalAnalysisAutoencoder2015`.
    - "regularized_nnls": Regularized non-negative least squares :cite:`cutlerArchetypalAnalysis1994`.

    See :data:`partipy.const.OPTIM_ALGS` for all available options.
"""

_init = f"""\
init : {{{", ".join(f'`"{alg}"`' for alg in INIT_ALGS)}}}, default `"{DEFAULT_INIT}"`
    Initialization method for the archetypes. Options are:

    - "plus_plus": Archetypal++ initialization :cite:`morupArchetypalAnalysisMachine2012`.
    - "furthest_sum": Utilizes the furthest sum algorithm :cite:`morupArchetypalAnalysisMachine2012`.
    - "uniform": Random initialization.

    See :data:`partipy.const.INIT_ALGS` for all available options.
"""

_weight = f"""\
weight : {{{", ".join(f'`"{alg}"`' if alg is not None else "`None`" for alg in WEIGHT_ALGS)}}}, default `{DEFAULT_WEIGHT}`
    Weighting scheme for robust archetypal analysis. Options are:

    - None: No weighting (standard archetypal analysis).
    - "bisquare": Bisquare weighting for robust estimation.
    - "huber": Huber weighting for robust estimation.

    See :data:`partipy.const.WEIGHT_ALGS` for all available options.
"""

_coreset = f"""\
coreset : {{{", ".join(f'`"{alg}"`' for alg in CORESET_ALGS)}}}, default `"default"`
    Coreset algorithm to use for data reduction. Options are:

    - "default": Standard coreset construction.
    - "lightweight_kmeans": Lightweight k-means based coreset.
    - "uniform": Uniform sampling based coreset.

    See :data:`partipy.const.CORESET_ALGS` for all available options.
"""

# Core parameters
_n_archetypes = """\
n_archetypes : int
    Number of archetypes to compute."""

# Data parameters
_obsm_key = f"""\
obsm_key : str, default `"{DEFAULT_OBSM_KEY}"`
    Key in `adata.obsm` containing the data matrix to use for archetypal analysis."""

# Computational parameters
_max_iter = f"""\
max_iter : int, default `{DEFAULT_MAX_ITER}`
    Maximum number of iterations for the optimization algorithm."""

_rel_tol = f"""\
rel_tol : float, default `{DEFAULT_REL_TOL}`
    Tolerance for convergence of the optimization algorithm."""

# Create the docstring processor with all parameters
docs = DocstringProcessor(
    n_archetypes=_n_archetypes,
    init=_init,
    optim=_optim,
    weight=_weight,
    coreset=_coreset,
    obsm_key=_obsm_key,
    max_iter=_max_iter,
    rel_tol=_rel_tol,
    seed=_seed,
    verbose=_verbose,
)
