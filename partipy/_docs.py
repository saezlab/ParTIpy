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

    - `"projected_gradients"`: Projected gradient descent (PCHA) :cite:`morupArchetypalAnalysisMachine2012`.
    - `"frank_wolfe"`: Frank-Wolfe algorithm :cite:`bauckhageArchetypalAnalysisAutoencoder2015`.
    - `"regularized_nnls"`: Regularized non-negative least squares :cite:`Cutler1994`.

    See `partipy.const.OPTIM_ALGS` for all available options.
"""

_init = f"""\
init : {{{", ".join(f'`"{alg}"`' for alg in INIT_ALGS)}}}, default `"{DEFAULT_INIT}"`
    Initialization method for the archetypes. Options are:

    - `"plus_plus"`: Archetypal++ initialization :cite:`mairArchetypalAnalysisRethinking2024`.
    - `"furthest_sum"`: Utilizes the furthest sum algorithm :cite:`morupArchetypalAnalysisMachine2012`.
    - `"uniform"`: Random initialization.

    See `partipy.const.INIT_ALGS` for all available options.
"""

_weight = f"""\
weight : {{{", ".join(f'`"{alg}"`' if alg is not None else "`None`" for alg in WEIGHT_ALGS)}}}, default `{DEFAULT_WEIGHT}`
    Weighting scheme for robust archetypal analysis, based on :cite:`eugsterWeightedRobustArchetypal2011`. Options are:

    - `None`: No weighting (standard archetypal analysis).
    - `"bisquare"`: Bisquare weighting for robust estimation.
    - `"huber"`: Huber weighting for robust estimation.

    See `partipy.const.WEIGHT_ALGS` for all available options.
"""

_coreset_algorithm = f"""\
coreset_algorithm : {{{", ".join(f'`"{alg}"`' for alg in CORESET_ALGS)}}}, default `None`
    Coreset algorithm to use for data reduction, based on :cite:`mairCoresetsArchetypalAnalysis2019`. Options are:

    - `None`: No coreset is used.
    - `"standard"`: Coreset construction for archetypal analysis :cite:`mairCoresetsArchetypalAnalysis2019`. Recommended option if data reduction is needed.
    - `"lightweight_kmeans"`: Lightweight coreset for k-means clustering :cite:`lucicStrongCoresetsHard2016`.
    - `"uniform"`: Coreset based on uniform sampling.

    See `partipy.const.CORESET_ALGS` for all available options.
"""

_delta = """\
delta: float, default: `0.0`
    Parameter that relaxes the constraint that B must be convex combination of the data points.
    Must be in the interval [0, 1).
"""

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
    coreset_algorithm=_coreset_algorithm,
    delta=_delta,
    obsm_key=_obsm_key,
    max_iter=_max_iter,
    rel_tol=_rel_tol,
    seed=_seed,
    verbose=_verbose,
)
