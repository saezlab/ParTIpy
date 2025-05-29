# allowed arguments
INIT_ALGS = ("uniform", "furthest_sum", "plus_plus")
WEIGHT_ALGS = (None, "bisquare", "huber")
OPTIM_ALGS = ("regularized_nnls", "projected_gradients", "frank_wolfe")
CORESET_ALGS = ("default", "lightweight_kmeans", "uniform")

# default arguments
DEFAULT_INIT: str = "plus_plus"
DEFAULT_WEIGHT: None | str = None
DEFAULT_OPTIM: str = "projected_gradients"

# constants
LAMBDA: float = 1_000.0
MIN_ITERATIONS: int = 10

# other defaults
DEFAULT_MAX_ITER: int = 500
DEFAULT_REL_TOL: float = 1e-4
DEFAULT_OBSM_KEY: str = "X_pca"

# default seed for reproducibility
DEFAULT_SEED: int = 42
