# allowed arguments
INIT_ALGS = ("uniform", "furthest_sum", "plus_plus")
WEIGHT_ALGS = (None, "bisquare", "huber")
OPTIM_ALGS = ("regularized_nnls", "projected_gradients", "frank_wolfe")
CORESET_ALGS = ("default", "lightweight_kmeans", "uniform")

# default arguments
DEFAULT_INIT: str = "plus_plus"
DEFAULT_WEIGHT = None
DEFAULT_OPTIM: str = "projected_gradients"

# constants
LAMBDA: float = 1_000.0
MIN_ITERATIONS: int = 10
