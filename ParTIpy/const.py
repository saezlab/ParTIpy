# allowed arguments
INIT_ALGS = ("random", "furthest_sum")
WEIGHT_ALGS = (None, "bisquare")
OPTIM_ALGS = ("regularized_nnls", "projected_gradients", "frank_wolfe")

# default arguments
DEFAULT_INIT: str = "furthest_sum"
DEFAULT_WEIGHT = None
DEFAULT_OPTIM: str = "projected_gradients"

# constants
LAMBDA: float = 1_000.0
