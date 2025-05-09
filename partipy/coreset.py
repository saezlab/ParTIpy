import numpy as np
from scipy.spatial.distance import cdist

REPLACE = True


def construct_coreset(X: np.ndarray, coreset_size: int, seed: int):
    """Construct coreset"""
    n_samples = X.shape[0]

    sq_dists = np.square(cdist(XA=X, XB=X.mean(axis=0, keepdims=True)).flatten())
    probs = sq_dists / sq_dists.sum()

    rng = np.random.default_rng(seed=seed)
    # NOTE: In the original implementation they sample WITH replacement
    # https://github.com/smair/archetypalanalysis-coreset/blob/6b34fce70ec1c47c9938d1f7887c506a131c94f6/code/coresets.py#L79
    coreset_indices = rng.choice(a=n_samples, size=coreset_size, p=probs, replace=REPLACE)

    weights = 1 / (probs[coreset_indices] * coreset_size)
    weights_root = np.sqrt(weights)
    weights_root = weights_root.astype(np.float32)

    return coreset_indices, weights_root


def construct_lightweight_coreset(X: np.ndarray, coreset_size: int, seed: int):  # pragma: no cover
    """Construct k-means clustering via lightweight coresets (Bachem et al. (2018))"""
    n_samples = X.shape[0]

    sq_dists = np.square(cdist(XA=X, XB=X.mean(axis=0, keepdims=True)).flatten())
    probs = 0.5 * (1 / n_samples) + 0.5 * (sq_dists / sq_dists.sum())

    rng = np.random.default_rng(seed=seed)
    coreset_indices = rng.choice(a=n_samples, size=coreset_size, p=probs, replace=REPLACE)

    weights = 1 / (probs[coreset_indices] * coreset_size)
    weights_root = np.sqrt(weights)
    weights_root = weights_root.astype(np.float32)

    return coreset_indices, weights_root


# NOTE: This is not really a coreset, but I rather use this for testing purposes
def construct_uniform_coreset(X: np.ndarray, coreset_size: int, seed: int):  # pragma: no cover
    """Construct mock coreset by uniform sampling"""
    n_samples = X.shape[0]
    rng = np.random.default_rng(seed=seed)
    coreset_indices = rng.choice(a=n_samples, size=coreset_size, replace=REPLACE)
    weights_root = np.ones(shape=coreset_size, dtype=np.float32)
    return coreset_indices, weights_root
