import numpy as np


def simulate(
    n_samples: int,
    n_archetypes: int,
    n_dimensions: int,
    noise_std: float,
    seed: int = 42,
):
    """
    TODO: Write docstring
    ...
    """
    assert noise_std >= 0

    rng = np.random.default_rng(seed=seed)
    Z = rng.uniform(low=-1, high=1, size=(n_archetypes, n_dimensions))
    A = rng.exponential(scale=1, size=(n_samples, n_archetypes))
    A /= A.sum(axis=1, keepdims=True)
    X = A @ Z
    if noise_std > 0:
        X += rng.normal(loc=0, scale=noise_std, size=X.shape)

    assert np.all(np.isclose(A.sum(axis=1), 1))
    assert A.min() > 0

    return X, A, Z
