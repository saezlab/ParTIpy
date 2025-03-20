import numpy as np


def simulate(
    n_samples: int,
    n_archetypes: int,
    n_dimensions: int,
    noise_std: float,
    seed: int = 42,
):
    """
    Simulate synthetic data for benchmarking archetypal analysis on datasets with known ground truth (archetypes Z and coefficients A).

    Archetypes (Z) are randomly sampled from the range [-1, 1]. The coefficients in A, which map each data point to a convex combination of archetypes, are sampled from an exponential distribution and normalized to have a row sum of 1.
    The coordinates of the data points (X) are then calculated as X = A @ Z. Optionally, Gaussian noise can be added to simulate real-world noise.

    Parameters
    ----------
    n_samples : int
        Number of data points (samples) to generate.
    n_archetypes : int
        Number of archetypes to use for generating the data.
    n_dimensions : int
        Number of dimensions (features) for each data point and archetype.
    noise_std : float
        Standard deviation of Gaussian noise added to the data. Set to 0 for no noise.
    seed : int, optional (default=42)
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray
        Generated data matrix of shape (n_samples, n_dimensions).
    A : np.ndarray
        Coefficient matrix of shape (n_samples, n_archetypes), representing the convex combinations
        of archetypes for each data point.
    Z : np.ndarray
        Archetype matrix of shape (n_archetypes, n_dimensions), representing the archetypes.
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
