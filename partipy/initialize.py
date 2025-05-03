import numpy as np
from scipy.spatial.distance import cdist

from .optim import _compute_A_projected_gradients


def _construct_B_from_indices(X: np.ndarray, indices: list) -> np.ndarray:
    X = X.astype(np.float32)
    n_samples, _ = X.shape
    B = np.zeros((len(indices), n_samples), dtype=np.float32)
    for arch_idx in range(len(indices)):
        B[arch_idx, indices[arch_idx]] = 1.0
    B = np.ascontiguousarray(B, dtype=np.float32)
    return B


def _init_A(n_samples: int, n_archetypes: int, seed: int, epsilon: float = 1e-9) -> np.ndarray:
    rng = np.random.default_rng(seed)  # Use a fixed seed
    A = -np.log(rng.random((n_samples, n_archetypes), dtype=np.float32) + epsilon)
    A /= np.sum(A, axis=1, keepdims=True)
    A = np.ascontiguousarray(A, dtype=np.float32)
    return A


def _init_uniform(
    X: np.ndarray, n_archetypes: int, seed: int = 42, return_indices: bool = False
) -> np.ndarray | tuple[np.ndarray, list[int]]:
    """Random selection of data points from X to form initial archetypes.

    Parameters
    ----------
    X: numpy 2d-array
        Data matrix with shape (n_samples x n_features).

    n_archetypes: int
        Number of candidate archetypes to extract.

    seed: int
        Seed for reproducibility

    Returns
    -------
    B : numpy 2d-array
        Matrix B with shape (n_archetypes, n_samples).
    """
    assert n_archetypes >= 2
    X = np.ascontiguousarray(X, dtype=np.float32)
    n_samples, _ = X.shape
    rng = np.random.default_rng(seed=seed)
    selected_indices = list(rng.choice(a=n_samples, size=n_archetypes, replace=False))
    B = _construct_B_from_indices(X=X, indices=selected_indices)
    if return_indices:
        return B, selected_indices
    else:
        return B


def _init_furthest_sum(
    X: np.ndarray, n_archetypes: int, seed: int = 42, return_indices: bool = False
) -> np.ndarray | tuple[np.ndarray, list[int]]:
    """Furthest sum initialization for archetypes.

    Reference: M. Mørup and L. K. Hansen, “Archetypal analysis for machine learning and data mining,” Neurocomputing, vol. 80, pp. 54-63, Mar. 2012, doi: https://doi.org/10.1016/j.neucom.2011.06.033.

    Parameters
    ----------
    X: numpy 2d-array
        Data matrix with shape (n_samples x n_features).

    n_archetypes: int
        Number of candidate archetypes to extract.

    seed: int
        Seed for reproducibility

    Returns
    -------
    B : numpy 2d-array
        Matrix B with shape (n_archetypes, n_samples).
    """
    assert n_archetypes >= 2
    X = np.ascontiguousarray(X, dtype=np.float32)
    n_samples, _ = X.shape
    rng = np.random.default_rng(seed=seed)

    first_idx = rng.choice(n_samples, size=1)[0]
    all_indices = np.arange(n_samples, dtype=int)
    selected_indices = [first_idx]

    # we replace the first 10 archetypes, see in the orginal implementation
    # https://github.com/ulfaslak/py_pcha/blob/f398d28c6a28c4a8121cb75443b221fc58fc10e4/py_pcha/furthest_sum.py#L46
    for iter in range(1, n_archetypes + 11):
        if iter > (n_archetypes - 1):
            selected_indices = selected_indices[1:]
        Z = X[selected_indices, :]
        distances = cdist(Z, X, metric="euclidean")
        mean_distances = distances.mean(axis=0)
        zero_distances = distances.min(axis=0)
        allowed_indices = all_indices[zero_distances > 0]
        selected_indices.append(allowed_indices[np.argmax(mean_distances[zero_distances > 0])])

    B = _construct_B_from_indices(X=X, indices=selected_indices)
    if return_indices:
        return B, selected_indices
    else:
        return B


def _init_plus_plus(
    X: np.ndarray,
    n_archetypes: int,
    seed: int = 42,
    return_indices: bool = False,
    epsilon: float = 1e-9,
) -> np.ndarray | tuple[np.ndarray, list[int]]:
    """Archetypal++ initialization for archetypes.

    Reference: Mair, S., Sjölund, J., 2024. Archetypal Analysis++: Rethinking the Initialization Strategy. doi: https://doi.org/10.48550/arXiv.2301.13748


    Parameters
    ----------
    X: numpy 2d-array
        Data matrix with shape (n_samples x n_features).

    n_archetypes: int
        Number of candidate archetypes to extract.

    seed: int
        Seed for reproducibility

    Returns
    -------
    B : numpy 2d-array
        Matrix B with shape (n_archetypes, n_samples).
    """
    assert n_archetypes >= 2
    X = X = np.ascontiguousarray(X, dtype=np.float32)
    n_samples, _ = X.shape
    rng = np.random.default_rng(seed=seed)

    first_idx = rng.choice(n_samples, size=1)[0]
    selected_indices = [first_idx]

    for iter in range(1, n_archetypes):
        Z = X[selected_indices, :]
        if iter == 1:
            A = np.ones((n_samples, 1))
        else:
            A = _init_A(n_samples=n_samples, n_archetypes=len(selected_indices), seed=seed)
            A = _compute_A_projected_gradients(X=X, Z=Z, A=A)
        p = np.sum(np.square(X - np.dot(A, Z)), axis=1) + epsilon
        p /= p.sum()
        selected_indices.append(rng.choice(a=n_samples, size=1, p=p)[0])

    B = _construct_B_from_indices(X=X, indices=selected_indices)
    if return_indices:
        return B, selected_indices
    else:
        return B
