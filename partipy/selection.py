import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _invert_SPD_mtx(mtx):
    # Cholesky factorization
    c, lower = cho_factor(mtx)

    # Inverse using Cholesky
    identity = np.eye(mtx.shape[0])
    cov_inv = cho_solve((c, lower), identity)
    return cov_inv


def compute_IC_approx(X, X_tilde, n_archetypes):
    """
    Compute information-theorectic criterion to access goodness-of-fit

    Reference: Suleman, A., 2017. Validation of archetypal analysis, 2017 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE)pp. 1-6. https://doi.org/10.1109/FUZZ-IEEE.2017.8015385
    (see Equation 13)

    Parameters
    ----------
    X: np.ndarray
        Data matrix
    X_tilde: np.ndarray
        Approximation of data matrix by archetypal analysis, i.e. X_tilde = A B X = A Z
    n_archetypes : int
        number of archetypes.

    Returns
    -------
    IC: float
    """
    assert np.all(X.shape == X_tilde.shape)
    n_samples, n_features = X.shape
    X_cov = np.cov(X, rowvar=False, bias=True)
    X_tilde_cov = np.cov(X_tilde, rowvar=False, bias=True)
    assert np.all(X_cov.shape == np.array(n_features))
    assert np.all(X_tilde_cov.shape == np.array(n_features))
    X_cov_inv = _invert_SPD_mtx(X_cov)
    IC = np.log(np.square(np.linalg.norm(X - X_tilde)) / (n_samples * n_features)) + 2 * (
        (2 * n_archetypes - 1) / np.trace(X_tilde_cov @ X_cov_inv)
    )
    return IC


def compute_IC(X, X_tilde, n_archetypes):
    """
    Compute information-theorectic criterion to access goodness-of-fit

    Reference: Suleman, A., 2017. Validation of archetypal analysis, 2017 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE)pp. 1-6. https://doi.org/10.1109/FUZZ-IEEE.2017.8015385
    (see Equation 12)

    Parameters
    ----------
    X: np.ndarray
        Data matrix
    X_tilde: np.ndarray
        Approximation of data matrix by archetypal analysis, i.e. X_tilde = A B X = A Z
    n_archetypes : int
        number of archetypes.

    Returns
    -------
    IC: float
    """
    assert np.all(X.shape == X_tilde.shape)
    n_samples, n_features = X.shape
    X_cov = np.cov(X, rowvar=False, bias=True)
    X_tilde_cov = np.cov(X_tilde, rowvar=False, bias=True)
    assert np.all(X_cov.shape == np.array(n_features))
    assert np.all(X_tilde_cov.shape == np.array(n_features))
    X_cov_inv = _invert_SPD_mtx(X_cov)
    K_mu = n_features * (n_archetypes - 1)
    K_beta = n_archetypes * (n_features - 1)
    IC = np.log(np.square(np.linalg.norm(X - X_tilde)) / (n_samples * n_features)) + 2 * (
        (K_mu + K_beta + 1) / (n_features * np.trace(X_tilde_cov @ X_cov_inv))
    )
    return IC
