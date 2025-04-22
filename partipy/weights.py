import numpy as np


def compute_bisquare_weights(R: np.ndarray) -> np.ndarray:
    """
    Compute the weights using the residuals per data point.

    Parameters
    ----------
    R : np.ndarray
        The residual vector, with shape (n_samples, n_features).

    Returns
    -------
    W : np.ndarray
        Weight vector of shape (n_samples,), where each entry corresponds to the bisquare weight of a data point.
    """
    R_scaled_l1_norm = np.abs(R).sum(axis=1)
    c = 6 * np.median(R_scaled_l1_norm) + 1e-9
    R_scaled_l1_norm /= c
    W = np.zeros_like(R_scaled_l1_norm)
    W[R_scaled_l1_norm < 1] = (1 - (R_scaled_l1_norm[R_scaled_l1_norm < 1]) ** 2) ** 2
    return W


# TODO
def compute_huber_weights(R: np.ndarray) -> np.ndarray:
    """
    Compute the weights using the residuals per data point
    :param R: residual vector, with shape (n_samples, n_features)
    """
    raise NotImplementedError()
