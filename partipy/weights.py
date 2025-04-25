import numpy as np


def compute_bisquare_weights(R: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
    """
    Compute Tukey's bisquare (biweight) weights for robust estimation based on residual magnitudes.

    For each data point, the ℓ₁ norm of the residual vector is computed. A robust
    scale parameter is estimated using the median absolute deviation (MAD), and a
    cutoff threshold c is defined as c = 6 * median(||r_i||₁) + ε, where ε is a small
    constant to prevent division by zero. Residual norms are scaled by c, and weights
    are assigned using Tukey's bisquare function: points with scaled residuals < 1
    receive smoothly decreasing weights; those ≥ 1 receive weight 0.

    This redescending M-estimator effectively suppresses the influence of extreme
    outliers while preserving the influence of inliers.

    References
    ----------
    Eugster, M.J.A., Leisch, F. (2011).
    Weighted and robust archetypal analysis.
    Computational Statistics & Data Analysis, 55(3), 1215-1225.
    https://doi.org/10.1016/j.csda.2010.10.017

    Fox, J., Weisberg, 2013.
    Robust Regrssion.
    http://users.stat.umn.edu/~sandy/courses/8053/handouts/robust.pdf

    Parameters
    ----------
    R : np.ndarray
        Residual matrix of shape (n_samples, n_features), where each row corresponds
        to the residual vector of a data point.

    Returns
    -------
    W : np.ndarray
        Weight vector of shape (n_samples,), with values in [0, 1].
        Points with residual norms well below the threshold receive weight near 1;
        points beyond the cutoff are fully downweighted to 0.
    """
    l1_norm = np.abs(R).sum(axis=1)
    selection_vec = l1_norm > epsilon
    if np.sum(selection_vec) > 0:
        c = 6 * np.median(l1_norm[selection_vec])
    else:
        c = 6 * np.median(l1_norm)
    l1_norm /= c
    W = np.zeros_like(l1_norm)
    W[l1_norm < 1] = np.square(1 - np.square(l1_norm[l1_norm < 1]))
    return W


def compute_huber_weights(R: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
    """
    Compute Huber weights for robust estimation based on residual magnitudes.

    For each data point, the Euclidean norm (l2 norm) of the residual vector
    is computed. A robust scale parameter is estimated using the median absolute
    deviation (MAD), and the threshold parameter δ is set according to the
    standard Huber rule: δ = 1.345 * σ̂. Weights are then assigned using the
    Huber function: points with residual norms below δ receive full weight (1),
    while those above δ are downweighted proportionally to δ / ||r_i||.

    This weighting scheme achieves high statistical efficiency under Gaussian noise
    while maintaining robustness to outliers.

    Reference
    ---------
    Huber, P.J. (1964). Robust Estimation of a Location Parameter.
    The Annals of Mathematical Statistics, 35(1), 73-101.
    https://doi.org/10.1214/aoms/1177703732

    Fox, J., Weisberg, 2013.
    Robust Regrssion.
    http://users.stat.umn.edu/~sandy/courses/8053/handouts/robust.pdf

    Parameters
    ----------
    R : np.ndarray
        Residual matrix of shape (n_samples, n_features), where each row corresponds
        to the residual vector of a data point.

    Returns
    -------
    W : np.ndarray
        Weight vector of shape (n_samples,), with values in (0, 1].
        Weights decrease with increasing residual magnitude beyond the threshold δ.
    """
    epsilon = 0.1

    l2_norm = np.sqrt(np.sum(np.square(R), axis=1))
    selection_vec = l2_norm > epsilon
    if np.sum(selection_vec) > 0:
        sigma_hat = np.median(l2_norm[l2_norm > epsilon]) / 0.6745  # Consistent estimator for Gaussian noise
    else:
        sigma_hat = np.median(l2_norm) / 0.6745  # Consistent estimator for Gaussian noise
    delta = 1.345 * sigma_hat
    W = np.ones_like(l2_norm)
    mask = l2_norm > delta
    W[mask] = delta / l2_norm[mask]
    return W
