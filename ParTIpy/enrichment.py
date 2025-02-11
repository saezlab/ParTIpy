"""
Functions to calculate the genes enriched at an archetype
"""
import numpy as np


def euclidean_distance(x, archetypes):
    """
    Calculate the euclidian distance between the cells and the archetypes.

    Parameters
    ----------
    x : numpy.ndarray
       The cells PCA coordinates array of shape (n_samples, n_features)

    archetypes : numpy.ndarray
        The archetypes PCA coordinates array of shape (n_features, n_archetypes)

    Returns
    -------
    distances : numpy.ndarray
        Distance matrix of shape (n_archetypes, n_samples)
    """
    # Compute squares
    sx = np.sum(x**2, axis=1, keepdims=True)
    sa = np.sum(archetypes**2, axis=0, keepdims=True)

    # Calculate squared distances
    squared_distances = sx + sa - 2 * x.dot(archetypes)

    # Clip small negative values to zero
    squared_distances[np.isclose(squared_distances, 0)] = 0

    # Calculate Euclidian distance
    distances = np.sqrt(squared_distances)
    
    return distances.T


def rbf_kernel(euclidean_dist, mode = 'automatic', length_scale = None, data = None, archetypes = None):
     """
     Calculate the weights with an squared exponential kernel

     Parameters
     ----------
     euclidean_dist : numpy.ndarray
        euclidean distance of the cells to the archetypes.

     mode : str
        Either 'automatic' (calculate length scale as half of the median distance to the data centroid) or 'manual' (use provided length_scale).
    
     length_scale : float, optional
        length scale of the kernel (required if mode='manual').

     data : numpy.ndarray, optional
        PCs of the cells (required if mode='automatic').

     archetypes : numpy.ndarray, optional
        coordinates of the archetypes (required if mode='automatic').

     Returns
     -------
     weights : numpy.ndarray
        weights based on distance of cells from archetypes.

     length_scale : float (optional)
        The calculated length_scale (if mode='automatic').
    
     """
     if mode not in ['automatic', 'manual']:
        raise ValueError("Mode must be either 'automatic' or 'manual'.")
     
     if mode == 'automatic':
        if data is None or archetypes is None:
            raise ValueError("For 'automatic' mode, both 'data' and 'archetypes' must be provided.")
        centroid = np.mean(data, axis=0).reshape(1, -1)
        length_scale = np.median(euclidean_distance(centroid, archetypes)) / 2
     
     if mode == 'manual':
        if length_scale is None:
            raise ValueError("For 'manual' mode, 'length_scale' must be provided.")
     
     weights = np.exp(- euclidean_dist**2 / (2* length_scale**2))

     if mode == 'automatic':
        return weights, length_scale
     return weights

def weighted_expr(weights, expr):
    """
     Calculate a weighted pseudobulk per archetype.

     Parameters
     ----------
     weights : numpy.ndarray
         weights of the cells based on their distance to the archetypes

     expr : numpy.ndarray
         gene expression of the cells

    Returns
    -------
     pseudobulk : numpy.ndarray
         pseudobulk of the archetypes
     """
    pseudobulk = np.zeros((weights.shape[0], expr.shape[1]))
    
    for archetype in range(weights.shape[0]):
        # multiply weights with cells expression
        weighted_expr = weights[archetype, :, None] * expr  
        # compute weighted sum
        pseudobulk[archetype, :] = weighted_expr.sum(axis=0) / sum(weights[archetype, :])

    return pseudobulk