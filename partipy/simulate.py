import numpy as np
from scipy.spatial import ConvexHull, QhullError, distance_matrix

from ._docs import docs


def _generate_extreme_archetypes(n_archetypes, n_dimensions, rng, distribution="normal", max_attempts=100):
    """
    Generate extreme archetypes, ensuring they are vertices of a convex hull
    and maximally separated from each other.

    Parameters
    ----------
    n_archetypes : int
        Number of archetypes to generate.
    n_dimensions : int
        Number of dimensions for each archetype.
    rng : numpy.random.Generator
        Random number generator.
    distribution : str, default "normal"
        Distribution to sample from, either "normal" or "uniform".
    max_attempts : int, default 100
        Maximum number of attempts to generate valid archetypes. If the maximum number of attempts is exceeded
        without generating enough archetypes, a ValueError is raised.

    Returns
    -------
    np.ndarray
        Array of shape (n_archetypes, n_dimensions) containing the archetypes.
    """
    # Determine minimum number of points needed based on dimensions
    min_points_needed = n_dimensions + 1

    for attempt in range(max_attempts):
        # Increase candidate count each attempt, and ensure we have enough points
        n_candidates = max(n_archetypes * (attempt + 2), min_points_needed)

        # Generate candidate points
        if distribution == "normal":
            candidates = rng.normal(loc=0, scale=1, size=(n_candidates, n_dimensions))
        elif distribution == "uniform":
            candidates = rng.uniform(low=-1, high=1, size=(n_candidates, n_dimensions))
        else:
            raise NotImplementedError(f"{distribution} not implemented")

        try:
            # Compute convex hull
            hull = ConvexHull(candidates)

            # Get the vertices of the convex hull
            vertices = candidates[hull.vertices]

            # If we have more vertices than needed, select the most distant subset
            if len(vertices) > n_archetypes:
                # Select a subset with maximum distance between points
                selected_vertices = _select_distant_vertices(vertices, n_archetypes, rng)
                return selected_vertices
            # If we have exactly the right number, return them
            elif len(vertices) == n_archetypes:
                return vertices
            # Otherwise, try again with more candidates

        except QhullError as e:
            error_msg = str(e)
            print("QhullError")
            # Check if the error is about not having enough points
            if "not enough points" in error_msg:
                continue
            # For other QhullError types, just try again
            print(error_msg)
            continue
    raise ValueError("Increase the max number of attempts")


def _select_distant_vertices(vertices, n_select, rng):
    """
    Select n_select vertices from the given set, maximizing the minimum distance between any two selected vertices.
    Uses the maxmin (farthest point) sampling algorithm.

    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (n_vertices, n_dimensions) containing the vertices.
    n_select : int
        Number of vertices to select.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of shape (n_select, n_dimensions) containing the selected vertices.
    """
    if len(vertices) <= n_select:
        return vertices

    # Calculate pairwise distance matrix
    dist_matrix = distance_matrix(vertices, vertices)

    # Set diagonal to infinity (distance to self)
    np.fill_diagonal(dist_matrix, np.inf)

    # Start with a random vertex
    selected_indices = [rng.integers(0, len(vertices))]

    # Greedily add vertices that maximize the minimum distance
    while len(selected_indices) < n_select:
        # Find the minimum distance from each remaining point to any selected point
        remaining_indices = [i for i in range(len(vertices)) if i not in selected_indices]
        min_distances = np.min(dist_matrix[remaining_indices][:, selected_indices], axis=1)

        # Select the point with maximum minimum distance
        max_idx = np.argmax(min_distances)
        selected_indices.append(remaining_indices[max_idx])

    return vertices[selected_indices]


@docs.dedent
def simulate_archetypes(
    n_samples: int,
    n_archetypes: int,
    n_dimensions: int,
    noise_std: float,
    max_attempts: int = 100,
    seed: int = 42,
):
    """
    Simulate synthetic data for benchmarking archetypal analysis on datasets with known ground truth (archetypes Z and coefficients A).

    Candidate archetypes are randomly sampled from the range [-1, 1].
    Then, we compute the convex hull of these candiate archetypes to ensure that in the final set of archetypes we only include points that are vertices in the convex hull
    The coefficients in A, which map each data point to a convex combination of archetypes, are sampled from an exponential distribution and normalized to have a row sum of 1.
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
    %(seed)s

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
    Z = _generate_extreme_archetypes(n_archetypes, n_dimensions, rng, max_attempts=max_attempts)
    A = rng.exponential(scale=1, size=(n_samples, n_archetypes))
    A /= A.sum(axis=1, keepdims=True)
    X = A @ Z
    if noise_std > 0:
        X += rng.normal(loc=0, scale=noise_std, size=X.shape)

    assert np.all(np.isclose(A.sum(axis=1), 1))
    assert A.min() > 0

    return X, A, Z
