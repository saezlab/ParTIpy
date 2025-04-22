import time

import numpy as np
import pytest
from partipy.simulate import simulate_archetypes
from scipy.optimize import nnls
from scipy.spatial import ConvexHull

N_SAMPLES = 1_000
MAX_TIME_IN_SEC = 5.0


def is_convex_combination(Z, arch_idx, tol=1e-6):
    """Check if archetype arch_idx is a convex combination of other archetypes"""
    # Get all archetypes except the current one
    other_archetypes = np.delete(Z, arch_idx, axis=0)
    target = Z[arch_idx, :]

    # Pad the data to enforce sum(weights) = 1 constraint
    # Add a row of 1s (for sum constraint) and a row of 0s (for padding)
    A_padded = np.vstack([other_archetypes.T, np.ones(other_archetypes.shape[0]), np.zeros(other_archetypes.shape[0])])

    # Pad the target with 1 (for sum constraint) and 0 (for padding)
    b_padded = np.concatenate([target, [1.0, 0.0]])

    # Solve NNLS problem
    weights, residual = nnls(A_padded, b_padded)

    # Calculate reconstruction error
    reconstructed = other_archetypes.T @ weights
    error = np.linalg.norm(reconstructed - target)

    # Check if it's a convex combination (small error and weights sum to ~1)
    return error < tol and np.abs(1 - weights.sum()) < tol


@pytest.mark.parametrize("n_archetypes", list(range(2, 16)))
@pytest.mark.parametrize("n_dimensions", list(range(2, 8)))
def test_output_shapes(n_archetypes, n_dimensions):
    """Test that the output shapes match the expected dimensions."""
    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES, n_archetypes=n_archetypes, n_dimensions=n_dimensions, noise_std=0.0, seed=111
    )

    assert X.shape == (N_SAMPLES, n_dimensions), "X shape is incorrect"
    assert A.shape == (N_SAMPLES, n_archetypes), "A shape is incorrect"
    assert Z.shape == (n_archetypes, n_dimensions), "Z shape is incorrect"


@pytest.mark.parametrize("n_archetypes", list(range(2, 16)))
@pytest.mark.parametrize("n_dimensions", list(range(2, 8)))
def test_stochastic_matrix_properties(n_archetypes, n_dimensions):
    """Test that A is a valid stochastic matrix (rows sum to 1, all values >= 0)."""
    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES, n_archetypes=n_archetypes, n_dimensions=n_dimensions, noise_std=0.0, seed=111
    )

    assert np.allclose(A.sum(axis=1), 1.0), "Rows in A do not sum to 1"
    assert np.all(A >= 0), "A contains negative values"


@pytest.mark.parametrize("n_archetypes", list(range(2, 16)))
@pytest.mark.parametrize("n_dimensions", list(range(2, 8)))
def test_reconstruction_accuracy(n_archetypes, n_dimensions):
    """Test that X = A @ Z holds (no noise case)."""
    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES, n_archetypes=n_archetypes, n_dimensions=n_dimensions, noise_std=0.0, seed=111
    )

    assert np.allclose(X, A @ Z), "X != A @ Z, reconstruction failed"


@pytest.mark.parametrize("n_archetypes", list(range(2, 16)))
@pytest.mark.parametrize("n_dimensions", list(range(2, 8)))
def test_archetypes_outside_convex_hull(n_archetypes, n_dimensions):
    """
    Test that archetypes are outside the convex hull of X.
    """
    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES, n_archetypes=n_archetypes, n_dimensions=n_dimensions, noise_std=0.0, seed=111
    )
    hull = ConvexHull(X, qhull_options="QJ")

    # Each equation defines a half-space: Ax + b <= 0
    A_hull, b_hull = hull.equations[:, :-1], hull.equations[:, -1]

    for arch_idx, z in enumerate(Z):
        if np.all(A_hull @ z + b_hull <= 0):  # If all inequalities hold, z is inside the hull
            pytest.fail(f"Archetype {arch_idx} is inside the convex hull")


@pytest.mark.parametrize("n_archetypes", list(range(3, 16)))
@pytest.mark.parametrize("n_dimensions", list(range(2, 8)))
def test_archetype_non_redundancy(n_archetypes, n_dimensions):
    """Test that no archetype is a convex combination of other archetypes."""
    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES, n_archetypes=n_archetypes, n_dimensions=n_dimensions, noise_std=0.0, seed=111
    )
    for arch_idx in range(n_archetypes):
        if is_convex_combination(Z, arch_idx):
            pytest.fail(f"Archetype {arch_idx} is a convex combination of other archetypes")


@pytest.mark.parametrize("n_archetypes", list(range(2, 16)))
@pytest.mark.parametrize("n_dimensions", list(range(2, 20)))
def test_simulation_performance(n_archetypes, n_dimensions):
    """Test that simulation runs in reasonable time."""
    start_time = time.time()

    simulate_archetypes(
        n_samples=N_SAMPLES,  # Smaller sample for speed test
        n_archetypes=n_archetypes,
        n_dimensions=n_dimensions,
        noise_std=0.0,
        seed=111,
    )

    elapsed = time.time() - start_time
    assert elapsed < MAX_TIME_IN_SEC, f"Simulation took too long: {elapsed:.2f} seconds"
