import numpy as np
import pytest
from partipy.optim import _compute_A_frank_wolfe, _compute_A_projected_gradients, _compute_A_regularized_nnls
from partipy.simulate import simulate_archetypes
from partipy.utils import compute_relative_rowwise_l2_distance

compute_a_functions = [_compute_A_regularized_nnls, _compute_A_projected_gradients, _compute_A_frank_wolfe]


@pytest.mark.parametrize(
    "n_archetypes, n_dimensions",
    [(n_a, n_d) for n_a in range(3, 8, 2) for n_d in range(4, 13, 2) if n_a <= n_d],
)
@pytest.mark.parametrize("n_samples", [100, 1_000, 10_000])
@pytest.mark.parametrize("compute_a_algorithm", compute_a_functions)
def test_that_archetypal_weights_can_be_identified(
    n_archetypes: int,
    n_dimensions: int,
    n_samples: int,
    compute_a_algorithm,
) -> None:
    X, A, Z = simulate_archetypes(
        n_samples=n_samples,
        n_archetypes=n_archetypes,
        n_dimensions=n_dimensions,
        noise_std=0.0,
        seed=42,
    )

    # important as the optimization expects this datatypes, otherwise get a numba.core.errors.TypingError
    X, A, Z = X.astype(np.float32), A.astype(np.float32), Z.astype(np.float32)

    rng = np.random.default_rng(42)
    A_init = -np.log(rng.random((n_samples, n_archetypes), dtype=np.float32))
    A_init /= np.sum(A_init, axis=1, keepdims=True)

    A_approx = compute_a_algorithm(X=X, Z=Z, A=A_init)

    assert np.all(np.isclose(A_approx.sum(axis=1), 1, atol=1e-3))
    assert np.mean(compute_relative_rowwise_l2_distance(A, A_approx)) < 0.10
