import numpy as np
import pytest
from partipy.generate_test_data import simulate
from partipy.initialize import _furthest_sum_init, _random_init

N_SAMPLES = 1_000

X, A, Z = simulate(n_samples=N_SAMPLES, n_archetypes=5, n_dimensions=10, noise_std=0.0)


@pytest.mark.parametrize("init_func", [_furthest_sum_init, _random_init])
@pytest.mark.parametrize("n_archetypes", list(range(2, 20)))
def test_that_initalized_B_fullfills_constraints(
    init_func,
    n_archetypes: int,
) -> None:
    B = init_func(X=X, n_archetypes=n_archetypes)
    assert B.shape[0] == n_archetypes
    assert B.shape[1] == N_SAMPLES
    assert np.all(np.isclose(B.sum(axis=1), 1))
    assert np.all(B >= 0)
