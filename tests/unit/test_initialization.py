import numpy as np
import pytest
from partipy.initialize import _init_A, _init_furthest_sum, _init_plus_plus, _init_uniform
from partipy.simulate import simulate_archetypes

N_SAMPLES = 1_000
SEED_LIST = [212341, 135134, 2411315]

X, A, Z = simulate_archetypes(n_samples=N_SAMPLES, n_archetypes=5, n_dimensions=10, noise_std=0.05)


@pytest.mark.parametrize("init_func", [_init_uniform, _init_furthest_sum, _init_plus_plus])
@pytest.mark.parametrize("n_archetypes", list(range(2, 20)))
@pytest.mark.parametrize("seed", SEED_LIST)
def test_that_initalized_B_fullfills_constraints(
    init_func,
    n_archetypes: int,
    seed: int,
) -> None:
    B = init_func(X=X, n_archetypes=n_archetypes, seed=seed)
    assert B.shape[0] == n_archetypes
    assert B.shape[1] == N_SAMPLES
    assert np.all(np.isclose(B.sum(axis=1), 1, atol=1e-5))
    assert np.all(B >= 0)


@pytest.mark.parametrize("n_samples", [10, 100, 1_000, 10_000, 100_000])
@pytest.mark.parametrize("n_archetypes", list(range(2, 20)))
@pytest.mark.parametrize("seed", SEED_LIST)
def test_that_initalized_A_fullfills_constraints(
    n_archetypes: int,
    n_samples: int,
    seed: int,
) -> None:
    A = _init_A(n_samples=n_samples, n_archetypes=n_archetypes, seed=seed)
    assert A.shape[0] == n_samples
    assert A.shape[1] == n_archetypes
    assert np.all(np.isclose(A.sum(axis=1), 1, atol=1e-5))
    assert np.all(A >= 0)
