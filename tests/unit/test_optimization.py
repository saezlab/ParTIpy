import numpy as np
import pytest
from partipy.arch import AA
from partipy.schema import INIT_ALGS, OPTIM_ALGS, WEIGHT_ALGS
from partipy.simulate import simulate_archetypes
from partipy.utils import align_archetypes, compute_relative_rowwise_l2_distance
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# for regularized_nnls the tests take much longer, and this algorithm is not recommended
FAST_OPTIM_ALGS = tuple(alg for alg in OPTIM_ALGS if alg != "regularized_nnls")


@pytest.mark.github_actions
@pytest.mark.parametrize("optim_str", OPTIM_ALGS)
@pytest.mark.parametrize("init_str", INIT_ALGS)
def test_that_archetypes_can_be_identified_gh(
    optim_str: str,
    init_str: str,
) -> None:
    n_archetypes = 3
    n_dimensions = 2
    N_SAMPLES = 200
    MAX_REL_DIST = 0.10
    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES,
        n_archetypes=n_archetypes,
        n_dimensions=n_dimensions,
        noise_std=0.05,
        seed=0,
    )
    AA_object = AA(n_archetypes=n_archetypes, optim=optim_str, init=init_str)
    AA_object.fit(X)
    Z_hat = AA_object.Z

    Z_hat = align_archetypes(Z, Z_hat)

    rel_dist_between_archetypes = compute_relative_rowwise_l2_distance(Z, Z_hat)

    assert np.any(rel_dist_between_archetypes < MAX_REL_DIST)


@pytest.mark.github_actions
@pytest.mark.parametrize("optim_str", OPTIM_ALGS)
def test_that_relaxation_leads_to_lower_RSS_gh(
    optim_str: str,
) -> None:
    seed = 111
    delta = 0.2
    n_archetypes = 3
    n_dimensions = 2
    n_samples = 500
    X, A, Z = simulate_archetypes(
        n_samples=n_samples,
        n_archetypes=n_archetypes,
        n_dimensions=n_dimensions,
        noise_std=0.15,
        seed=seed,
    )

    AA_object = AA(n_archetypes=n_archetypes, delta=0.0, seed=seed, optim=optim_str)
    AA_object.fit(X)
    RSS_no_delta = AA_object.RSS

    AA_object = AA(n_archetypes=n_archetypes, delta=delta, seed=seed, optim=optim_str)
    AA_object.fit(X)
    RSS_delta = AA_object.RSS

    assert RSS_delta < RSS_no_delta


@pytest.mark.parametrize(
    "n_archetypes, n_dimensions",
    [(n_a, n_d) for n_a in range(2, 9) for n_d in range(2, 19, 4) if n_a <= n_d],
)
@pytest.mark.parametrize("optim_str", OPTIM_ALGS)
def test_that_archetypes_can_be_identified_fail_if_we_dont_optimize(
    n_archetypes: int,
    n_dimensions: int,
    optim_str: str,
) -> None:
    N_SAMPLES = 1_000
    MAX_REL_DIST = 0.10 if n_archetypes < n_dimensions else 0.15
    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES,
        n_archetypes=n_archetypes,
        n_dimensions=n_dimensions,
        noise_std=0.05,
        seed=0,
    )

    AA_object = AA(n_archetypes=n_archetypes, init="uniform", optim=optim_str, max_iter=0)
    AA_object.fit(X)
    Z_hat = AA_object.Z

    Z_hat = align_archetypes(Z, Z_hat)

    rel_dist_between_archetypes = compute_relative_rowwise_l2_distance(Z, Z_hat)

    assert np.any(rel_dist_between_archetypes > MAX_REL_DIST)


@pytest.mark.parametrize(
    "n_archetypes, n_dimensions",
    [(n_a, n_d) for n_a in range(3, 5) for n_d in range(2, 6) if n_a <= n_d],
)
@pytest.mark.parametrize("optim_str", OPTIM_ALGS)
@pytest.mark.parametrize("init_str", INIT_ALGS)
def test_that_all_algorithms_can_identify_archetypes(
    n_archetypes: int,
    n_dimensions: int,
    optim_str: str,
    init_str: str,
) -> None:
    N_SAMPLES = 2_000
    MAX_REL_DIST = 0.8 if n_archetypes < n_dimensions else 0.14
    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES,
        n_archetypes=n_archetypes,
        n_dimensions=n_dimensions,
        noise_std=0.05,
        seed=0,
    )

    AA_object = AA(n_archetypes=n_archetypes, init=init_str, optim=optim_str)
    AA_object.fit(X)
    Z_hat = AA_object.Z

    Z_hat = align_archetypes(Z, Z_hat)

    rel_dist_between_archetypes = compute_relative_rowwise_l2_distance(Z, Z_hat)

    assert np.all(rel_dist_between_archetypes < MAX_REL_DIST)

    # checkig the constraints
    assert np.all(np.isclose(AA_object.A.sum(axis=1), 1, atol=1e-3))
    assert np.all(AA_object.A >= 0)
    assert np.all(np.isclose(AA_object.B.sum(axis=1), 1, atol=1e-3))
    assert np.all(AA_object.B >= 0)


@pytest.mark.parametrize(
    "n_archetypes, n_dimensions",
    [(n_a, n_d) for n_a in range(3, 8, 2) for n_d in range(2, 13, 2) if n_a <= n_d],
)
@pytest.mark.parametrize("optim_str", FAST_OPTIM_ALGS)
def test_that_fast_algorithms_can_identify_archetypes(
    n_archetypes: int,
    n_dimensions: int,
    optim_str: str,
) -> None:
    N_SAMPLES = 20_000
    if n_dimensions > 10:
        MAX_REL_DIST = 0.25
    else:
        MAX_REL_DIST = 0.20

    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES,
        n_archetypes=n_archetypes,
        n_dimensions=n_dimensions,
        noise_std=0.05,
        seed=0,
    )

    AA_object = AA(n_archetypes=n_archetypes, optim=optim_str)
    AA_object.fit(X)
    Z_hat = AA_object.Z

    Z_hat = align_archetypes(Z, Z_hat)

    rel_dist_between_archetypes = compute_relative_rowwise_l2_distance(Z, Z_hat)

    assert np.all(rel_dist_between_archetypes < MAX_REL_DIST)


@pytest.mark.parametrize("optim_str", OPTIM_ALGS)
@pytest.mark.parametrize("weight_str", WEIGHT_ALGS)
@pytest.mark.parametrize("init_str", INIT_ALGS)
def test_that_input_to_AA_is_not_modfied(optim_str, weight_str, init_str) -> None:
    N_SAMPLES = 100
    N_ARCHETYPES = 3
    N_DIMENSIONS = 4
    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES,
        n_archetypes=N_ARCHETYPES,
        n_dimensions=N_DIMENSIONS,
        noise_std=0.05,
        seed=0,
    )
    X_in = X.copy()

    AA_object = AA(
        n_archetypes=N_ARCHETYPES,
        optim=optim_str,
        weight=weight_str,
        init=init_str,
        early_stopping=True if weight_str is None else False,
    )
    AA_object.fit(X)

    assert np.all(np.isclose(X_in, X))


@pytest.mark.parametrize("optim_str", OPTIM_ALGS)
@pytest.mark.parametrize("init_str", INIT_ALGS)
@pytest.mark.parametrize("seed", list(range(3)))
def test_that_shifting_X_does_not_affect_A(
    optim_str: str,
    init_str: str,
    seed: int,
) -> None:
    N_SAMPLES = 1_000
    N_ARCHETYPES = 3
    N_DIMENSIONS = 4
    MAX_REL_DIST_A = 0.50  # TODO: This threshold should be lower (but we anyway rather care for Z)
    MAX_REL_DIST_Z = 0.10
    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES,
        n_archetypes=N_ARCHETYPES,
        n_dimensions=N_DIMENSIONS,
        noise_std=0.05,
        seed=seed,
    )
    rng = np.random.default_rng(seed=seed)
    shift_vec = rng.normal(loc=0.0, scale=100.0, size=N_DIMENSIONS)
    X += shift_vec
    Z_shifted = Z + shift_vec

    AA_object = AA(n_archetypes=N_ARCHETYPES, init=init_str, optim=optim_str)
    AA_object.fit(X)

    euclidean_d = cdist(Z_shifted, AA_object.Z)
    ref_idx, query_idx = linear_sum_assignment(euclidean_d)
    AA_object.A = AA_object.A[:, query_idx]
    AA_object.Z = AA_object.Z[query_idx, :]

    rel_distance_A = compute_relative_rowwise_l2_distance(A, AA_object.A)
    assert np.all(rel_distance_A < MAX_REL_DIST_A)

    rel_distance_to_scaled_Z = compute_relative_rowwise_l2_distance(Z_shifted, AA_object.Z)
    assert np.all(rel_distance_to_scaled_Z < MAX_REL_DIST_Z)


@pytest.mark.parametrize("optim_str", OPTIM_ALGS)
@pytest.mark.parametrize("init_str", INIT_ALGS)
@pytest.mark.parametrize("seed", list(range(3)))
@pytest.mark.parametrize("scale", [1e-9, 1e-4, 1e4, 1e9])
def test_that_scaling_X_does_not_affect_A(
    optim_str: str,
    init_str: str,
    seed: int,
    scale: float,
) -> None:
    N_SAMPLES = 1_000
    N_ARCHETYPES = 3
    N_DIMENSIONS = 4
    MAX_REL_DIST_A = 0.50  # TODO: This threshold should be lower (but we anyway rather care for Z)
    MAX_REL_DIST_Z = 0.10
    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES,
        n_archetypes=N_ARCHETYPES,
        n_dimensions=N_DIMENSIONS,
        noise_std=0.05,
        seed=seed,
    )
    X *= scale
    Z_scaled = Z * scale

    AA_object = AA(n_archetypes=N_ARCHETYPES, init=init_str, optim=optim_str)
    AA_object.fit(X)

    euclidean_d = cdist(Z_scaled, AA_object.Z)
    ref_idx, query_idx = linear_sum_assignment(euclidean_d)
    AA_object.A = AA_object.A[:, query_idx]
    AA_object.Z = AA_object.Z[query_idx, :]

    rel_distance_A = compute_relative_rowwise_l2_distance(A, AA_object.A)
    assert np.all(rel_distance_A < MAX_REL_DIST_A)

    rel_distance_to_scaled_Z = compute_relative_rowwise_l2_distance(Z_scaled, AA_object.Z)
    assert np.all(rel_distance_to_scaled_Z < MAX_REL_DIST_Z)


@pytest.mark.parametrize("optim_str", OPTIM_ALGS)
@pytest.mark.parametrize("coreset_algorithm", [None, "standard"])
def test_that_relaxation_leads_to_lower_RSS(
    optim_str: str,
    coreset_algorithm,
) -> None:
    seed = 42
    delta = 0.2
    n_archetypes = 3
    n_dimensions = 2
    N_SAMPLES = 10_000 if coreset_algorithm else 1_000
    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES,
        n_archetypes=n_archetypes,
        n_dimensions=n_dimensions,
        noise_std=0.15,
        seed=seed,
    )

    AA_object = AA(
        n_archetypes=n_archetypes,
        coreset_algorithm=coreset_algorithm,
        coreset_fraction=0.1,
        delta=0.0,
        seed=seed,
        optim=optim_str,
    )
    AA_object.fit(X)
    RSS_no_delta = AA_object.RSS

    AA_object = AA(
        n_archetypes=n_archetypes,
        coreset_algorithm=coreset_algorithm,
        coreset_fraction=0.1,
        delta=delta,
        seed=seed,
        optim=optim_str,
    )
    AA_object.fit(X)
    RSS_delta = AA_object.RSS

    assert RSS_delta < RSS_no_delta


@pytest.mark.parametrize("optim_str", FAST_OPTIM_ALGS)
@pytest.mark.parametrize("seed", list(range(3)))
@pytest.mark.parametrize("coreset_algorithm", [None, "standard"])
@pytest.mark.parametrize("delta", [0.1, 0.2, 0.4, 0.8])
@pytest.mark.parametrize("n_archetypes, n_dimensions", [(3, 2), (4, 3)])
def test_that_relaxation_leads_to_lower_RSS_fast_algos(
    optim_str: str,
    seed: int,
    coreset_algorithm,
    delta: float,
    n_archetypes: int,
    n_dimensions: int,
) -> None:
    N_SAMPLES = 10_000 if coreset_algorithm else 1_000
    X, A, Z = simulate_archetypes(
        n_samples=N_SAMPLES,
        n_archetypes=n_archetypes,
        n_dimensions=n_dimensions,
        noise_std=0.15,
        seed=seed,
    )

    AA_object = AA(
        n_archetypes=n_archetypes,
        coreset_algorithm=coreset_algorithm,
        coreset_fraction=0.1,
        delta=0.0,
        seed=seed,
        optim=optim_str,
    )
    AA_object.fit(X)
    RSS_no_delta = AA_object.RSS

    AA_object = AA(
        n_archetypes=n_archetypes,
        coreset_algorithm=coreset_algorithm,
        coreset_fraction=0.1,
        delta=delta,
        seed=seed,
        optim=optim_str,
    )
    AA_object.fit(X)
    RSS_delta = AA_object.RSS

    assert RSS_delta < RSS_no_delta


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("PCHA", "projected_gradients"),
        ("FW", "frank_wolfe"),
    ],
)
def test_optimizer_aliases_match_canonical_results(alias: str, canonical: str) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5)).astype(np.float32)
    n_archetypes = 3
    init = "plus_plus"
    seed_value = 7
    max_iter = 8
    rel_tol = 1e-6
    early_stopping = False

    def _build_model(optim_name: str) -> AA:
        return AA(
            n_archetypes=n_archetypes,
            init=init,
            seed=seed_value,
            max_iter=max_iter,
            rel_tol=rel_tol,
            early_stopping=early_stopping,
            optim=optim_name,
        )

    alias_model = _build_model(alias)
    alias_model.fit(X)

    canonical_model = _build_model(canonical)
    canonical_model.fit(X)

    np.testing.assert_allclose(alias_model.Z, canonical_model.Z)
    np.testing.assert_allclose(alias_model.A, canonical_model.A)
    np.testing.assert_allclose(alias_model.B, canonical_model.B)
    assert alias_model.optim == canonical_model.optim == canonical
    assert alias_model.RSS == pytest.approx(canonical_model.RSS)
