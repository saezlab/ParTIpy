import pytest
from partipy.schema import ArchetypeConfig


def _config_kwargs(**overrides):
    base = {
        "obsm_key": "X_pca",
        "n_dimensions": (0, 1),
        "n_archetypes": 3,
        "init": "plus_plus",
        "optim": "projected_gradients",
        "weight": None,
        "max_iter": 50,
        "rel_tol": 1e-4,
        "early_stopping": True,
        "coreset_algorithm": None,
        "coreset_fraction": 0.1,
        "coreset_size": None,
        "delta": 0.0,
        "seed": 0,
        "optim_kwargs": {},
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("PCHA", "projected_gradients"),
        ("pcha", "projected_gradients"),
        ("FW", "frank_wolfe"),
        ("fw", "frank_wolfe"),
    ],
)
def test_archetype_config_normalizes_optimizer_aliases(alias: str, expected: str) -> None:
    cfg = ArchetypeConfig(**_config_kwargs(optim=alias))
    assert cfg.optim == expected
