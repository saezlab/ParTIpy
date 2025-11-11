from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# -----------------------------
# Allowed sets (from your consts)
# -----------------------------
INIT_ALG_TYPES = Literal["uniform", "furthest_sum", "plus_plus"]
OPTIM_ALGS_TYPES = Literal["regularized_nnls", "projected_gradients", "frank_wolfe"]
WEIGHT_ALGS_TYPES = Literal[None, "bisquare", "huber"]
CORESET_ALGS_TYPES = Literal[None, "standard", "lightweight_kmeans", "uniform"]

INIT_ALGS = get_args(INIT_ALG_TYPES)
OPTIM_ALGS = get_args(OPTIM_ALGS_TYPES)
WEIGHT_ALGS = get_args(WEIGHT_ALGS_TYPES)
CORESET_ALGS = get_args(CORESET_ALGS_TYPES)

_OPTIM_ALIAS_MAP: dict[str, OPTIM_ALGS_TYPES] = {name: name for name in OPTIM_ALGS}
_OPTIM_ALIAS_MAP.update(
    {
        "pcha": "projected_gradients",
        "projectedgradients": "projected_gradients",
        "fw": "frank_wolfe",
        "frankwolfe": "frank_wolfe",
    }
)


def canonicalize_optim(value: str) -> OPTIM_ALGS_TYPES:
    """Normalize optimizer identifiers (case/spacing insensitive) to canonical schema literals."""
    if not isinstance(value, str):
        raise TypeError("Optimization algorithm must be provided as a string.")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "":
        raise ValueError("Optimization algorithm must be a non-empty string.")
    canonical = _OPTIM_ALIAS_MAP.get(normalized)
    if canonical is None:
        allowed = ", ".join(sorted(_OPTIM_ALIAS_MAP))
        raise ValueError(f"Optimization algorithm '{value}' is not supported. Allowed names include: {allowed}.")
    return cast(OPTIM_ALGS_TYPES, canonical)


# -----------------------------
# Defaults / constants
# -----------------------------
DEFAULT_INIT: INIT_ALG_TYPES = "plus_plus"
DEFAULT_WEIGHT: WEIGHT_ALGS_TYPES = None
DEFAULT_OPTIM: OPTIM_ALGS_TYPES = "projected_gradients"

LAMBDA: float = 1_000.0
MIN_ITERATIONS: int = 20

DEFAULT_MAX_ITER: int = 500
DEFAULT_REL_TOL: float = 1e-4
DEFAULT_OBSM_KEY: str = "X_pca"

DEFAULT_SEED: int = 42


# -----------------------------
# Hashable freezing helper
# -----------------------------
def _freeze(x: Any) -> Any:
    from collections.abc import Mapping, Sequence

    import numpy as np

    if x is None or isinstance(x, int | float | str | bool):
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Mapping):
        return tuple(sorted((k, _freeze(v)) for k, v in x.items()))
    if isinstance(x, set | frozenset):
        return tuple(sorted(_freeze(v) for v in x))
    if isinstance(x, Sequence) and not isinstance(x, str | bytes):
        return tuple(_freeze(v) for v in x)
    return repr(x)


# -----------------------------
# Immutable, hashable schema
# -----------------------------
class ArchetypeConfig(BaseModel):
    """Hashable Specification of Archetype Analysis Optimization Configuration"""

    model_config = ConfigDict(frozen=True)  # -> immutable & hashable

    # data keys
    obsm_key: str = DEFAULT_OBSM_KEY
    n_dimensions: tuple[int, ...]  # NOTE: n_dimensions is actually list[int] but we need to use immutable tuple instead

    # core optimization knobs
    n_archetypes: int
    init: INIT_ALG_TYPES = DEFAULT_INIT
    optim: OPTIM_ALGS_TYPES = DEFAULT_OPTIM
    weight: WEIGHT_ALGS_TYPES = DEFAULT_WEIGHT

    # early stopping
    max_iter: int = Field(DEFAULT_MAX_ITER, ge=MIN_ITERATIONS)
    rel_tol: float = Field(DEFAULT_REL_TOL, gt=0.0, lt=1.0)
    early_stopping: bool = True

    # coreset control
    coreset_algorithm: CORESET_ALGS_TYPES = None
    coreset_fraction: float | None = Field(default=None, gt=0.0, lt=1.0)
    coreset_size: int | None = Field(default=None, ge=1)

    # objective relaxation
    delta: float = Field(0.0, ge=0.0)

    # misc
    seed: int = DEFAULT_SEED

    # free-form optimizer kwargs (validated + frozen)
    optim_kwargs: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("optim", mode="before")
    @classmethod
    def _normalize_optim(cls, value: str) -> OPTIM_ALGS_TYPES:
        return canonicalize_optim(value)

    @field_validator("optim_kwargs")
    @classmethod
    def _freeze_ok(cls, v: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
        return tuple(sorted((k, _freeze(val)) for k, val in dict(v).items()))

    @model_validator(mode="after")
    def _check_coreset_logic(self) -> ArchetypeConfig:
        # Exclusivity: at most one of fraction/size
        if self.coreset_fraction is not None and self.coreset_size is not None:
            raise ValueError("Set either `coreset_fraction` or `coreset_size` (not both).")
        else:
            # If algorithm is set, require exactly one of fraction/size
            if (self.coreset_fraction is None) == (self.coreset_size is None):
                raise ValueError("With a coreset algorithm, set exactly one of fraction or size.")
        return self

    def _signature(self, *, ignore_fields: Iterable[str] = ("n_archetypes",)) -> tuple[tuple[str, Any], ...]:
        ignore = set(ignore_fields)
        fields = type(self).model_fields  # <-- class-level access (v2-safe)
        return tuple((name, getattr(self, name)) for name in fields if name not in ignore)

    def _matches_signature(
        self,
        other: ArchetypeConfig,
        *,
        ignore_fields: Iterable[str] = ("n_archetypes",),
    ) -> bool:
        return self._signature(ignore_fields=ignore_fields) == other._signature(ignore_fields=ignore_fields)


def query_configs_by_signature(
    configs: Iterable[ArchetypeConfig],
    reference: ArchetypeConfig,
    *,
    ignore_fields: Iterable[str] = ("n_archetypes",),
) -> list[ArchetypeConfig]:
    """
    Return all `ArchetypeConfig` objects from `configs` that match a given reference configuration,
    up to a set of ignored fields.

    Two configurations are considered equivalent if their field-value pairs match exactly
    after excluding all fields listed in `ignore_fields`. This function is useful for grouping
    or retrieving configurations that share the same hyperparameter signature except for
    specific parameters such as `n_archetypes`.

    Parameters
    ----------
    configs : Iterable[ArchetypeConfig]
        Collection of configuration objects to be filtered.
    reference : ArchetypeConfig
        Reference configuration against which all others are compared.
    ignore_fields : Iterable[str], optional
        Field names to exclude from the equality check (default: ("n_archetypes",)).

    Returns
    -------
    list[ArchetypeConfig]
        All configurations whose signatures match the reference configuration
        when ignoring the specified fields.

    Examples
    --------
    >>> ref = ArchetypeConfig(n_archetypes=5, n_dimensions=(50,), init="plus_plus", optim="projected_gradients")
    >>> cfgs = [
    ...     ref,
    ...     ref.model_copy(update={"n_archetypes": 10}),
    ...     ref.model_copy(update={"optim": "frank_wolfe"}),
    ... ]
    >>> query_configs_by_signature(cfgs, ref)
    [cfgs[0], cfgs[1]]  # only differ by n_archetypes
    """
    ignore = tuple(ignore_fields)
    ref_signature = reference._signature(ignore_fields=ignore)
    return [cfg for cfg in configs if cfg._signature(ignore_fields=ignore) == ref_signature]
