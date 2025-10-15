from __future__ import annotations

import json
from collections.abc import Iterable, MutableMapping
from contextlib import contextmanager

import anndata as ad

from .schema import ArchetypeConfig

_CONFIG_PREFIX = "ArchetypeConfig::"
_SERIALIZED_UNS_KEYS: tuple[str, ...] = (
    "AA_results",
    "AA_selection_metrics",
    "AA_cell_weights",
    "AA_bootstrap",
    "AA_t_ratio",
    "AA_permutation",
)


def _encode_config_key(config: ArchetypeConfig) -> str:
    data = config.model_dump(mode="python", warnings=False)
    optim_kwargs = data.get("optim_kwargs")
    if isinstance(optim_kwargs, (list, tuple)):
        # Convert the frozen tuple of pairs back to a dict for stable JSON encoding
        data["optim_kwargs"] = dict(optim_kwargs)
    return f"{_CONFIG_PREFIX}{json.dumps(data, sort_keys=True)}"


def _decode_config_key(key: object) -> ArchetypeConfig | None:
    if not isinstance(key, str) or not key.startswith(_CONFIG_PREFIX):
        return None
    payload = key[len(_CONFIG_PREFIX) :]
    try:
        data = json.loads(payload)
        optim_kwargs = data.get("optim_kwargs")
        if isinstance(optim_kwargs, (list, tuple)):
            data["optim_kwargs"] = dict(optim_kwargs)
        return ArchetypeConfig.model_validate(data)
    except ValueError as err:  # pragma: no cover - defensive; should not happen for valid files
        raise ValueError(
            "Unable to decode an ArchetypeConfig key stored in `adata.uns`. "
            "The stored value does not match the expected JSON encoding."
        ) from err


def _serialize_store(store: MutableMapping) -> tuple[MutableMapping, bool]:
    encoded: dict = {}
    changed = False
    for key, value in store.items():
        if isinstance(key, ArchetypeConfig):
            encoded[_encode_config_key(key)] = value
            changed = True
        else:
            encoded[key] = value
    return (encoded if changed else store, changed)


def _deserialize_store(store: MutableMapping) -> tuple[MutableMapping, bool]:
    decoded: dict = {}
    changed = False
    for key, value in store.items():
        cfg = _decode_config_key(key)
        if cfg is not None:
            decoded[cfg] = value
            changed = True
        else:
            decoded[key] = value
    return (decoded if changed else store, changed)


def serialize_archetype_caches(
    adata: ad.AnnData,
    *,
    uns_keys: Iterable[str] = _SERIALIZED_UNS_KEYS,
) -> dict[str, MutableMapping]:
    """Convert ArchetypeConfig keys in selected ``adata.uns`` stores to JSON strings.

    Returns a mapping of the stores that were modified so they can be restored later.
    """
    originals: dict[str, MutableMapping] = {}
    for uns_key in uns_keys:
        store = adata.uns.get(uns_key)
        if not isinstance(store, MutableMapping) or len(store) == 0:
            continue
        serialized, changed = _serialize_store(store)
        if changed:
            originals[uns_key] = store
            adata.uns[uns_key] = serialized
    return originals


def ensure_archetype_config_keys(
    adata: ad.AnnData,
    *,
    uns_keys: Iterable[str] = _SERIALIZED_UNS_KEYS,
) -> None:
    """Restore JSON-string keys in ``adata.uns`` back into ``ArchetypeConfig`` objects."""
    for uns_key in uns_keys:
        store = adata.uns.get(uns_key)
        if not isinstance(store, MutableMapping) or len(store) == 0:
            continue
        deserialized, changed = _deserialize_store(store)
        if changed:
            adata.uns[uns_key] = deserialized


@contextmanager
def _temporarily_serialized(adata: ad.AnnData, *, uns_keys: Iterable[str] = _SERIALIZED_UNS_KEYS):
    originals = serialize_archetype_caches(adata, uns_keys=uns_keys)
    try:
        yield
    finally:
        for uns_key, store in originals.items():
            adata.uns[uns_key] = store


def write_h5ad(
    adata: ad.AnnData,
    filename: str,
    *,
    uns_keys: Iterable[str] = _SERIALIZED_UNS_KEYS,
    **kwargs,
) -> None:
    """Write an AnnData object to disk after temporarily serializing cached config keys."""
    with _temporarily_serialized(adata, uns_keys=uns_keys):
        adata.write_h5ad(filename, **kwargs)


def read_h5ad(
    filename: str,
    *,
    uns_keys: Iterable[str] = _SERIALIZED_UNS_KEYS,
    **kwargs,
) -> ad.AnnData:
    """Load an AnnData file and rehydrate any serialized ``ArchetypeConfig`` cache keys."""
    adata = ad.read_h5ad(filename, **kwargs)
    ensure_archetype_config_keys(adata, uns_keys=uns_keys)
    return adata


__all__ = [
    "ensure_archetype_config_keys",
    "read_h5ad",
    "serialize_archetype_caches",
    "write_h5ad",
]
