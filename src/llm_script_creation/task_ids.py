"""Dataset-id helpers shared by the benchmark and the admin-agent adapter.

Benchmark tasks exist in more than one split version. v2 keeps the original ids
(``plan_year``, ``upr_recommendations``, ...); v3 uses a ``_v3`` suffix
(``plan_year_v3``) in the same catalog, so old artifacts stay untouched and the
viewer can show both versions side by side.

Task logic should key on :func:`base_dataset_id`, never on the raw id, so a
versioned id behaves like its original task. Exactly one version suffix is
stripped; an id that merely ends in ``_v3`` for another reason is not expected
in this catalog.
"""

from __future__ import annotations

V2 = "v2"
V3 = "v3"
VERSIONS: tuple[str, ...] = (V2, V3)

V3_SUFFIX = "_v3"


def dataset_version(dataset_id: str) -> str:
    """Return the split version encoded in a dataset id (``v2`` by default)."""
    return V3 if str(dataset_id).endswith(V3_SUFFIX) else V2


def base_dataset_id(dataset_id: str) -> str:
    """Strip the version suffix so task logic sees the original task id."""
    value = str(dataset_id)
    if value.endswith(V3_SUFFIX):
        return value[: -len(V3_SUFFIX)]
    return value


def versioned_dataset_id(dataset_id: str, version: str) -> str:
    """Return the id of ``dataset_id`` in ``version`` (idempotent for v2)."""
    base = base_dataset_id(dataset_id)
    if version == V3:
        return f"{base}{V3_SUFFIX}"
    if version == V2:
        return base
    raise ValueError(f"unknown dataset version {version!r}; choose from {VERSIONS}")


__all__ = [
    "V2",
    "V3",
    "V3_SUFFIX",
    "VERSIONS",
    "base_dataset_id",
    "dataset_version",
    "versioned_dataset_id",
]
