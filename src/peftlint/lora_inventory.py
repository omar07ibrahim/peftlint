"""Exact, payload-free inventory of saved PEFT 0.19.1 LoRA tensors."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import NoReturn

from peftlint.safetensors import (
    SafetensorsDtype,
    SafetensorsManifest,
    TensorManifest,
)

LORA_INVENTORY_SCHEMA = "peft-0.19.1-lora-inventory-v1"

_U64_MAX = 2**64 - 1
_REQUIRED_TENSOR_FIELDS = frozenset({"dtype", "shape", "data_offsets"})


class LoraTensorRole(StrEnum):
    """A role proved solely from one exact saved-state key."""

    WEIGHT_A = "weight_a"
    WEIGHT_B = "weight_b"
    EMBEDDING_A = "embedding_a"
    EMBEDDING_B = "embedding_b"
    UNCLASSIFIED = "unclassified"


class LoraPairKind(StrEnum):
    """The two PEFT 0.19.1 saved-key pair families modeled here."""

    WEIGHT = "weight"
    EMBEDDING = "embedding"


class LoraInventoryIssueKind(StrEnum):
    """A structural reason an inventory member cannot close silently."""

    EMPTY_INVENTORY = "empty_inventory"
    UNCLASSIFIED_TENSOR = "unclassified_tensor"
    UNKNOWN_TENSOR_FIELDS = "unknown_tensor_fields"
    ORPHAN_MEMBER = "orphan_member"
    MIXED_PAIR_KIND = "mixed_pair_kind"
    UNMODELED_WEIGHT_ORIENTATION = "unmodeled_weight_orientation"


_SAVED_SUFFIXES = (
    (".lora_embedding_A", LoraTensorRole.EMBEDDING_A),
    (".lora_embedding_B", LoraTensorRole.EMBEDDING_B),
    (".lora_A.weight", LoraTensorRole.WEIGHT_A),
    (".lora_B.weight", LoraTensorRole.WEIGHT_B),
)
_RESERVED_TARGET_COMPONENTS = frozenset(
    {"lora_embedding_A", "lora_embedding_B", "lora_A", "lora_B"}
)


@dataclass(frozen=True, slots=True)
class LoraTensor:
    """An owned projection of one manifest tensor and its exact saved-key role."""

    name: str = field(repr=False)
    target: str | None = field(repr=False)
    role: LoraTensorRole
    dtype: SafetensorsDtype
    shape: tuple[int, ...]
    unknown_fields: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        _require_unicode("tensor name", self.name)
        if type(self.dtype) is not SafetensorsDtype:
            raise TypeError("tensor dtype must be SafetensorsDtype")
        if type(self.shape) is not tuple or any(
            type(dimension) is not int for dimension in self.shape
        ):
            raise TypeError("tensor shape must be a tuple of integers")
        if any(not 0 <= dimension <= _U64_MAX for dimension in self.shape):
            raise ValueError("tensor shape dimensions must fit unsigned 64-bit values")
        _require_unicode_tuple("unknown_fields", self.unknown_fields)
        if self.unknown_fields != tuple(sorted(frozenset(self.unknown_fields))):
            raise ValueError("unknown_fields must be sorted and unique")
        if frozenset(self.unknown_fields) & _REQUIRED_TENSOR_FIELDS:
            raise ValueError("required tensor fields cannot be unknown fields")
        if type(self.role) is not LoraTensorRole:
            raise TypeError("tensor role must be LoraTensorRole")
        if self.target is not None:
            _require_text("tensor target", self.target)

        expected_role, expected_target = _classify_saved_key(self.name)
        if self.role is not expected_role or self.target != expected_target:
            raise ValueError("tensor role and target must match the exact saved-state key")

    @property
    def sort_key(self) -> tuple[str, str]:
        """Return the canonical member order."""

        return self.name, self.role.value

    @property
    def evidence_path(self) -> str:
        """Return an injective printable scope for any valid raw tensor key."""

        return f"tensor:{json.dumps(self.name, ensure_ascii=True)}"

    def __setstate__(self, _state: object) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is immutable")


@dataclass(frozen=True, slots=True)
class LoraPair:
    """A syntactically complete A/B pair for one exact target path."""

    target: str = field(repr=False)
    kind: LoraPairKind
    a: LoraTensor = field(repr=False)
    b: LoraTensor = field(repr=False)

    def __post_init__(self) -> None:
        _require_text("pair target", self.target)
        if type(self.kind) is not LoraPairKind:
            raise TypeError("pair kind must be LoraPairKind")
        if type(self.a) is not LoraTensor or type(self.b) is not LoraTensor:
            raise TypeError("pair members must be LoraTensor values")
        object.__setattr__(self, "a", _copy_tensor(self.a))
        object.__setattr__(self, "b", _copy_tensor(self.b))

        expected_roles = {
            LoraPairKind.WEIGHT: (LoraTensorRole.WEIGHT_A, LoraTensorRole.WEIGHT_B),
            LoraPairKind.EMBEDDING: (
                LoraTensorRole.EMBEDDING_A,
                LoraTensorRole.EMBEDDING_B,
            ),
        }[self.kind]
        if (self.a.role, self.b.role) != expected_roles:
            raise ValueError("pair members do not match the declared pair kind")
        if self.a.target != self.target or self.b.target != self.target:
            raise ValueError("pair members must share the exact pair target")

    @property
    def sort_key(self) -> tuple[str, str]:
        """Return the canonical pair order."""

        return self.target, self.kind.value

    def __setstate__(self, _state: object) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is immutable")


@dataclass(frozen=True, slots=True)
class LoraInventoryIssue:
    """One deterministic issue, scoped to a tensor key or exact target."""

    kind: LoraInventoryIssueKind
    tensor: LoraTensor | None = field(default=None, repr=False)
    target: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.kind) is not LoraInventoryIssueKind:
            raise TypeError("inventory issue kind must be LoraInventoryIssueKind")
        if self.tensor is not None and type(self.tensor) is not LoraTensor:
            raise TypeError("inventory issue tensor must be LoraTensor or None")
        if self.tensor is not None:
            object.__setattr__(self, "tensor", _copy_tensor(self.tensor))
        if self.target is not None:
            _require_text("inventory issue target", self.target)

        if self.kind is LoraInventoryIssueKind.EMPTY_INVENTORY:
            if self.tensor is not None or self.target is not None:
                raise ValueError("empty-inventory issue must not name a tensor or target")
            return
        if self.kind is LoraInventoryIssueKind.MIXED_PAIR_KIND:
            if self.tensor is not None or self.target is None:
                raise ValueError("mixed-kind issue requires only an exact target")
            return
        if self.tensor is None or self.target is not None:
            raise ValueError("tensor-scoped inventory issue requires a tensor")
        if self.kind is LoraInventoryIssueKind.UNCLASSIFIED_TENSOR:
            if self.tensor.role is not LoraTensorRole.UNCLASSIFIED:
                raise ValueError("unclassified issue requires an unclassified tensor")
        elif self.tensor.role is LoraTensorRole.UNCLASSIFIED:
            raise ValueError("classified inventory issue requires a recognized tensor")
        if (
            self.kind is LoraInventoryIssueKind.UNKNOWN_TENSOR_FIELDS
            and not self.tensor.unknown_fields
        ):
            raise ValueError("unknown-fields issue requires retained extension fields")

    @property
    def logical_path(self) -> str | None:
        """Return the injective printable evidence scope, when present."""

        if self.tensor is not None:
            return self.tensor.evidence_path
        if self.target is not None:
            return f"target:{json.dumps(self.target, ensure_ascii=True)}"
        return None

    @property
    def sort_key(self) -> tuple[str, str, str]:
        """Return the canonical issue order."""

        role = "" if self.tensor is None else self.tensor.role.value
        return self.logical_path or "", self.kind.value, role

    def __setstate__(self, _state: object) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is immutable")


@dataclass(frozen=True, slots=True)
class LoraInventory:
    """Canonical semantic inventory derived without retaining payload access."""

    schema: str
    weights: SafetensorsManifest = field(repr=False)
    tensors: tuple[LoraTensor, ...] = field(repr=False)
    pairs: tuple[LoraPair, ...]
    issues: tuple[LoraInventoryIssue, ...]

    def __post_init__(self) -> None:
        if type(self.schema) is not str:
            raise TypeError("inventory schema must be a string")
        if self.schema != LORA_INVENTORY_SCHEMA:
            raise ValueError("schema must identify the pinned LoRA inventory schema")
        if type(self.weights) is not SafetensorsManifest:
            raise TypeError("inventory weights must be SafetensorsManifest")
        _require_exact_tuple("tensors", self.tensors, LoraTensor)
        _require_exact_tuple("pairs", self.pairs, LoraPair)
        _require_exact_tuple("issues", self.issues, LoraInventoryIssue)

        weights = _copy_manifest(self.weights)
        tensors = tuple(_copy_tensor(tensor) for tensor in self.tensors)
        pairs = tuple(_copy_pair(pair) for pair in self.pairs)
        issues = tuple(_copy_issue(issue) for issue in self.issues)
        if tensors != tuple(sorted(tensors, key=lambda tensor: tensor.sort_key)):
            raise ValueError("inventory tensors must use canonical key order")
        if len(tensors) != len(frozenset(tensor.name for tensor in tensors)):
            raise ValueError("inventory tensor names must be unique")
        if pairs != tuple(sorted(pairs, key=lambda pair: pair.sort_key)):
            raise ValueError("inventory pairs must use canonical target order")
        if issues != tuple(sorted(issues, key=lambda issue: issue.sort_key)):
            raise ValueError("inventory issues must use canonical scope order")
        if len(issues) != len(frozenset(issue.sort_key for issue in issues)):
            raise ValueError("inventory issues must be unique")

        expected_tensors = tuple(
            sorted(
                (_project_tensor(tensor) for tensor in weights.tensors),
                key=lambda tensor: tensor.sort_key,
            )
        )
        if tensors != expected_tensors:
            raise ValueError("inventory tensors are inconsistent with its weights manifest")
        expected_pairs, expected_issues = _compile_relationships(tensors)
        if pairs != expected_pairs or issues != expected_issues:
            raise ValueError("inventory relationships are inconsistent with its tensors")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "tensors", tensors)
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "issues", issues)

    def __setstate__(self, _state: object) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is immutable")


def inspect_lora_inventory(weights: SafetensorsManifest) -> LoraInventory:
    """Compile exact saved-key roles from an already validated manifest.

    The source manifest is defensively reconstructed and retained as immutable
    value evidence. The semantic projection adds only names, dtypes, shapes,
    and retained header-extension names; payload bytes, paths, and source
    handles are never accepted or kept.
    """

    if type(weights) is not SafetensorsManifest:
        raise TypeError("weights must be SafetensorsManifest")
    validated = _copy_manifest(weights)
    tensors = tuple(
        sorted(
            (_project_tensor(tensor) for tensor in validated.tensors),
            key=lambda tensor: tensor.sort_key,
        )
    )
    pairs, issues = _compile_relationships(tensors)
    return LoraInventory(
        schema=LORA_INVENTORY_SCHEMA,
        weights=validated,
        tensors=tensors,
        pairs=pairs,
        issues=issues,
    )


def _project_tensor(tensor: TensorManifest) -> LoraTensor:
    role, target = _classify_saved_key(tensor.name)
    return LoraTensor(
        name=tensor.name,
        target=target,
        role=role,
        dtype=tensor.dtype,
        shape=tensor.shape,
        unknown_fields=tensor.unknown_fields,
    )


def _classify_saved_key(name: str) -> tuple[LoraTensorRole, str | None]:
    matches = tuple((suffix, role) for suffix, role in _SAVED_SUFFIXES if name.endswith(suffix))
    if len(matches) != 1:
        return LoraTensorRole.UNCLASSIFIED, None
    suffix, role = matches[0]
    target = name[: -len(suffix)]
    if not _is_unambiguous_target(target):
        return LoraTensorRole.UNCLASSIFIED, None
    return role, target


def _is_unambiguous_target(target: str) -> bool:
    components = target.split(".")
    return bool(
        target and all(components) and not _RESERVED_TARGET_COMPONENTS.intersection(components)
    )


def _compile_relationships(
    tensors: tuple[LoraTensor, ...],
) -> tuple[tuple[LoraPair, ...], tuple[LoraInventoryIssue, ...]]:
    if not tensors:
        return (), (LoraInventoryIssue(LoraInventoryIssueKind.EMPTY_INVENTORY),)

    grouped: dict[tuple[str, LoraPairKind], dict[LoraTensorRole, LoraTensor]] = {}
    kinds_by_target: dict[str, set[LoraPairKind]] = defaultdict(set)
    issues: list[LoraInventoryIssue] = []
    for tensor in tensors:
        if tensor.role is LoraTensorRole.UNCLASSIFIED:
            issues.append(LoraInventoryIssue(LoraInventoryIssueKind.UNCLASSIFIED_TENSOR, tensor))
            continue
        if tensor.unknown_fields:
            issues.append(LoraInventoryIssue(LoraInventoryIssueKind.UNKNOWN_TENSOR_FIELDS, tensor))
        if tensor.target is None:  # Defensive guard for forged values.
            raise ValueError("recognized inventory tensor must have a target")
        kind = _kind_for_role(tensor.role)
        grouped.setdefault((tensor.target, kind), {})[tensor.role] = tensor
        kinds_by_target.setdefault(tensor.target, set()).add(kind)

    pairs: list[LoraPair] = []
    for (target, kind), members in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1].value)
    ):
        a_role, b_role = {
            LoraPairKind.WEIGHT: (LoraTensorRole.WEIGHT_A, LoraTensorRole.WEIGHT_B),
            LoraPairKind.EMBEDDING: (
                LoraTensorRole.EMBEDDING_A,
                LoraTensorRole.EMBEDDING_B,
            ),
        }[kind]
        if a_role in members and b_role in members:
            pair = LoraPair(target, kind, members[a_role], members[b_role])
            pairs.append(pair)
            if kind is LoraPairKind.WEIGHT and (len(pair.a.shape) > 2 or len(pair.b.shape) > 2):
                issues.append(
                    LoraInventoryIssue(
                        LoraInventoryIssueKind.UNMODELED_WEIGHT_ORIENTATION,
                        pair.a,
                    )
                )
        else:
            issues.extend(
                LoraInventoryIssue(LoraInventoryIssueKind.ORPHAN_MEMBER, tensor)
                for tensor in members.values()
            )

    issues.extend(
        LoraInventoryIssue(LoraInventoryIssueKind.MIXED_PAIR_KIND, target=target)
        for target, kinds in kinds_by_target.items()
        if len(kinds) > 1
    )

    return (
        tuple(sorted(pairs, key=lambda pair: pair.sort_key)),
        tuple(sorted(issues, key=lambda issue: issue.sort_key)),
    )


def _kind_for_role(role: LoraTensorRole) -> LoraPairKind:
    if role in {LoraTensorRole.WEIGHT_A, LoraTensorRole.WEIGHT_B}:
        return LoraPairKind.WEIGHT
    if role in {LoraTensorRole.EMBEDDING_A, LoraTensorRole.EMBEDDING_B}:
        return LoraPairKind.EMBEDDING
    raise ValueError("unclassified tensor has no LoRA pair kind")


def _copy_tensor(tensor: LoraTensor) -> LoraTensor:
    return LoraTensor(
        name=tensor.name,
        target=tensor.target,
        role=tensor.role,
        dtype=tensor.dtype,
        shape=tensor.shape,
        unknown_fields=tensor.unknown_fields,
    )


def _copy_manifest(weights: SafetensorsManifest) -> SafetensorsManifest:
    return SafetensorsManifest(
        plan=weights.plan,
        tensors=weights.tensors,
        metadata=weights.metadata,
        metadata_form=weights.metadata_form,
        notices=weights.notices,
    )


def _copy_pair(pair: LoraPair) -> LoraPair:
    return LoraPair(pair.target, pair.kind, pair.a, pair.b)


def _copy_issue(issue: LoraInventoryIssue) -> LoraInventoryIssue:
    return LoraInventoryIssue(issue.kind, issue.tensor, issue.target)


def _require_text(name: str, value: object) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or _has_lone_surrogate(value):
        raise ValueError(f"{name} must be non-empty valid Unicode")


def _require_unicode(name: str, value: object) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if _has_lone_surrogate(value):
        raise ValueError(f"{name} must contain valid Unicode")


def _require_unicode_tuple(name: str, value: object) -> None:
    if type(value) is not tuple or any(type(item) is not str for item in value):
        raise TypeError(f"{name} must be a tuple of strings")
    if any(_has_lone_surrogate(item) for item in value):
        raise ValueError(f"{name} must contain valid Unicode strings")


def _require_exact_tuple(name: str, value: object, item_type: type[object]) -> None:
    if type(value) is not tuple or any(type(item) is not item_type for item in value):
        raise TypeError(f"{name} must be a tuple of {item_type.__name__} values")


def _has_lone_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


__all__ = [
    "LORA_INVENTORY_SCHEMA",
    "LoraInventory",
    "LoraInventoryIssue",
    "LoraInventoryIssueKind",
    "LoraPair",
    "LoraPairKind",
    "LoraTensor",
    "LoraTensorRole",
    "inspect_lora_inventory",
]
