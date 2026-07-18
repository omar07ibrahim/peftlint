from __future__ import annotations

import inspect
import json
import random
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import TypedDict, cast

import pytest

import peftlint
import peftlint.lora_inventory as inventory_module
from peftlint.lora_inventory import (
    LORA_INVENTORY_SCHEMA,
    LoraInventory,
    LoraInventoryIssueKind,
    LoraPairKind,
    LoraTensorRole,
    inspect_lora_inventory,
)
from peftlint.safetensors import (
    HeaderNotice,
    HeaderReadPlan,
    MetadataForm,
    SafetensorsDtype,
    SafetensorsLimits,
    SafetensorsManifest,
    TensorManifest,
)


def evidence_path(name: str) -> str:
    return f"tensor:{json.dumps(name, ensure_ascii=True)}"


class FixtureTensor(TypedDict):
    key: str
    shape: list[int]


class FixturePair(TypedDict):
    kind: str
    target: str
    a: FixtureTensor
    b: FixtureTensor


class PeftSourceFixture(TypedDict):
    schema: str
    peft_version: str
    commit: str
    sources: dict[str, str]
    pairs: list[FixturePair]
    unclassified_keys: list[str]


def weights_manifest(
    tensors: list[tuple[str, tuple[int, ...], tuple[str, ...]]],
) -> SafetensorsManifest:
    cursor = 0
    members: list[TensorManifest] = []
    for name, shape, unknown_fields in tensors:
        elements = 1
        for dimension in shape:
            elements *= dimension
        size = elements * 4
        members.append(
            TensorManifest(
                name=name,
                dtype=SafetensorsDtype.F32,
                shape=shape,
                data_offsets=(cursor, cursor + size),
                unknown_fields=unknown_fields,
            )
        )
        cursor += size
    notices = (
        (HeaderNotice.UNKNOWN_TENSOR_FIELDS,)
        if any(tensor.unknown_fields for tensor in members)
        else ()
    )
    return SafetensorsManifest(
        plan=HeaderReadPlan(
            file_size=8 + cursor,
            header_size=0,
            header_offset=8,
            data_offset=8,
            data_size=cursor,
            limits=SafetensorsLimits(),
        ),
        tensors=tuple(members),
        metadata=(),
        metadata_form=MetadataForm.ABSENT,
        notices=notices,
    )


def test_exact_linear_and_embedding_pairs_are_compiled() -> None:
    weights = weights_manifest(
        [
            ("model.embed.lora_embedding_B", (4096, 8), ()),
            ("model.q_proj.lora_B.weight", (32, 8), ()),
            ("model.embed.lora_embedding_A", (8, 32000), ()),
            ("model.q_proj.lora_A.weight", (8, 64), ()),
        ]
    )

    inventory = inspect_lora_inventory(weights)

    assert inventory.schema == LORA_INVENTORY_SCHEMA
    assert tuple(tensor.name for tensor in inventory.tensors) == (
        "model.embed.lora_embedding_A",
        "model.embed.lora_embedding_B",
        "model.q_proj.lora_A.weight",
        "model.q_proj.lora_B.weight",
    )
    assert tuple((pair.target, pair.kind) for pair in inventory.pairs) == (
        ("model.embed", LoraPairKind.EMBEDDING),
        ("model.q_proj", LoraPairKind.LINEAR),
    )
    assert inventory.pairs[0].a.shape == (8, 32000)
    assert inventory.pairs[0].b.shape == (4096, 8)
    assert inventory.pairs[1].a.shape == (8, 64)
    assert inventory.pairs[1].b.shape == (32, 8)
    assert inventory.issues == ()


def test_pinned_peft_source_fixture_compiles_without_runtime_dependencies() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "peft-0.19.1-lora-state.json"
    fixture = cast(PeftSourceFixture, json.loads(fixture_path.read_text(encoding="utf-8")))
    tensors: list[tuple[str, tuple[int, ...], tuple[str, ...]]] = [
        (member["key"], tuple(member["shape"]), ())
        for pair in fixture["pairs"]
        for member in (pair["a"], pair["b"])
    ]
    inventory = inspect_lora_inventory(weights_manifest(tensors))
    expected_pairs = sorted(fixture["pairs"], key=lambda pair: (pair["target"], pair["kind"]))

    assert fixture["schema"] == "peftlint.peft-source-fixture.v1"
    assert fixture["peft_version"] == "0.19.1"
    assert fixture["commit"] == "ba6a19060d6ab54a87538a6e77e3e4d5a907375b"
    assert set(fixture["sources"]) == {
        "auxiliary_wrappers",
        "lora_shapes",
        "saved_key_rewrite",
    }
    assert tuple((pair.kind.value, pair.target) for pair in inventory.pairs) == tuple(
        (pair["kind"], pair["target"]) for pair in expected_pairs
    )
    assert tuple((pair.a.shape, pair.b.shape) for pair in inventory.pairs) == tuple(
        (tuple(pair["a"]["shape"]), tuple(pair["b"]["shape"])) for pair in expected_pairs
    )

    unknown = inspect_lora_inventory(
        weights_manifest([(name, (1,), ()) for name in fixture["unclassified_keys"]])
    )
    assert all(tensor.role is LoraTensorRole.UNCLASSIFIED for tensor in unknown.tensors)


def test_seeded_terminal_key_corpus_is_exact_and_order_independent() -> None:
    rng = random.Random(0x50454654)
    suffixes = (
        (".lora_A.weight", LoraTensorRole.LINEAR_A),
        (".lora_B.weight", LoraTensorRole.LINEAR_B),
        (".lora_embedding_A", LoraTensorRole.EMBEDDING_A),
        (".lora_embedding_B", LoraTensorRole.EMBEDDING_B),
    )
    components = ("q_proj", "v_proj", "lora_Aux", "caf\u00e9", "cafe\u0301")
    tensors: list[tuple[str, tuple[int, ...], tuple[str, ...]]] = []
    expected: dict[str, tuple[LoraTensorRole, str | None]] = {}
    for index in range(128):
        target = f"model.layers.{index}.{rng.choice(components)}"
        suffix, role = rng.choice(suffixes)
        canonical = f"{target}{suffix}"
        if role is LoraTensorRole.LINEAR_A:
            near_match = f"{target}.lora_A.default.weight"
        elif role is LoraTensorRole.LINEAR_B:
            near_match = f"{target}.lora_B.bias"
        elif role is LoraTensorRole.EMBEDDING_A:
            near_match = f"{target}.lora_embedding_A.weight"
        else:
            near_match = f"{target}.lora_embedding_B.default"
        tensors.extend(((canonical, (1,), ()), (near_match, (1,), ())))
        expected[canonical] = (role, target)
        expected[near_match] = (LoraTensorRole.UNCLASSIFIED, None)
    rng.shuffle(tensors)

    inventory = inspect_lora_inventory(weights_manifest(tensors))

    assert len(inventory.tensors) == 256
    assert {tensor.name: (tensor.role, tensor.target) for tensor in inventory.tensors} == expected


def test_metadata_does_not_change_semantic_inventory() -> None:
    source = weights_manifest(
        [
            ("model.q.lora_A.weight", (4, 7), ()),
            ("model.q.lora_B.weight", (9, 4), ()),
        ]
    )
    with_metadata = SafetensorsManifest(
        plan=source.plan,
        tensors=source.tensors,
        metadata=(("format", "pt"),),
        metadata_form=MetadataForm.OBJECT,
        notices=source.notices,
    )

    plain = inspect_lora_inventory(source)
    annotated = inspect_lora_inventory(with_metadata)

    assert (plain.tensors, plain.pairs, plain.issues) == (
        annotated.tensors,
        annotated.pairs,
        annotated.issues,
    )


def test_unicode_normalization_is_never_applied_to_targets() -> None:
    composed = "model.caf\u00e9"
    decomposed = "model.cafe\u0301"
    inventory = inspect_lora_inventory(
        weights_manifest(
            [
                (f"{composed}.lora_A.weight", (4, 7), ()),
                (f"{composed}.lora_B.weight", (9, 4), ()),
                (f"{decomposed}.lora_A.weight", (4, 7), ()),
                (f"{decomposed}.lora_B.weight", (9, 4), ()),
            ]
        )
    )

    assert tuple(pair.target for pair in inventory.pairs) == (decomposed, composed)
    assert inventory.pairs[0].target != inventory.pairs[1].target


def test_inventory_ignores_physical_storage_order() -> None:
    a = ("model.q_proj.lora_A.weight", (8, 64), ())
    b = ("model.q_proj.lora_B.weight", (32, 8), ())

    left = inspect_lora_inventory(weights_manifest([a, b]))
    right = inspect_lora_inventory(weights_manifest([b, a]))

    assert (left.tensors, left.pairs, left.issues) == (
        right.tensors,
        right.pairs,
        right.issues,
    )


@pytest.mark.parametrize(
    "name",
    [
        "model.q_proj.lora_A",
        "model.q_proj.lora_B.bias",
        "model.q_proj.lora_embedding_A.weight",
        "model.q_proj.lora_A.default.weight",
        "model.q_proj.lora_A.weight.extra",
        "model.q_proj.LORA_A.weight",
        "lora_A.weight",
        ".lora_A.weight",
        "model..q_proj.lora_A.weight",
        "model.lora_A.named.q_proj.lora_B.weight",
        "model.q_proj.lora_magnitude_vector",
        "model.lm_head.weight",
    ],
)
def test_near_matches_and_ambiguous_markers_stay_unclassified(name: str) -> None:
    inventory = inspect_lora_inventory(weights_manifest([(name, (1,), ())]))

    assert inventory.tensors[0].role is LoraTensorRole.UNCLASSIFIED
    assert inventory.tensors[0].target is None
    assert tuple(issue.kind for issue in inventory.issues) == (
        LoraInventoryIssueKind.UNCLASSIFIED_TENSOR,
    )
    assert inventory.issues[0].logical_path == evidence_path(name)


@pytest.mark.parametrize("name", ["", " ", "\t"])
def test_any_valid_raw_tensor_key_has_an_injective_evidence_scope(name: str) -> None:
    inventory = inspect_lora_inventory(weights_manifest([(name, (1,), ())]))

    assert inventory.tensors[0].role is LoraTensorRole.UNCLASSIFIED
    assert inventory.tensors[0].evidence_path == evidence_path(name)
    assert inventory.issues[0].logical_path == evidence_path(name)


def test_reserved_roles_are_exact_path_components() -> None:
    accepted = inspect_lora_inventory(
        weights_manifest(
            [
                ("model.lora_Aux.q.lora_A.weight", (4, 7), ()),
                ("model.lora_Aux.q.lora_B.weight", (9, 4), ()),
            ]
        )
    )
    rejected = inspect_lora_inventory(
        weights_manifest(
            [
                ("lora_A.q.lora_A.weight", (4, 7), ()),
                ("lora_A.q.lora_B.weight", (9, 4), ()),
            ]
        )
    )

    assert accepted.pairs[0].target == "model.lora_Aux.q"
    assert accepted.issues == ()
    assert rejected.pairs == ()
    assert all(tensor.role is LoraTensorRole.UNCLASSIFIED for tensor in rejected.tensors)


def test_target_prefixes_are_preserved_without_normalization() -> None:
    inventory = inspect_lora_inventory(
        weights_manifest(
            [
                ("base_model.model.default.q_proj.lora_A.weight", (4, 7), ()),
                ("base_model.model.default.q_proj.lora_B.weight", (9, 4), ()),
            ]
        )
    )

    assert inventory.pairs[0].target == "base_model.model.default.q_proj"


def test_orphans_are_scoped_to_their_raw_member_keys() -> None:
    inventory = inspect_lora_inventory(
        weights_manifest(
            [
                ("model.a.lora_A.weight", (4, 7), ()),
                ("model.b.lora_embedding_B", (9, 4), ()),
            ]
        )
    )

    assert inventory.pairs == ()
    assert tuple((issue.kind, issue.logical_path) for issue in inventory.issues) == (
        (LoraInventoryIssueKind.ORPHAN_MEMBER, 'tensor:"model.a.lora_A.weight"'),
        (LoraInventoryIssueKind.ORPHAN_MEMBER, 'tensor:"model.b.lora_embedding_B"'),
    )


def test_linear_and_embedding_members_at_one_target_are_not_guessed() -> None:
    inventory = inspect_lora_inventory(
        weights_manifest(
            [
                ("model.shared.lora_A.weight", (4, 7), ()),
                ("model.shared.lora_B.weight", (9, 4), ()),
                ("model.shared.lora_embedding_A", (4, 11), ()),
            ]
        )
    )

    assert tuple(pair.kind for pair in inventory.pairs) == (LoraPairKind.LINEAR,)
    assert {issue.kind for issue in inventory.issues} == {
        LoraInventoryIssueKind.MIXED_PAIR_KIND,
        LoraInventoryIssueKind.ORPHAN_MEMBER,
    }
    assert (
        next(
            issue.logical_path
            for issue in inventory.issues
            if issue.kind is LoraInventoryIssueKind.MIXED_PAIR_KIND
        )
        == 'target:"model.shared"'
    )


def test_retained_tensor_extensions_are_explicit_inventory_issues() -> None:
    inventory = inspect_lora_inventory(
        weights_manifest(
            [
                ("model.q.lora_A.weight", (4, 7), ("future",)),
                ("model.q.lora_B.weight", (9, 4), ()),
            ]
        )
    )

    assert len(inventory.pairs) == 1
    assert tuple(issue.kind for issue in inventory.issues) == (
        LoraInventoryIssueKind.UNKNOWN_TENSOR_FIELDS,
    )
    assert inventory.issues[0].logical_path == 'tensor:"model.q.lora_A.weight"'


def test_empty_extension_field_name_stays_unknown_instead_of_crashing() -> None:
    inventory = inspect_lora_inventory(
        weights_manifest(
            [
                ("model.q.lora_A.weight", (4, 7), ("",)),
                ("model.q.lora_B.weight", (9, 4), ()),
            ]
        )
    )

    assert tuple(issue.kind for issue in inventory.issues) == (
        LoraInventoryIssueKind.UNKNOWN_TENSOR_FIELDS,
    )


def test_empty_manifest_is_not_vacuously_closed() -> None:
    inventory = inspect_lora_inventory(weights_manifest([]))

    assert inventory.tensors == ()
    assert inventory.pairs == ()
    assert tuple(issue.kind for issue in inventory.issues) == (
        LoraInventoryIssueKind.EMPTY_INVENTORY,
    )
    assert inventory.issues[0].logical_path is None


def test_public_inventory_is_frozen_owned_and_content_redacted() -> None:
    marker = "private-target"
    source = weights_manifest(
        [
            (f"{marker}.lora_A.weight", (4, 7), ()),
            (f"{marker}.lora_B.weight", (9, 4), ()),
        ]
    )
    inventory = inspect_lora_inventory(source)

    assert inventory.weights is not source
    assert inventory.weights.tensors is not source.tensors
    assert marker not in repr(inventory)
    assert marker not in repr(inventory.pairs[0])
    assert marker not in repr(inventory.pairs[0].a)
    with pytest.raises(FrozenInstanceError):
        inventory.schema = "changed"  # type: ignore[misc]
    for value in (
        inventory,
        inventory.tensors[0],
        inventory.pairs[0],
    ):
        with pytest.raises(TypeError, match="is immutable"):
            value.__setstate__(object())


def test_inventory_rejects_inconsistent_aggregate_state() -> None:
    inventory = inspect_lora_inventory(
        weights_manifest(
            [
                ("model.q.lora_A.weight", (4, 7), ()),
                ("model.q.lora_B.weight", (9, 4), ()),
            ]
        )
    )

    with pytest.raises(ValueError, match="relationships are inconsistent"):
        replace(inventory, pairs=())
    with pytest.raises(ValueError, match="canonical key order"):
        LoraInventory(
            schema=LORA_INVENTORY_SCHEMA,
            weights=inventory.weights,
            tensors=tuple(reversed(inventory.tensors)),
            pairs=inventory.pairs,
            issues=inventory.issues,
        )


class AlwaysEqual:
    def __eq__(self, _other: object) -> bool:
        return True


def test_equality_coercion_cannot_bypass_exact_string_fields() -> None:
    inventory = inspect_lora_inventory(
        weights_manifest(
            [
                ("model.q.lora_A.weight", (4, 7), ()),
                ("model.q.lora_B.weight", (9, 4), ()),
            ]
        )
    )

    with pytest.raises(TypeError, match="tensor target must be a string"):
        replace(inventory.tensors[0], target=AlwaysEqual())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="inventory schema must be a string"):
        replace(inventory, schema=AlwaysEqual())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    [
        ("shape", [4, 7]),
        ("_nbytes", 99),
        ("unknown_fields", ["future"]),
    ],
)
def test_inspection_revalidates_forged_source_manifest_state(
    field_name: str, forged_value: object
) -> None:
    source = weights_manifest(
        [
            ("model.q.lora_A.weight", (4, 7), ()),
            ("model.q.lora_B.weight", (9, 4), ()),
        ]
    )
    object.__setattr__(source.tensors[0], field_name, forged_value)

    with pytest.raises((TypeError, ValueError)):
        inspect_lora_inventory(source)


def test_public_inventory_surface_is_payload_free_and_exported_at_root() -> None:
    signature = inspect.signature(inspect_lora_inventory)

    assert tuple(signature.parameters) == ("weights",)
    assert tuple(inventory_module.__all__) == (
        "LORA_INVENTORY_SCHEMA",
        "LoraInventory",
        "LoraInventoryIssue",
        "LoraInventoryIssueKind",
        "LoraPair",
        "LoraPairKind",
        "LoraTensor",
        "LoraTensorRole",
        "inspect_lora_inventory",
    )
    assert peftlint.inspect_lora_inventory is inventory_module.inspect_lora_inventory
    for forbidden in ("path", "payload", "source", "model", "url"):
        assert forbidden not in signature.parameters


def test_inspection_requires_a_validated_manifest_value() -> None:
    with pytest.raises(TypeError, match="weights must be SafetensorsManifest"):
        inspect_lora_inventory(object())  # type: ignore[arg-type]
