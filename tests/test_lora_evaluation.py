from __future__ import annotations

import inspect
import json
import re
from collections.abc import Mapping

import pytest

from peftlint import (
    LORA_V1_LOAD_RULES,
    LORA_V1_RULESET,
    AdapterConfigManifest,
    HeaderNotice,
    HeaderReadPlan,
    MetadataForm,
    Profile,
    RuleOutcome,
    SafetensorsDtype,
    SafetensorsLimits,
    SafetensorsManifest,
    Severity,
    TensorManifest,
    Verdict,
    evaluate_lora_inventory,
    inspect_lora_inventory,
    parse_adapter_config,
    summarize_load,
)
from peftlint.evidence import RuleResult

AUDIT_ID = f"audit:sha256:{'a' * 64}"
ARTIFACT = "adapter_model.safetensors@sha256:def"


def config_manifest(**overrides: object) -> AdapterConfigManifest:
    document: dict[str, object] = {
        "peft_type": "LORA",
        "r": 8,
        "target_modules": ["q_proj"],
        **overrides,
    }
    return parse_adapter_config(json.dumps(document, separators=(",", ":")).encode())


def weights_manifest(
    tensors: list[tuple[str, tuple[int, ...], SafetensorsDtype, Mapping[str, object]]],
) -> SafetensorsManifest:
    cursor = 0
    members: list[TensorManifest] = []
    for name, shape, dtype, extensions in tensors:
        elements = 1
        for dimension in shape:
            elements *= dimension
        size = elements * dtype.bits_per_element // 8
        members.append(
            TensorManifest(
                name=name,
                dtype=dtype,
                shape=shape,
                data_offsets=(cursor, cursor + size),
                unknown_fields=tuple(sorted(extensions)),
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


def evaluate(
    tensors: list[tuple[str, tuple[int, ...], SafetensorsDtype, Mapping[str, object]]],
    **config_overrides: object,
) -> tuple[RuleResult, ...]:
    inventory = inspect_lora_inventory(weights_manifest(tensors))
    return evaluate_lora_inventory(
        config_manifest(**config_overrides),
        inventory,
        audit_id=AUDIT_ID,
        artifact=ARTIFACT,
    )


def by_rule(results: tuple[RuleResult, ...], rule_id: str) -> tuple[RuleResult, ...]:
    return tuple(result for result in results if result.rule_id == rule_id)


def outcomes(results: tuple[RuleResult, ...], rule_id: str) -> tuple[RuleOutcome, ...]:
    return tuple(result.outcome for result in by_rule(results, rule_id))


def linear_pair(
    *,
    rank: int = 8,
    a_width: int = 64,
    b_width: int = 32,
    a_shape: tuple[int, ...] | None = None,
    b_shape: tuple[int, ...] | None = None,
    a_dtype: SafetensorsDtype = SafetensorsDtype.F32,
    b_dtype: SafetensorsDtype = SafetensorsDtype.F32,
    a_extensions: Mapping[str, object] | None = None,
    b_extensions: Mapping[str, object] | None = None,
) -> list[tuple[str, tuple[int, ...], SafetensorsDtype, Mapping[str, object]]]:
    return [
        (
            "base_model.model.q_proj.lora_A.weight",
            (rank, a_width) if a_shape is None else a_shape,
            a_dtype,
            {} if a_extensions is None else a_extensions,
        ),
        (
            "base_model.model.q_proj.lora_B.weight",
            (b_width, rank) if b_shape is None else b_shape,
            b_dtype,
            {} if b_extensions is None else b_extensions,
        ),
    ]


def embedding_pair() -> list[tuple[str, tuple[int, ...], SafetensorsDtype, Mapping[str, object]]]:
    return [
        (
            "base_model.model.embed_tokens.lora_embedding_A",
            (8, 32000),
            SafetensorsDtype.BF16,
            {},
        ),
        (
            "base_model.model.embed_tokens.lora_embedding_B",
            (4096, 8),
            SafetensorsDtype.BF16,
            {},
        ),
    ]


def test_linear_and_embedding_pairs_emit_path_scoped_proofs() -> None:
    results = evaluate([*embedding_pair(), *linear_pair()])

    assert outcomes(results, "PL102") == (RuleOutcome.PASS,)
    assert outcomes(results, "PL110") == (RuleOutcome.PASS, RuleOutcome.PASS)
    assert outcomes(results, "PL111") == (RuleOutcome.PASS, RuleOutcome.PASS)
    assert outcomes(results, "PL112") == (RuleOutcome.PASS, RuleOutcome.PASS)
    assert tuple(result.logical_path for result in by_rule(results, "PL111")) == (
        'tensor:"base_model.model.embed_tokens.lora_embedding_A"',
        'tensor:"base_model.model.q_proj.lora_A.weight"',
    )


def test_orphan_is_inventory_closed_but_pair_incompatible() -> None:
    results = evaluate(linear_pair()[:1])

    assert outcomes(results, "PL102") == (RuleOutcome.PASS,)
    assert outcomes(results, "PL110") == (RuleOutcome.CONTRADICTION,)
    assert outcomes(results, "PL111") == (RuleOutcome.UNKNOWN,)
    assert outcomes(results, "PL112") == (RuleOutcome.UNKNOWN,)


@pytest.mark.parametrize(
    ("a_shape", "b_shape"),
    [
        ((), (32, 8)),
        ((8,), (32, 8)),
        ((0, 64), (32, 0)),
        ((8, 0), (32, 8)),
        ((8, 64), (0, 8)),
        ((8, 64), (32, 7)),
    ],
)
def test_malformed_pair_dimensions_never_become_rank_evidence(
    a_shape: tuple[int, ...], b_shape: tuple[int, ...]
) -> None:
    results = evaluate(linear_pair(a_shape=a_shape, b_shape=b_shape))

    assert outcomes(results, "PL102") == (RuleOutcome.PASS,)
    assert outcomes(results, "PL110") == (RuleOutcome.PASS,)
    assert outcomes(results, "PL111") == (RuleOutcome.CONTRADICTION,)
    assert outcomes(results, "PL112") == (RuleOutcome.UNKNOWN,)


@pytest.mark.parametrize(
    ("a_shape", "b_shape"),
    [
        ((8, 64, 3), (32, 8, 1)),
        ((8, 64, 3, 3), (32, 8, 1, 1)),
        ((8, 64, 3, 3, 3), (32, 8, 1, 1, 1)),
    ],
    ids=("conv1d", "conv2d", "conv3d"),
)
def test_convolution_weight_pairs_are_unsupported_not_incompatible(
    a_shape: tuple[int, ...], b_shape: tuple[int, ...]
) -> None:
    results = evaluate(linear_pair(a_shape=a_shape, b_shape=b_shape))

    assert outcomes(results, "PL102") == (RuleOutcome.UNKNOWN,)
    assert outcomes(results, "PL110") == (RuleOutcome.PASS,)
    assert outcomes(results, "PL111") == (RuleOutcome.UNKNOWN,)
    assert outcomes(results, "PL112") == (RuleOutcome.UNKNOWN,)


@pytest.mark.parametrize(
    ("a_shape", "b_shape"),
    [
        ((8, 32000, 1), (4096, 8)),
        ((8, 32000), (4096, 8, 1)),
    ],
)
def test_embedding_suffixes_make_non_matrix_shapes_contradictory(
    a_shape: tuple[int, ...], b_shape: tuple[int, ...]
) -> None:
    tensors: list[tuple[str, tuple[int, ...], SafetensorsDtype, Mapping[str, object]]] = [
        (
            "base_model.model.embed_tokens.lora_embedding_A",
            a_shape,
            SafetensorsDtype.F32,
            {},
        ),
        (
            "base_model.model.embed_tokens.lora_embedding_B",
            b_shape,
            SafetensorsDtype.F32,
            {},
        ),
    ]

    results = evaluate(tensors)

    assert outcomes(results, "PL102") == (RuleOutcome.PASS,)
    assert outcomes(results, "PL110") == (RuleOutcome.PASS,)
    assert outcomes(results, "PL111") == (RuleOutcome.CONTRADICTION,)
    assert outcomes(results, "PL112") == (RuleOutcome.UNKNOWN,)


def test_nonsquare_dimensions_and_different_dtypes_do_not_invent_a_constraint() -> None:
    results = evaluate(
        linear_pair(
            rank=8,
            a_width=17,
            b_width=19,
            a_dtype=SafetensorsDtype.F16,
            b_dtype=SafetensorsDtype.BF16,
        )
    )

    assert outcomes(results, "PL111") == (RuleOutcome.PASS,)
    assert outcomes(results, "PL112") == (RuleOutcome.PASS,)


def test_unclassified_state_changes_only_inventory_closure() -> None:
    results = evaluate(
        [
            (
                "base_model.model.lm_head.weight",
                (32, 64),
                SafetensorsDtype.F32,
                {},
            )
        ]
    )

    assert outcomes(results, "PL102") == (RuleOutcome.UNKNOWN,)
    assert outcomes(results, "PL110") == (RuleOutcome.PASS,)
    assert outcomes(results, "PL111") == (RuleOutcome.PASS,)
    assert outcomes(results, "PL112") == (RuleOutcome.PASS,)


@pytest.mark.parametrize("name", ["", " ", "\t"])
def test_unclassified_edge_case_names_have_valid_rule_scopes(name: str) -> None:
    results = evaluate([(name, (1,), SafetensorsDtype.F32, {})])

    finding = by_rule(results, "PL102")[0]
    assert finding.outcome is RuleOutcome.UNKNOWN
    assert finding.logical_path == f"tensor:{json.dumps(name, ensure_ascii=True)}"


@pytest.mark.parametrize(
    ("a_extensions", "b_extensions", "extension_count"),
    [
        ({"future_a": True}, {}, 1),
        ({}, {"future_b": True}, 1),
        ({"future_a": True}, {"future_b": True}, 2),
    ],
)
def test_unknown_tensor_fields_gate_every_pair_semantic_rule(
    a_extensions: Mapping[str, object],
    b_extensions: Mapping[str, object],
    extension_count: int,
) -> None:
    results = evaluate(linear_pair(a_extensions=a_extensions, b_extensions=b_extensions))

    assert outcomes(results, "PL102") == (RuleOutcome.UNKNOWN,) * extension_count
    assert outcomes(results, "PL110") == (RuleOutcome.UNKNOWN,)
    assert outcomes(results, "PL111") == (RuleOutcome.UNKNOWN,)
    assert outcomes(results, "PL112") == (RuleOutcome.UNKNOWN,)


def test_mixed_linear_and_embedding_target_is_never_guessed() -> None:
    tensors = [
        *linear_pair(),
        (
            "base_model.model.q_proj.lora_embedding_A",
            (8, 64),
            SafetensorsDtype.F32,
            {},
        ),
    ]

    results = evaluate(tensors)

    assert outcomes(results, "PL102") == (RuleOutcome.UNKNOWN,)
    assert outcomes(results, "PL110") == (RuleOutcome.PASS, RuleOutcome.CONTRADICTION)
    assert outcomes(results, "PL111") == (RuleOutcome.PASS, RuleOutcome.UNKNOWN)
    assert outcomes(results, "PL112") == (RuleOutcome.PASS, RuleOutcome.UNKNOWN)


def test_complete_linear_and_embedding_pairs_keep_independent_proofs() -> None:
    prefix = "base_model.model.shared"
    tensors: list[tuple[str, tuple[int, ...], SafetensorsDtype, Mapping[str, object]]] = [
        (f"{prefix}.lora_A.weight", (8, 17), SafetensorsDtype.F32, {}),
        (f"{prefix}.lora_B.weight", (19, 8), SafetensorsDtype.F32, {}),
        (f"{prefix}.lora_embedding_A", (8, 23), SafetensorsDtype.F32, {}),
        (f"{prefix}.lora_embedding_B", (29, 8), SafetensorsDtype.F32, {}),
    ]

    results = evaluate(tensors)

    assert outcomes(results, "PL102") == (RuleOutcome.UNKNOWN,)
    assert outcomes(results, "PL110") == (RuleOutcome.PASS, RuleOutcome.PASS)
    assert outcomes(results, "PL111") == (RuleOutcome.PASS, RuleOutcome.PASS)
    assert outcomes(results, "PL112") == (RuleOutcome.PASS, RuleOutcome.PASS)


def test_mixed_target_and_unknown_fields_have_distinct_evidence_scopes() -> None:
    tensors = [
        *linear_pair(a_extensions={"future": True}),
        (
            "base_model.model.q_proj.lora_embedding_A",
            (8, 64),
            SafetensorsDtype.F32,
            {},
        ),
    ]

    results = evaluate(tensors)
    pl102 = by_rule(results, "PL102")

    assert outcomes(results, "PL102") == (RuleOutcome.UNKNOWN, RuleOutcome.UNKNOWN)
    assert len(pl102) == len(frozenset(result.sort_key for result in pl102))
    assert {result.logical_path for result in pl102} == {
        'target:"base_model.model.q_proj"',
        'tensor:"base_model.model.q_proj.lora_A.weight"',
    }


def test_modules_to_save_requires_base_topology_evidence() -> None:
    results = evaluate(linear_pair(), modules_to_save=["lm_head"])

    assert outcomes(results, "PL102") == (RuleOutcome.UNKNOWN,)
    assert outcomes(results, "PL110") == (RuleOutcome.PASS,)
    assert outcomes(results, "PL111") == (RuleOutcome.PASS,)
    assert outcomes(results, "PL112") == (RuleOutcome.PASS,)


def test_configured_rank_mismatch_is_a_path_scoped_contradiction() -> None:
    results = evaluate(linear_pair(rank=4), r=8)

    finding = by_rule(results, "PL112")[0]
    assert finding.outcome is RuleOutcome.CONTRADICTION
    assert finding.logical_path == 'tensor:"base_model.model.q_proj.lora_A.weight"'
    assert finding.observed == 4
    assert finding.expected == 8


def test_rank_pattern_is_unknown_without_executing_user_regex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = config_manifest(rank_pattern={"^(a+)+$": 4})
    inventory = inspect_lora_inventory(weights_manifest(linear_pair()))

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("user-controlled regex was executed")

    for name in ("compile", "match", "search", "fullmatch"):
        monkeypatch.setattr(re, name, forbidden)

    results = evaluate_lora_inventory(
        config,
        inventory,
        audit_id=AUDIT_ID,
        artifact=ARTIFACT,
    )

    assert outcomes(results, "PL112") == (RuleOutcome.UNKNOWN,)


def test_closed_profile_is_required_before_configured_rank_can_pass() -> None:
    results = evaluate(linear_pair(), use_dora=True)

    assert outcomes(results, "PL102") == (RuleOutcome.UNKNOWN,)
    assert outcomes(results, "PL110") == (RuleOutcome.UNKNOWN,)
    assert outcomes(results, "PL111") == (RuleOutcome.UNKNOWN,)
    assert outcomes(results, "PL112") == (RuleOutcome.UNKNOWN,)


def test_unmodeled_config_suppresses_orphan_and_shape_contradictions() -> None:
    orphan_results = evaluate(linear_pair()[:1], use_dora=True)
    shaped_results = evaluate(linear_pair(a_shape=(8,), b_shape=(32, 8)), use_dora=True)

    for results in (orphan_results, shaped_results):
        assert {result.outcome for result in results} == {RuleOutcome.UNKNOWN}


def test_empty_inventory_semantics_are_explicit() -> None:
    results = evaluate([])

    assert outcomes(results, "PL102") == (RuleOutcome.UNKNOWN,)
    assert outcomes(results, "PL110") == (RuleOutcome.PASS,)
    assert outcomes(results, "PL111") == (RuleOutcome.PASS,)
    assert outcomes(results, "PL112") == (RuleOutcome.PASS,)


def test_evaluation_results_are_canonical_and_scopes_are_unique() -> None:
    tensors = [
        *linear_pair(),
        (
            "z.lora_A.weight",
            (8, 2),
            SafetensorsDtype.F32,
            {},
        ),
        (
            "a.unknown",
            (1,),
            SafetensorsDtype.F32,
            {},
        ),
    ]

    results = evaluate(list(reversed(tensors)))

    assert tuple(result.sort_key for result in results) == tuple(
        sorted(result.sort_key for result in results)
    )
    assert len(results) == len(frozenset(result.sort_key for result in results))


def test_evaluator_accepts_only_value_evidence_not_paths_or_payloads() -> None:
    signature = inspect.signature(evaluate_lora_inventory)

    assert tuple(signature.parameters) == (
        "config",
        "inventory",
        "audit_id",
        "artifact",
    )
    for forbidden in ("path", "payload", "source", "model", "url"):
        assert forbidden not in signature.parameters


def test_evaluator_rejects_unvalidated_input_types() -> None:
    inventory = inspect_lora_inventory(weights_manifest(linear_pair()))

    with pytest.raises(TypeError, match="config must be AdapterConfigManifest"):
        evaluate_lora_inventory(
            object(),  # type: ignore[arg-type]
            inventory,
            audit_id=AUDIT_ID,
            artifact=ARTIFACT,
        )
    with pytest.raises(TypeError, match="inventory must be LoraInventory"):
        evaluate_lora_inventory(
            config_manifest(),
            object(),  # type: ignore[arg-type]
            audit_id=AUDIT_ID,
            artifact=ARTIFACT,
        )


@pytest.mark.parametrize("field_name", ["schema", "declared_peft_type"])
def test_evaluator_revalidates_equality_coercing_config_state(field_name: str) -> None:
    class AlwaysEqual:
        def __eq__(self, _other: object) -> bool:
            return True

    config = config_manifest()
    inventory = inspect_lora_inventory(weights_manifest(linear_pair()))
    object.__setattr__(config, field_name, AlwaysEqual())

    with pytest.raises(TypeError):
        evaluate_lora_inventory(
            config,
            inventory,
            audit_id=AUDIT_ID,
            artifact=ARTIFACT,
        )


def test_evaluator_revalidates_forged_nested_config_and_inventory_state() -> None:
    config = config_manifest()
    assert config.lora is not None
    object.__setattr__(config.lora, "r", 0)
    inventory = inspect_lora_inventory(weights_manifest(linear_pair()))

    with pytest.raises(ValueError):
        evaluate_lora_inventory(
            config,
            inventory,
            audit_id=AUDIT_ID,
            artifact=ARTIFACT,
        )

    valid_config = config_manifest()
    object.__setattr__(inventory.pairs[0].a, "shape", [8, 64])
    with pytest.raises(TypeError):
        evaluate_lora_inventory(
            valid_config,
            inventory,
            audit_id=AUDIT_ID,
            artifact=ARTIFACT,
        )


def test_path_findings_reduce_with_the_rest_of_the_load_profile() -> None:
    tensors = [
        *linear_pair(),
        (
            "base_model.model.q_proj.lora_embedding_A",
            (8, 64),
            SafetensorsDtype.F32,
            {},
        ),
    ]
    results = list(evaluate(tensors))
    inventory_rule_ids = {"PL102", "PL110", "PL111", "PL112"}
    results.extend(
        RuleResult(
            rule_id=rule_id,
            ruleset=LORA_V1_RULESET,
            audit_id=AUDIT_ID,
            profile=Profile.LOAD,
            outcome=RuleOutcome.PASS,
            severity=Severity.INFO,
            artifact=ARTIFACT,
            message=f"fixture pass for {rule_id}",
        )
        for rule_id in LORA_V1_LOAD_RULES
        if rule_id not in inventory_rule_ids
    )

    summary = summarize_load(
        audit_id=AUDIT_ID,
        ruleset=LORA_V1_RULESET,
        results=results,
    )

    assert summary.verdict is Verdict.INCOMPATIBLE
    assert summary.missing_rules == ()
    assert summary.contradicting_rules == ("PL110",)
    assert summary.unknown_rules == ("PL102", "PL111", "PL112")
