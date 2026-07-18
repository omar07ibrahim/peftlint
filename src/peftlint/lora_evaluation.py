"""Fail-closed rule evaluation for a verified LoRA tensor inventory."""

from __future__ import annotations

from collections import defaultdict

from peftlint.adapter_config import AdapterConfigManifest
from peftlint.evidence import (
    LORA_V1_RULESET,
    EvidenceField,
    Profile,
    RuleOutcome,
    RuleResult,
    Severity,
)
from peftlint.lora_inventory import (
    LoraInventory,
    LoraInventoryIssueKind,
    LoraPair,
    LoraPairKind,
)

_PL102_UNKNOWN_ISSUES = frozenset(
    {
        LoraInventoryIssueKind.EMPTY_INVENTORY,
        LoraInventoryIssueKind.UNCLASSIFIED_TENSOR,
        LoraInventoryIssueKind.UNKNOWN_TENSOR_FIELDS,
        LoraInventoryIssueKind.MIXED_PAIR_KIND,
        LoraInventoryIssueKind.UNMODELED_WEIGHT_ORIENTATION,
    }
)
_PAIR_DEPENDENCY_ISSUES = frozenset(
    {
        LoraInventoryIssueKind.ORPHAN_MEMBER,
    }
)


def evaluate_lora_inventory(
    config: AdapterConfigManifest,
    inventory: LoraInventory,
    *,
    audit_id: str,
    artifact: str,
) -> tuple[RuleResult, ...]:
    """Evaluate PL102, PL110, PL111, and bounded PL112 evidence.

    Inputs are immutable values and are defensively reconstructed. The
    evaluator performs no I/O, payload access, imports of model libraries, or
    execution of user-controlled rank-pattern expressions.
    """

    validated_config = _copy_config(config)
    validated_inventory = _copy_inventory(inventory)
    results = (
        *_evaluate_pl102(validated_config, validated_inventory, audit_id, artifact),
        *_evaluate_pl110(validated_config, validated_inventory, audit_id, artifact),
        *_evaluate_pl111(validated_config, validated_inventory, audit_id, artifact),
        *_evaluate_pl112(validated_config, validated_inventory, audit_id, artifact),
    )
    ordered = tuple(sorted(results, key=lambda result: result.sort_key))
    scopes = tuple(result.sort_key for result in ordered)
    if len(scopes) != len(frozenset(scopes)):
        raise ValueError("LoRA inventory evaluation produced a duplicate evidence scope")
    return ordered


def _evaluate_pl102(
    config: AdapterConfigManifest,
    inventory: LoraInventory,
    audit_id: str,
    artifact: str,
) -> tuple[RuleResult, ...]:
    reasons_by_path: dict[str | None, set[str]] = defaultdict(set)
    for issue in inventory.issues:
        if issue.kind in _PL102_UNKNOWN_ISSUES:
            reasons_by_path[issue.logical_path].add(issue.kind.value)

    if not config.closed_profile:
        reasons_by_path[None].add("configuration_profile_unavailable")
    elif config.lora is not None and config.lora.modules_to_save:
        reasons_by_path[None].add("modules_to_save_requires_base_topology")

    if not reasons_by_path:
        return (
            _result(
                "PL102",
                RuleOutcome.PASS,
                audit_id,
                artifact,
                message="every checkpoint tensor has one modeled saved-state role",
                observed=len(inventory.tensors),
                witness=(EvidenceField("tensor_count", len(inventory.tensors)),),
            ),
        )

    return tuple(
        _result(
            "PL102",
            RuleOutcome.UNKNOWN,
            audit_id,
            artifact,
            logical_path=path,
            message="tensor inventory closure needs evidence outside the bounded model",
            observed="unresolved",
            expected="one modeled role per tensor",
            witness=(
                EvidenceField("reason_count", len(reasons)),
                EvidenceField("reasons", ",".join(sorted(reasons))),
            ),
        )
        for path, reasons in sorted(reasons_by_path.items(), key=lambda item: item[0] or "")
    )


def _evaluate_pl110(
    config: AdapterConfigManifest,
    inventory: LoraInventory,
    audit_id: str,
    artifact: str,
) -> tuple[RuleResult, ...]:
    if not config.closed_profile:
        return (
            _result(
                "PL110",
                RuleOutcome.UNKNOWN,
                audit_id,
                artifact,
                message="pair completeness needs a closed ordinary LoRA profile",
                expected="closed LoRA configuration",
            ),
        )

    results: list[RuleResult] = []
    for issue in inventory.issues:
        if issue.kind not in _PAIR_DEPENDENCY_ISSUES or issue.tensor is None:
            continue
        if issue.tensor.unknown_fields:
            outcome = RuleOutcome.UNKNOWN
            message = "pair completeness is ambiguous for this saved-state member"
        else:
            outcome = RuleOutcome.CONTRADICTION
            message = "saved-state member has no exact opposite LoRA pair member"
        results.append(
            _result(
                "PL110",
                outcome,
                audit_id,
                artifact,
                logical_path=issue.tensor.evidence_path,
                message=message,
                observed=issue.tensor.role.value,
                expected="complete A/B pair",
            )
        )

    for pair in inventory.pairs:
        members_with_extensions = sum(bool(member.unknown_fields) for member in (pair.a, pair.b))
        outcome = RuleOutcome.UNKNOWN if members_with_extensions else RuleOutcome.PASS
        results.append(
            _result(
                "PL110",
                outcome,
                audit_id,
                artifact,
                logical_path=pair.a.evidence_path,
                message=(
                    "pair completeness cannot close across unknown tensor fields"
                    if members_with_extensions
                    else "LoRA pair has one exact A member and one exact B member"
                ),
                observed=2,
                expected=2,
                witness=(
                    EvidenceField("members_with_extensions", members_with_extensions),
                    EvidenceField("pair_kind", pair.kind.value),
                ),
            )
        )

    if results:
        return tuple(results)
    return (
        _result(
            "PL110",
            RuleOutcome.PASS,
            audit_id,
            artifact,
            message="every recognized LoRA member has one exact opposite member",
            observed=len(inventory.pairs),
            witness=(EvidenceField("complete_pair_count", len(inventory.pairs)),),
        ),
    )


def _evaluate_pl111(
    config: AdapterConfigManifest,
    inventory: LoraInventory,
    audit_id: str,
    artifact: str,
) -> tuple[RuleResult, ...]:
    if not config.closed_profile:
        return (
            _result(
                "PL111",
                RuleOutcome.UNKNOWN,
                audit_id,
                artifact,
                message="pair dimensions need a closed ordinary LoRA profile",
                expected="closed LoRA configuration",
            ),
        )

    results: list[RuleResult] = []
    for pair in inventory.pairs:
        results.append(_dimension_result(pair, audit_id, artifact))
    for issue in inventory.issues:
        if issue.kind not in _PAIR_DEPENDENCY_ISSUES or issue.tensor is None:
            continue
        results.append(
            _result(
                "PL111",
                RuleOutcome.UNKNOWN,
                audit_id,
                artifact,
                logical_path=issue.tensor.evidence_path,
                message="pair dimensions need a unique complete LoRA orientation",
                observed=issue.tensor.role.value,
                expected="complete rank-two pair",
            )
        )

    if results:
        return tuple(results)
    return (
        _result(
            "PL111",
            RuleOutcome.PASS,
            audit_id,
            artifact,
            message="all complete LoRA pairs have a valid pinned orientation",
            observed=0,
            witness=(EvidenceField("evaluated_pair_count", 0),),
        ),
    )


def _dimension_result(pair: LoraPair, audit_id: str, artifact: str) -> RuleResult:
    path = pair.a.evidence_path
    if pair.a.unknown_fields or pair.b.unknown_fields:
        return _result(
            "PL111",
            RuleOutcome.UNKNOWN,
            audit_id,
            artifact,
            logical_path=path,
            message="pair dimensions cannot close across unknown tensor fields",
            witness=(EvidenceField("pair_kind", pair.kind.value),),
        )

    if pair.kind is LoraPairKind.WEIGHT and (len(pair.a.shape) > 2 or len(pair.b.shape) > 2):
        return _result(
            "PL111",
            RuleOutcome.UNKNOWN,
            audit_id,
            artifact,
            logical_path=path,
            message="weight-backed pair may use an unmodeled convolution orientation",
            observed=f"{len(pair.a.shape)}:{len(pair.b.shape)}",
            expected="modeled dense or embedding orientation",
            witness=(EvidenceField("pair_kind", pair.kind.value),),
        )

    if len(pair.a.shape) != 2 or len(pair.b.shape) != 2:
        return _result(
            "PL111",
            RuleOutcome.CONTRADICTION,
            audit_id,
            artifact,
            logical_path=path,
            message="LoRA pair members must both be rank-two tensors",
            observed=f"{len(pair.a.shape)}:{len(pair.b.shape)}",
            expected="2:2",
            witness=(EvidenceField("pair_kind", pair.kind.value),),
        )

    a_rank, a_width = pair.a.shape
    b_width, b_rank = pair.b.shape
    witness = (
        EvidenceField("a_inner_width", a_width),
        EvidenceField("a_rank", a_rank),
        EvidenceField("b_inner_rank", b_rank),
        EvidenceField("b_outer_width", b_width),
        EvidenceField("pair_kind", pair.kind.value),
    )
    if 0 in (a_rank, a_width, b_width, b_rank):
        return _result(
            "PL111",
            RuleOutcome.CONTRADICTION,
            audit_id,
            artifact,
            logical_path=path,
            message="LoRA pair dimensions must all be positive",
            observed=0,
            expected="positive dimensions",
            witness=witness,
        )
    if a_rank != b_rank:
        return _result(
            "PL111",
            RuleOutcome.CONTRADICTION,
            audit_id,
            artifact,
            logical_path=path,
            message="LoRA pair members disagree on their inner rank",
            observed=a_rank,
            expected=b_rank,
            witness=witness,
        )
    return _result(
        "PL111",
        RuleOutcome.PASS,
        audit_id,
        artifact,
        logical_path=path,
        message="LoRA pair orientation and inner rank agree",
        observed=a_rank,
        expected=b_rank,
        witness=witness,
    )


def _evaluate_pl112(
    config: AdapterConfigManifest,
    inventory: LoraInventory,
    audit_id: str,
    artifact: str,
) -> tuple[RuleResult, ...]:
    if not config.closed_profile or config.lora is None:
        return (
            _result(
                "PL112",
                RuleOutcome.UNKNOWN,
                audit_id,
                artifact,
                message="configured rank needs a closed ordinary LoRA profile",
                expected="closed LoRA configuration",
            ),
        )

    if config.lora.rank_pattern:
        scoped_paths = tuple(pair.a.evidence_path for pair in inventory.pairs) + tuple(
            issue.tensor.evidence_path
            for issue in inventory.issues
            if issue.kind in _PAIR_DEPENDENCY_ISSUES and issue.tensor is not None
        )
        paths: tuple[str | None, ...] = tuple(sorted(scoped_paths)) or (None,)
        return tuple(
            _result(
                "PL112",
                RuleOutcome.UNKNOWN,
                audit_id,
                artifact,
                logical_path=path,
                message="rank-pattern expressions are retained but not executed",
                observed="pattern-dependent rank",
                expected="bounded exact rank evidence",
                witness=(EvidenceField("pattern_count", len(config.lora.rank_pattern)),),
            )
            for path in paths
        )

    results: list[RuleResult] = []
    for pair in inventory.pairs:
        rank = _proved_rank(pair)
        if rank is None:
            results.append(
                _result(
                    "PL112",
                    RuleOutcome.UNKNOWN,
                    audit_id,
                    artifact,
                    logical_path=pair.a.evidence_path,
                    message="configured rank needs a dimensionally valid LoRA pair",
                    expected=config.lora.r,
                )
            )
            continue
        outcome = RuleOutcome.PASS if rank == config.lora.r else RuleOutcome.CONTRADICTION
        results.append(
            _result(
                "PL112",
                outcome,
                audit_id,
                artifact,
                logical_path=pair.a.evidence_path,
                message=(
                    "observed LoRA rank agrees with the configured rank"
                    if outcome is RuleOutcome.PASS
                    else "observed LoRA rank contradicts the configured rank"
                ),
                observed=rank,
                expected=config.lora.r,
                witness=(EvidenceField("pair_kind", pair.kind.value),),
            )
        )

    for issue in inventory.issues:
        if issue.kind not in _PAIR_DEPENDENCY_ISSUES or issue.tensor is None:
            continue
        results.append(
            _result(
                "PL112",
                RuleOutcome.UNKNOWN,
                audit_id,
                artifact,
                logical_path=issue.tensor.evidence_path,
                message="configured rank needs a unique complete LoRA pair",
                observed=issue.tensor.role.value,
                expected=config.lora.r,
            )
        )

    if results:
        return tuple(results)
    return (
        _result(
            "PL112",
            RuleOutcome.PASS,
            audit_id,
            artifact,
            message="all observed LoRA pair ranks agree with the configured rank",
            observed=0,
            expected=config.lora.r,
            witness=(EvidenceField("evaluated_pair_count", 0),),
        ),
    )


def _proved_rank(pair: LoraPair) -> int | None:
    if pair.a.unknown_fields or pair.b.unknown_fields:
        return None
    if len(pair.a.shape) != 2 or len(pair.b.shape) != 2:
        return None
    a_rank, a_width = pair.a.shape
    b_width, b_rank = pair.b.shape
    if 0 in (a_rank, a_width, b_width, b_rank) or a_rank != b_rank:
        return None
    return a_rank


def _result(
    rule_id: str,
    outcome: RuleOutcome,
    audit_id: str,
    artifact: str,
    *,
    message: str,
    logical_path: str | None = None,
    witness: tuple[EvidenceField, ...] = (),
    observed: str | int | bool | None = None,
    expected: str | int | bool | None = None,
) -> RuleResult:
    severity = {
        RuleOutcome.PASS: Severity.INFO,
        RuleOutcome.UNKNOWN: Severity.WARNING,
        RuleOutcome.CONTRADICTION: Severity.ERROR,
    }[outcome]
    return RuleResult(
        rule_id=rule_id,
        ruleset=LORA_V1_RULESET,
        audit_id=audit_id,
        profile=Profile.LOAD,
        outcome=outcome,
        severity=severity,
        artifact=artifact,
        logical_path=logical_path,
        message=message,
        witness=witness,
        observed=observed,
        expected=expected,
    )


def _copy_config(config: AdapterConfigManifest) -> AdapterConfigManifest:
    if type(config) is not AdapterConfigManifest:
        raise TypeError("config must be AdapterConfigManifest")
    return AdapterConfigManifest(
        schema=config.schema,
        method_status=config.method_status,
        declared_peft_type=config.declared_peft_type,
        explicit_fields=config.explicit_fields,
        lora=config.lora,
        issues=config.issues,
    )


def _copy_inventory(inventory: LoraInventory) -> LoraInventory:
    if type(inventory) is not LoraInventory:
        raise TypeError("inventory must be LoraInventory")
    return LoraInventory(
        schema=inventory.schema,
        weights=inventory.weights,
        tensors=inventory.tensors,
        pairs=inventory.pairs,
        issues=inventory.issues,
    )


__all__ = ["evaluate_lora_inventory"]
