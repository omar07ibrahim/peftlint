from __future__ import annotations

from dataclasses import replace

import pytest

from peftlint import (
    LORA_V1_LOAD_RULES,
    LORA_V1_RULESET,
    EvidenceField,
    Profile,
    ProfileSummary,
    RuleOutcome,
    RuleResult,
    Severity,
    Verdict,
    summarize_load,
)

AUDIT_ID = f"audit:sha256:{'a' * 64}"
OTHER_AUDIT_ID = f"audit:sha256:{'b' * 64}"


def result(
    rule_id: str,
    outcome: RuleOutcome = RuleOutcome.PASS,
    *,
    audit_id: str = AUDIT_ID,
    ruleset: str = LORA_V1_RULESET,
    profile: Profile = Profile.LOAD,
    artifact: str = "adapter@sha256:def",
    path: str | None = None,
) -> RuleResult:
    severity = {
        RuleOutcome.PASS: Severity.INFO,
        RuleOutcome.UNKNOWN: Severity.WARNING,
        RuleOutcome.CONTRADICTION: Severity.ERROR,
    }[outcome]
    return RuleResult(
        rule_id=rule_id,
        ruleset=ruleset,
        audit_id=audit_id,
        profile=profile,
        outcome=outcome,
        severity=severity,
        artifact=artifact,
        logical_path=path,
        message=f"evaluated {rule_id}",
    )


def all_pass_results() -> list[RuleResult]:
    return [result(rule_id) for rule_id in LORA_V1_LOAD_RULES]


def summarize(results: list[RuleResult]) -> ProfileSummary:
    return summarize_load(audit_id=AUDIT_ID, ruleset=LORA_V1_RULESET, results=results)


def test_every_registered_rule_must_pass_for_compatible_verdict() -> None:
    summary = summarize(list(reversed(all_pass_results())))

    assert summary.verdict is Verdict.COMPATIBLE
    assert summary.audit_id == AUDIT_ID
    assert summary.ruleset == LORA_V1_RULESET
    assert summary.required_rules == LORA_V1_LOAD_RULES
    assert summary.missing_rules == ()
    assert summary.unknown_rules == ()
    assert summary.contradicting_rules == ()
    assert tuple(item.rule_id for item in summary.results) == LORA_V1_LOAD_RULES


def test_missing_rule_makes_verdict_unknown() -> None:
    results = [item for item in all_pass_results() if item.rule_id != "PL140"]

    summary = summarize(results)

    assert summary.verdict is Verdict.UNKNOWN
    assert summary.missing_rules == ("PL140",)


def test_unknown_rule_makes_verdict_unknown() -> None:
    results = all_pass_results()
    results[0] = replace(results[0], outcome=RuleOutcome.UNKNOWN, severity=Severity.WARNING)

    summary = summarize(results)

    assert summary.verdict is Verdict.UNKNOWN
    assert summary.unknown_rules == ("PL001",)


def test_contradiction_takes_precedence_over_unknown_and_missing() -> None:
    results = [item for item in all_pass_results() if item.rule_id != "PL003"]
    results[0] = replace(results[0], outcome=RuleOutcome.UNKNOWN, severity=Severity.WARNING)
    results[1] = replace(results[1], outcome=RuleOutcome.CONTRADICTION, severity=Severity.ERROR)

    summary = summarize(results)

    assert summary.verdict is Verdict.INCOMPATIBLE
    assert summary.missing_rules == ("PL003",)
    assert summary.unknown_rules == ("PL001",)
    assert summary.contradicting_rules == ("PL002",)


def test_different_artifacts_can_contribute_to_the_same_audit() -> None:
    results = all_pass_results()
    results[0] = replace(results[0], artifact="adapter@sha256:one")
    results[9] = replace(results[9], artifact="base@sha256:two")

    assert summarize(results).verdict is Verdict.COMPATIBLE


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("audit_id", OTHER_AUDIT_ID, "belongs to audit"),
        ("ruleset", "future-v2", "uses ruleset"),
        ("profile", Profile.HOTSWAP, "uses profile 'hotswap'"),
        ("rule_id", "PL999", "is not registered"),
    ],
)
def test_mixed_or_unregistered_results_are_rejected(
    field: str, value: object, message: str
) -> None:
    results = all_pass_results()
    results[0] = replace(results[0], **{field: value})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        summarize(results)


def test_unregistered_summary_ruleset_is_rejected() -> None:
    with pytest.raises(ValueError, match="unregistered ruleset"):
        summarize_load(audit_id=AUDIT_ID, ruleset="future-v2", results=[])


def test_duplicate_result_is_rejected() -> None:
    results = all_pass_results()
    results.append(results[0])

    with pytest.raises(ValueError, match="duplicate result for load rule PL001"):
        summarize(results)


def test_one_rule_can_report_multiple_path_scoped_findings() -> None:
    results = all_pass_results()
    results[8] = replace(results[8], logical_path="model.layers.0.q_proj")
    results.append(
        replace(
            results[8],
            logical_path="model.layers.1.q_proj",
            message="evaluated another tensor pair",
        )
    )

    summary = summarize(list(reversed(results)))

    assert summary.verdict is Verdict.COMPATIBLE
    assert tuple(item.logical_path for item in summary.results if item.rule_id == "PL102") == (
        "model.layers.0.q_proj",
        "model.layers.1.q_proj",
    )


def test_unknown_path_makes_an_otherwise_passing_rule_unknown() -> None:
    results = all_pass_results()
    results[8] = replace(results[8], logical_path="known")
    results.append(
        replace(
            results[8],
            outcome=RuleOutcome.UNKNOWN,
            severity=Severity.WARNING,
            logical_path="unclassified",
        )
    )

    summary = summarize(results)

    assert summary.verdict is Verdict.UNKNOWN
    assert summary.unknown_rules == ("PL102",)
    assert summary.contradicting_rules == ()


def test_contradicting_path_owns_rule_outcome_over_unknown_path() -> None:
    results = all_pass_results()
    results[10] = replace(
        results[10],
        outcome=RuleOutcome.UNKNOWN,
        severity=Severity.WARNING,
        logical_path="unknown",
    )
    results.append(
        replace(
            results[10],
            outcome=RuleOutcome.CONTRADICTION,
            severity=Severity.ERROR,
            logical_path="mismatched",
        )
    )

    summary = summarize(results)

    assert summary.verdict is Verdict.INCOMPATIBLE
    assert summary.unknown_rules == ()
    assert summary.contradicting_rules == ("PL111",)


def test_duplicate_scope_is_rejected_even_when_finding_differs() -> None:
    results = all_pass_results()
    results.append(
        replace(
            results[0],
            outcome=RuleOutcome.UNKNOWN,
            severity=Severity.WARNING,
            message="a conflicting evaluation for the same evidence scope",
        )
    )

    with pytest.raises(ValueError, match="duplicate result for load rule PL001"):
        summarize(results)


@pytest.mark.parametrize("rule_id", ["", "PL1", "PL0000", "pl001", "XX001"])
def test_invalid_rule_id_is_rejected(rule_id: str) -> None:
    with pytest.raises(ValueError, match="invalid rule id"):
        result(rule_id)


def test_non_string_rule_id_is_rejected() -> None:
    with pytest.raises(TypeError, match="rule_id must be a string"):
        replace(result("PL001"), rule_id=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("audit_id", ["", "audit:sha256:abc", f"audit:sha256:{'A' * 64}"])
def test_invalid_audit_id_is_rejected(audit_id: str) -> None:
    with pytest.raises(ValueError, match="audit_id must use"):
        replace(result("PL001"), audit_id=audit_id)


def test_non_string_audit_id_is_rejected() -> None:
    with pytest.raises(TypeError, match="audit_id must be a string"):
        replace(result("PL001"), audit_id=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("profile", "load", "profile must be a Profile"),
        ("outcome", "unknown", "outcome must be a RuleOutcome"),
        ("severity", "warning", "severity must be a Severity"),
    ],
)
def test_enum_fields_are_validated_at_runtime(field: str, value: str, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        replace(result("PL001"), **{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["ruleset", "artifact", "message"])
def test_required_text_fields_must_not_be_blank(field: str) -> None:
    with pytest.raises(ValueError, match=rf"{field} must not be blank"):
        replace(result("PL001"), **{field: "   "})  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["ruleset", "artifact", "message", "logical_path"])
def test_text_fields_reject_non_string_values(field: str) -> None:
    with pytest.raises(TypeError, match=rf"{field} must be a string"):
        replace(result("PL001"), **{field: 1})  # type: ignore[arg-type]


def test_logical_path_must_not_be_blank_when_present() -> None:
    with pytest.raises(ValueError, match="logical_path must not be blank"):
        replace(result("PL001"), logical_path=" ")


def test_witness_fields_are_unique_and_sorted() -> None:
    item = replace(
        result("PL001"),
        witness=(EvidenceField("z", 2), EvidenceField("a", "value")),
        observed=2,
        expected=3,
    )

    assert item.witness == (EvidenceField("a", "value"), EvidenceField("z", 2))
    assert item.observed == 2
    assert item.expected == 3

    with pytest.raises(ValueError, match="witness field names must be unique"):
        replace(
            item,
            witness=(EvidenceField("rank", 8), EvidenceField("rank", 16)),
        )


def test_witness_must_be_a_tuple_of_evidence_fields() -> None:
    with pytest.raises(TypeError, match="witness must be a tuple"):
        replace(result("PL001"), witness=[EvidenceField("rank", 8)])  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["observed", "expected"])
def test_observed_and_expected_accept_only_canonical_json_scalars(field: str) -> None:
    with pytest.raises(TypeError, match=rf"{field} must be a JSON"):
        replace(result("PL001"), **{field: 1.5})  # type: ignore[arg-type]
