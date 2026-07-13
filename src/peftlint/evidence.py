"""Typed rule results and fail-closed load-verdict reduction."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

_RULE_ID_PATTERN = re.compile(r"PL[0-9]{3}\Z")
_AUDIT_ID_PATTERN = re.compile(r"audit:sha256:[0-9a-f]{64}\Z")

LORA_V1_RULESET = "peft-0.19.1-lora-v1"
LORA_V1_LOAD_RULES = (
    "PL001",
    "PL002",
    "PL003",
    "PL004",
    "PL010",
    "PL011",
    "PL100",
    "PL101",
    "PL102",
    "PL110",
    "PL111",
    "PL112",
    "PL120",
    "PL121",
    "PL122",
    "PL130",
    "PL140",
)

_LOAD_RULES_BY_RULESET = {LORA_V1_RULESET: LORA_V1_LOAD_RULES}

EvidenceScalar: TypeAlias = str | int | bool | None


class Profile(StrEnum):
    """Compatibility question answered by a set of rules."""

    LOAD = "load"
    HOTSWAP = "hotswap"


class RuleOutcome(StrEnum):
    """Outcome of evaluating one rule against available evidence."""

    PASS = "pass"
    CONTRADICTION = "contradiction"
    UNKNOWN = "unknown"


class Severity(StrEnum):
    """Importance assigned to a rule result in human and machine reports."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Verdict(StrEnum):
    """Aggregate conclusion for one compatibility profile."""

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


@dataclass(frozen=True, order=True, slots=True)
class EvidenceField:
    """One safe scalar in a structural witness."""

    name: str
    value: EvidenceScalar

    def __post_init__(self) -> None:
        _require_text("witness field name", self.name)
        _validate_scalar("witness field value", self.value)


@dataclass(frozen=True, slots=True)
class RuleResult:
    """A single rule evaluation bound to one immutable audit scope."""

    rule_id: str
    ruleset: str
    audit_id: str
    profile: Profile
    outcome: RuleOutcome
    severity: Severity
    artifact: str
    message: str
    logical_path: str | None = None
    witness: tuple[EvidenceField, ...] = ()
    observed: EvidenceScalar = None
    expected: EvidenceScalar = None

    def __post_init__(self) -> None:
        _validate_rule_id(self.rule_id)
        _require_text("ruleset", self.ruleset)
        _validate_audit_id(self.audit_id)
        _require_enum("profile", self.profile, Profile)
        _require_enum("outcome", self.outcome, RuleOutcome)
        _require_enum("severity", self.severity, Severity)
        _require_text("artifact", self.artifact)
        _require_text("message", self.message)
        if self.logical_path is not None:
            _require_text("logical_path", self.logical_path)

        if not isinstance(self.witness, tuple) or not all(
            isinstance(field, EvidenceField) for field in self.witness
        ):
            raise TypeError("witness must be a tuple of EvidenceField values")

        names = tuple(field.name for field in self.witness)
        if len(names) != len(frozenset(names)):
            raise ValueError("witness field names must be unique")
        object.__setattr__(self, "witness", tuple(sorted(self.witness)))

        _validate_scalar("observed", self.observed)
        _validate_scalar("expected", self.expected)

    @property
    def sort_key(self) -> tuple[str, str, str, str]:
        """Return the stable order used by evidence serialization."""

        return (
            self.profile.value,
            self.rule_id,
            self.artifact,
            self.logical_path or "",
        )


@dataclass(frozen=True, slots=True)
class ProfileSummary:
    """Load verdict and the rule-level reasons that produced it."""

    audit_id: str
    ruleset: str
    profile: Profile
    verdict: Verdict
    required_rules: tuple[str, ...]
    results: tuple[RuleResult, ...]
    missing_rules: tuple[str, ...]
    unknown_rules: tuple[str, ...]
    contradicting_rules: tuple[str, ...]


def summarize_load(
    *,
    audit_id: str,
    ruleset: str,
    results: Iterable[RuleResult],
) -> ProfileSummary:
    """Reduce one audit's load results under a registered ruleset.

    The registered ruleset owns the mandatory rule list. Mixing audits,
    rulesets, profiles, or unregistered rule identifiers is rejected instead of
    being filtered away. A contradiction takes precedence over unknown or
    missing results; every other mandatory rule must explicitly pass before the
    verdict can be compatible.
    """

    _validate_audit_id(audit_id)
    _require_text("ruleset", ruleset)
    try:
        required = _LOAD_RULES_BY_RULESET[ruleset]
    except KeyError as error:
        raise ValueError(f"unregistered ruleset: {ruleset!r}") from error

    required_set = frozenset(required)
    materialized = tuple(results)
    for result in materialized:
        if result.audit_id != audit_id:
            raise ValueError(
                f"result {result.rule_id} belongs to audit {result.audit_id!r}, "
                f"expected {audit_id!r}"
            )
        if result.ruleset != ruleset:
            raise ValueError(
                f"result {result.rule_id} uses ruleset {result.ruleset!r}, expected {ruleset!r}"
            )
        if result.profile is not Profile.LOAD:
            raise ValueError(
                f"result {result.rule_id} uses profile {result.profile.value!r}, expected 'load'"
            )
        if result.rule_id not in required_set:
            raise ValueError(
                f"rule {result.rule_id} is not registered for {ruleset!r} load profile"
            )

    ordered = tuple(sorted(materialized, key=lambda result: result.sort_key))
    by_rule: dict[str, RuleResult] = {}
    for result in ordered:
        if result.rule_id in by_rule:
            raise ValueError(f"duplicate result for load rule {result.rule_id}")
        by_rule[result.rule_id] = result

    missing = tuple(rule_id for rule_id in required if rule_id not in by_rule)
    unknown = tuple(
        rule_id
        for rule_id in required
        if rule_id in by_rule and by_rule[rule_id].outcome is RuleOutcome.UNKNOWN
    )
    contradicting = tuple(
        rule_id
        for rule_id in required
        if rule_id in by_rule and by_rule[rule_id].outcome is RuleOutcome.CONTRADICTION
    )
    passing = tuple(
        rule_id
        for rule_id in required
        if rule_id in by_rule and by_rule[rule_id].outcome is RuleOutcome.PASS
    )

    if contradicting:
        verdict = Verdict.INCOMPATIBLE
    elif missing or unknown:
        verdict = Verdict.UNKNOWN
    elif len(passing) == len(required):
        verdict = Verdict.COMPATIBLE
    else:  # Defensive guard for malformed values that bypassed RuleResult validation.
        raise ValueError("mandatory rule results contain an invalid outcome")

    return ProfileSummary(
        audit_id=audit_id,
        ruleset=ruleset,
        profile=Profile.LOAD,
        verdict=verdict,
        required_rules=required,
        results=ordered,
        missing_rules=missing,
        unknown_rules=unknown,
        contradicting_rules=contradicting,
    )


def _validate_rule_id(rule_id: object) -> None:
    if not isinstance(rule_id, str):
        raise TypeError("rule_id must be a string")
    if _RULE_ID_PATTERN.fullmatch(rule_id) is None:
        raise ValueError(f"invalid rule id: {rule_id!r}")


def _validate_audit_id(audit_id: object) -> None:
    if not isinstance(audit_id, str):
        raise TypeError("audit_id must be a string")
    if _AUDIT_ID_PATTERN.fullmatch(audit_id) is None:
        raise ValueError("audit_id must use the audit:sha256:<64 lowercase hex> format")


def _require_text(field: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} must not be blank")


def _require_enum(field: str, value: object, enum_type: type[StrEnum]) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{field} must be a {enum_type.__name__}")


def _validate_scalar(field: str, value: object) -> None:
    if type(value) not in (str, int, bool, type(None)):
        raise TypeError(f"{field} must be a JSON string, integer, boolean, or null")
