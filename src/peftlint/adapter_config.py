"""Pure, bounded inspection of PEFT ``adapter_config.json`` bytes.

The schema is pinned to PEFT 0.19.1. This module performs no filesystem,
network, import, regular-expression, model, or tensor work. It turns a JSON
document into immutable structural evidence and reports profile-unsupported
fields without executing or retaining their opaque values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, DecimalException
from enum import StrEnum
from typing import Generic, NoReturn, TypeVar, cast

from peftlint._bounded_json import (
    BoundedJsonError,
    BoundedJsonErrorCode,
    FloatLexeme,
    IntegerLexeme,
    InvalidJson,
    JsonLimitExceeded,
    JsonLimits,
    decode_json,
)
from peftlint.evidence import LORA_V1_RULESET

ADAPTER_CONFIG_SCHEMA = LORA_V1_RULESET
"""Schema identifier attached to every adapter-config manifest."""

PINNED_PEFT_VERSION = "0.19.1"
"""PEFT version whose configuration semantics are modeled by this parser."""

_MAX_INTEGER = 2**63 - 1
_MAX_DECIMAL_MAGNITUDE = Decimal(_MAX_INTEGER)
_MAX_DECIMAL_EXPONENT = 9999
_MISSING = object()

_BASE_FIELDS = frozenset(
    {
        "task_type",
        "peft_type",
        "auto_mapping",
        "peft_version",
        "base_model_name_or_path",
        "revision",
        "inference_mode",
    }
)
_LORA_FIELDS = frozenset(
    {
        "r",
        "target_modules",
        "exclude_modules",
        "lora_alpha",
        "lora_dropout",
        "fan_in_fan_out",
        "bias",
        "use_rslora",
        "modules_to_save",
        "init_lora_weights",
        "layers_to_transform",
        "layers_pattern",
        "rank_pattern",
        "alpha_pattern",
        "megatron_config",
        "megatron_core",
        "trainable_token_indices",
        "loftq_config",
        "eva_config",
        "corda_config",
        "lora_ga_config",
        "use_dora",
        "alora_invocation_tokens",
        "use_qalora",
        "qalora_group_size",
        "layer_replication",
        "runtime_config",
        "lora_bias",
        "target_parameters",
        "use_bdlora",
        "arrow_config",
        "ensure_weight_tying",
    }
)
_KNOWN_FIELDS = _BASE_FIELDS | _LORA_FIELDS


class AdapterConfigErrorCode(StrEnum):
    """Stable machine-readable failures produced by document inspection."""

    DOCUMENT_EXCEEDS_POLICY_LIMIT = "document_exceeds_policy_limit"
    DOCUMENT_UTF8 = "document_utf8"
    DOCUMENT_JSON = "document_json"
    DUPLICATE_JSON_KEY = "duplicate_json_key"
    JSON_DEPTH_EXCEEDS_POLICY_LIMIT = "json_depth_exceeds_policy_limit"
    JSON_STRING_EXCEEDS_POLICY_LIMIT = "json_string_exceeds_policy_limit"
    JSON_TOKEN_EXCEEDS_POLICY_LIMIT = "json_token_exceeds_policy_limit"
    JSON_NUMBER_EXCEEDS_POLICY_LIMIT = "json_number_exceeds_policy_limit"
    INVALID_UNICODE_SCALAR = "invalid_unicode_scalar"
    ROOT_NOT_OBJECT = "root_not_object"
    ROOT_FIELD_COUNT_EXCEEDS_POLICY_LIMIT = "root_field_count_exceeds_policy_limit"
    COLLECTION_COUNT_EXCEEDS_POLICY_LIMIT = "collection_count_exceeds_policy_limit"
    NAME_EXCEEDS_POLICY_LIMIT = "name_exceeds_policy_limit"


_ERROR_MESSAGES = {
    AdapterConfigErrorCode.DOCUMENT_EXCEEDS_POLICY_LIMIT: (
        "adapter config exceeds the document inspection limit"
    ),
    AdapterConfigErrorCode.DOCUMENT_UTF8: "adapter config is not valid UTF-8",
    AdapterConfigErrorCode.DOCUMENT_JSON: "adapter config is not valid JSON",
    AdapterConfigErrorCode.DUPLICATE_JSON_KEY: "adapter config contains a duplicate JSON key",
    AdapterConfigErrorCode.JSON_DEPTH_EXCEEDS_POLICY_LIMIT: (
        "adapter config exceeds the JSON nesting limit"
    ),
    AdapterConfigErrorCode.JSON_STRING_EXCEEDS_POLICY_LIMIT: (
        "adapter config exceeds the JSON string limit"
    ),
    AdapterConfigErrorCode.JSON_TOKEN_EXCEEDS_POLICY_LIMIT: (
        "adapter config exceeds the JSON token limit"
    ),
    AdapterConfigErrorCode.JSON_NUMBER_EXCEEDS_POLICY_LIMIT: (
        "adapter config exceeds the JSON number limit"
    ),
    AdapterConfigErrorCode.INVALID_UNICODE_SCALAR: (
        "adapter config contains an invalid Unicode scalar"
    ),
    AdapterConfigErrorCode.ROOT_NOT_OBJECT: "adapter config root must be a JSON object",
    AdapterConfigErrorCode.ROOT_FIELD_COUNT_EXCEEDS_POLICY_LIMIT: (
        "adapter config exceeds the root field limit"
    ),
    AdapterConfigErrorCode.COLLECTION_COUNT_EXCEEDS_POLICY_LIMIT: (
        "adapter config exceeds the collection item limit"
    ),
    AdapterConfigErrorCode.NAME_EXCEEDS_POLICY_LIMIT: (
        "adapter config exceeds the retained name limit"
    ),
}

_INVALID_ERROR_CODES = frozenset(
    {
        AdapterConfigErrorCode.DOCUMENT_UTF8,
        AdapterConfigErrorCode.DOCUMENT_JSON,
        AdapterConfigErrorCode.DUPLICATE_JSON_KEY,
        AdapterConfigErrorCode.INVALID_UNICODE_SCALAR,
        AdapterConfigErrorCode.ROOT_NOT_OBJECT,
    }
)
_LIMIT_ERROR_CODES = frozenset(set(AdapterConfigErrorCode) - _INVALID_ERROR_CODES)


class AdapterConfigInspectionError(Exception):
    """Base class for classified, content-redacted inspection failures."""

    code: AdapterConfigErrorCode

    def __init__(self, code: AdapterConfigErrorCode) -> None:
        if type(self) is AdapterConfigInspectionError:
            raise TypeError("AdapterConfigInspectionError is an abstract base class")
        if type(code) is not AdapterConfigErrorCode:
            raise TypeError("adapter config error code must be AdapterConfigErrorCode")
        self.code = code
        super().__init__(_ERROR_MESSAGES[code])


class InvalidAdapterConfig(AdapterConfigInspectionError):
    """The supplied bytes contradict the adapter-config document format."""

    rule_id = "PL001"

    def __init__(self, code: AdapterConfigErrorCode) -> None:
        _require_error_category(code, _INVALID_ERROR_CODES, "invalid adapter config")
        super().__init__(code)


class AdapterConfigLimitExceeded(AdapterConfigInspectionError):
    """Inspection stopped at a configured local resource boundary."""

    limit: int

    def __init__(self, code: AdapterConfigErrorCode, *, limit: int) -> None:
        _require_error_category(code, _LIMIT_ERROR_CODES, "inspection limit")
        if type(limit) is not int:
            raise TypeError("limit must be an integer")
        if limit < 0:
            raise ValueError("limit must not be negative")
        self.limit = limit
        super().__init__(code)


def _require_error_category(
    code: AdapterConfigErrorCode,
    allowed: frozenset[AdapterConfigErrorCode],
    category: str,
) -> None:
    if type(code) is not AdapterConfigErrorCode:
        raise TypeError("adapter config error code must be AdapterConfigErrorCode")
    if code not in allowed:
        raise ValueError(f"{code.value} is not an {category} error code")


@dataclass(frozen=True, slots=True)
class AdapterConfigLimits:
    """Resource limits enforced before and during JSON materialization."""

    max_document_bytes: int = 1024 * 1024
    max_json_depth: int = 32
    max_json_string_chars: int = 65_536
    max_json_tokens: int = 50_000
    max_json_number_chars: int = 128
    max_root_fields: int = 256
    max_collection_items: int = 10_000
    max_name_bytes: int = 4096

    def __post_init__(self) -> None:
        for name, value in (
            ("max_document_bytes", self.max_document_bytes),
            ("max_json_depth", self.max_json_depth),
            ("max_json_string_chars", self.max_json_string_chars),
            ("max_json_tokens", self.max_json_tokens),
            ("max_json_number_chars", self.max_json_number_chars),
            ("max_root_fields", self.max_root_fields),
            ("max_collection_items", self.max_collection_items),
            ("max_name_bytes", self.max_name_bytes),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")

    def __setstate__(self, _state: object) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is immutable")


DEFAULT_ADAPTER_CONFIG_LIMITS = AdapterConfigLimits()


class AdapterMethodStatus(StrEnum):
    """Whether the document identifies the pinned LoRA method."""

    SUPPORTED_LORA = "supported_lora"
    MISSING = "missing"
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"


class ConfigFieldIssueKind(StrEnum):
    """Why a field cannot contribute to an ordinary pinned LoRA profile."""

    UNKNOWN_FIELD = "unknown_field"
    INVALID_TYPE = "invalid_type"
    INVALID_VALUE = "invalid_value"
    UNSUPPORTED_VALUE = "unsupported_value"
    IGNORED_FIELD = "ignored_field"


@dataclass(frozen=True, slots=True)
class ConfigFieldIssue:
    """One redacted root-field classification."""

    field_name: str = field(repr=False)
    kind: ConfigFieldIssueKind

    def __post_init__(self) -> None:
        _require_text("field_name", self.field_name)
        _require_exact_enum("kind", self.kind, ConfigFieldIssueKind)

    @property
    def blocking(self) -> bool:
        """Return whether this issue prevents a closed ordinary profile."""

        return self.kind is not ConfigFieldIssueKind.IGNORED_FIELD

    @property
    def sort_key(self) -> tuple[str, str]:
        """Return the canonical order used in manifests."""

        return self.field_name, self.kind.value

    def __setstate__(self, _state: object) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is immutable")


class PeftTaskType(StrEnum):
    """Task values accepted by PEFT 0.19.1."""

    SEQ_CLS = "SEQ_CLS"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    CAUSAL_LM = "CAUSAL_LM"
    TOKEN_CLS = "TOKEN_CLS"
    QUESTION_ANS = "QUESTION_ANS"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"


class ModuleSelectorKind(StrEnum):
    """Normalized form of a PEFT module selector."""

    DEFAULT = "default"
    ALL_LINEAR = "all_linear"
    REGEX = "regex"
    NAMES = "names"


_STRING_SELECTOR_KINDS = frozenset({ModuleSelectorKind.REGEX, ModuleSelectorKind.ALL_LINEAR})


@dataclass(frozen=True, slots=True)
class ModuleSelector:
    """An immutable module selector; user-controlled text is hidden from repr."""

    kind: ModuleSelectorKind
    values: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        _require_exact_enum("kind", self.kind, ModuleSelectorKind)
        _require_text_tuple("values", self.values, allow_empty_items=False)
        if self.kind in {ModuleSelectorKind.DEFAULT, ModuleSelectorKind.ALL_LINEAR}:
            if self.values:
                raise ValueError(f"{self.kind.value} selector must not contain values")
        elif self.kind is ModuleSelectorKind.REGEX:
            if len(self.values) != 1:
                raise ValueError("regex selector must contain exactly one value")
        elif self.values != tuple(sorted(frozenset(self.values))):
            raise ValueError("named selector values must be sorted and unique")

    def __setstate__(self, _state: object) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is immutable")


class LoraBiasMode(StrEnum):
    """Bias modes understood by the pinned LoRA configuration."""

    NONE = "none"
    ALL = "all"
    LORA_ONLY = "lora_only"


class LoraInitializerKind(StrEnum):
    """Initialization values accepted by PEFT 0.19.1."""

    DEFAULT = "default"
    RANDOM = "random"
    GAUSSIAN = "gaussian"
    EVA = "eva"
    OLORA = "olora"
    PISSA = "pissa"
    PISSA_NITER = "pissa_niter"
    CORDA = "corda"
    LOFTQ = "loftq"
    ORTHOGONAL = "orthogonal"
    LORA_GA = "lora_ga"


@dataclass(frozen=True, slots=True)
class LoraInitializer:
    """Normalized initializer, including the bounded PiSSA iteration count."""

    kind: LoraInitializerKind
    iterations: int | None = None

    def __post_init__(self) -> None:
        _require_exact_enum("kind", self.kind, LoraInitializerKind)
        if self.kind is LoraInitializerKind.PISSA_NITER:
            _require_integer("iterations", self.iterations, minimum=0)
        elif self.iterations is not None:
            raise ValueError("iterations are only valid for pissa_niter")

    def __setstate__(self, _state: object) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is immutable")


@dataclass(frozen=True, slots=True)
class AutoMapping:
    """Canonical PEFT custom-model mapping without importing its library."""

    base_model_class: str = field(repr=False)
    parent_library: str = field(repr=False)

    def __post_init__(self) -> None:
        _require_nonempty_text("base_model_class", self.base_model_class)
        _require_nonempty_text("parent_library", self.parent_library)

    def __setstate__(self, _state: object) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is immutable")


@dataclass(frozen=True, slots=True)
class RankPatternEntry:
    """One ordered regex-to-rank override; the expression is never executed here."""

    pattern: str = field(repr=False)
    rank: int

    def __post_init__(self) -> None:
        _require_nonempty_text("pattern", self.pattern)
        _require_integer("rank", self.rank, minimum=1)

    def __setstate__(self, _state: object) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is immutable")


@dataclass(frozen=True, slots=True)
class AlphaPatternEntry:
    """One ordered regex-to-alpha override; the expression is never executed here."""

    pattern: str = field(repr=False)
    alpha: Decimal

    def __post_init__(self) -> None:
        _require_decimal("alpha", self.alpha)
        _require_nonempty_text("pattern", self.pattern)

    def __setstate__(self, _state: object) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is immutable")


@dataclass(frozen=True, slots=True)
class LoraConfigProfile:
    """Normalized ordinary LoRA fields needed by later static checks."""

    task_type: PeftTaskType | None
    auto_mapping: AutoMapping | None = field(repr=False)
    peft_version: str = field(repr=False)
    peft_version_was_declared: bool
    base_model_name_or_path: str | None = field(repr=False)
    revision: str | None = field(repr=False)
    inference_mode: bool
    r: int
    target_modules: ModuleSelector
    exclude_modules: ModuleSelector
    lora_alpha: Decimal
    lora_dropout: Decimal
    fan_in_fan_out: bool
    bias: LoraBiasMode
    use_rslora: bool
    modules_to_save: tuple[str, ...] | None = field(repr=False)
    initializer: LoraInitializer
    layers_to_transform: tuple[int, ...] | None
    layers_pattern: tuple[str, ...] | None = field(repr=False)
    rank_pattern: tuple[RankPatternEntry, ...] = field(repr=False)
    alpha_pattern: tuple[AlphaPatternEntry, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if self.task_type is not None:
            _require_exact_enum("task_type", self.task_type, PeftTaskType)
        if self.auto_mapping is not None and type(self.auto_mapping) is not AutoMapping:
            raise TypeError("auto_mapping must be AutoMapping or None")
        if self.auto_mapping is not None:
            object.__setattr__(
                self,
                "auto_mapping",
                AutoMapping(
                    self.auto_mapping.base_model_class,
                    self.auto_mapping.parent_library,
                ),
            )
        _require_nonempty_text("peft_version", self.peft_version)
        _require_bool("peft_version_was_declared", self.peft_version_was_declared)
        _require_optional_text("base_model_name_or_path", self.base_model_name_or_path)
        _require_optional_text("revision", self.revision)
        _require_bool("inference_mode", self.inference_mode)
        _require_integer("r", self.r, minimum=1)
        if type(self.target_modules) is not ModuleSelector:
            raise TypeError("target_modules must be ModuleSelector")
        if type(self.exclude_modules) is not ModuleSelector:
            raise TypeError("exclude_modules must be ModuleSelector")
        object.__setattr__(
            self,
            "target_modules",
            ModuleSelector(self.target_modules.kind, self.target_modules.values),
        )
        object.__setattr__(
            self,
            "exclude_modules",
            ModuleSelector(self.exclude_modules.kind, self.exclude_modules.values),
        )
        _require_decimal("lora_alpha", self.lora_alpha)
        _require_decimal("lora_dropout", self.lora_dropout, minimum=Decimal(0), maximum=Decimal(1))
        _require_bool("fan_in_fan_out", self.fan_in_fan_out)
        _require_exact_enum("bias", self.bias, LoraBiasMode)
        _require_bool("use_rslora", self.use_rslora)
        if self.modules_to_save is not None:
            _require_text_tuple("modules_to_save", self.modules_to_save, allow_empty_items=False)
        if type(self.initializer) is not LoraInitializer:
            raise TypeError("initializer must be LoraInitializer")
        object.__setattr__(
            self,
            "initializer",
            LoraInitializer(self.initializer.kind, self.initializer.iterations),
        )
        if self.layers_to_transform is not None:
            _require_integer_tuple("layers_to_transform", self.layers_to_transform, minimum=0)
            if self.layers_to_transform != tuple(sorted(frozenset(self.layers_to_transform))):
                raise ValueError("layers_to_transform must be sorted and unique")
        if self.layers_pattern is not None:
            _require_text_tuple("layers_pattern", self.layers_pattern, allow_empty_items=False)
        _require_exact_tuple("rank_pattern", self.rank_pattern, RankPatternEntry)
        _require_exact_tuple("alpha_pattern", self.alpha_pattern, AlphaPatternEntry)
        object.__setattr__(
            self,
            "rank_pattern",
            tuple(RankPatternEntry(entry.pattern, entry.rank) for entry in self.rank_pattern),
        )
        object.__setattr__(
            self,
            "alpha_pattern",
            tuple(AlphaPatternEntry(entry.pattern, entry.alpha) for entry in self.alpha_pattern),
        )
        if self.target_modules.kind in _STRING_SELECTOR_KINDS:
            if self.layers_to_transform is not None or self.layers_pattern is not None:
                raise ValueError("string target_modules cannot use layer filters")
        if self.target_modules.kind is ModuleSelectorKind.NAMES and not self.target_modules.values:
            raise ValueError("ordinary profile requires non-empty named target_modules")
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("layers_pattern requires non-empty layers_to_transform")

    def __setstate__(self, _state: object) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is immutable")


@dataclass(frozen=True, slots=True)
class AdapterConfigManifest:
    """Immutable adapter-config evidence with opaque values redacted from repr."""

    schema: str
    method_status: AdapterMethodStatus
    declared_peft_type: str | None = field(repr=False)
    explicit_fields: tuple[str, ...] = field(repr=False)
    lora: LoraConfigProfile | None = field(repr=False)
    issues: tuple[ConfigFieldIssue, ...]

    def __post_init__(self) -> None:
        if type(self.schema) is not str:
            raise TypeError("adapter config schema must be a string")
        if self.schema != ADAPTER_CONFIG_SCHEMA:
            raise ValueError("schema must identify the pinned adapter config schema")
        if self.declared_peft_type is not None and type(self.declared_peft_type) is not str:
            raise TypeError("declared_peft_type must be a string or None")
        _require_exact_enum("method_status", self.method_status, AdapterMethodStatus)
        _require_text_tuple("explicit_fields", self.explicit_fields, allow_empty_items=True)
        if self.explicit_fields != tuple(sorted(frozenset(self.explicit_fields))):
            raise ValueError("explicit_fields must be sorted and unique")
        if self.lora is not None and type(self.lora) is not LoraConfigProfile:
            raise TypeError("lora must be LoraConfigProfile or None")
        if self.lora is not None:
            object.__setattr__(self, "lora", _validated_lora_profile_copy(self.lora))
        _require_exact_tuple("issues", self.issues, ConfigFieldIssue)
        object.__setattr__(
            self,
            "issues",
            tuple(ConfigFieldIssue(issue.field_name, issue.kind) for issue in self.issues),
        )
        if self.issues != tuple(sorted(self.issues, key=lambda issue: issue.sort_key)):
            raise ValueError("issues must use canonical order")
        if len(self.issues) != len(frozenset(issue.sort_key for issue in self.issues)):
            raise ValueError("issues must be unique")

        explicit = frozenset(self.explicit_fields)
        issue_pairs = frozenset(issue.sort_key for issue in self.issues)
        if any(issue.field_name not in explicit for issue in self.issues):
            raise ValueError("every issue must refer to an explicit field")

        if self.method_status is AdapterMethodStatus.SUPPORTED_LORA:
            if self.declared_peft_type != "LORA":
                raise ValueError("supported LoRA status requires declared peft_type LORA")
            if "peft_type" not in explicit:
                raise ValueError("supported LoRA status requires an explicit peft_type field")

            unknown_fields = explicit - _KNOWN_FIELDS
            unknown_issue_fields = frozenset(
                issue.field_name
                for issue in self.issues
                if issue.kind is ConfigFieldIssueKind.UNKNOWN_FIELD
            )
            if unknown_issue_fields != unknown_fields:
                raise ValueError("unknown-field issues must exactly cover explicit unknown fields")
            if any(
                issue.kind is not ConfigFieldIssueKind.UNKNOWN_FIELD
                and issue.field_name in unknown_fields
                for issue in self.issues
            ):
                raise ValueError("unknown fields may only use unknown-field issues")

            runtime_issue = (
                "runtime_config",
                ConfigFieldIssueKind.IGNORED_FIELD.value,
            )
            if ("runtime_config" in explicit) != (runtime_issue in issue_pairs):
                raise ValueError("runtime_config presence requires exactly one ignored-field issue")
            if any(
                issue.kind is ConfigFieldIssueKind.IGNORED_FIELD
                and issue.field_name != "runtime_config"
                for issue in self.issues
            ):
                raise ValueError("only runtime_config may use an ignored-field issue")

            if self.lora is None and not any(issue.blocking for issue in self.issues):
                raise ValueError("missing LoRA profile requires a blocking field issue")
            if self.lora is not None:
                _validate_profile_field_provenance(self.lora, explicit)
                if self.lora.peft_version_was_declared and "peft_version" not in explicit:
                    raise ValueError(
                        "PEFT version declaration flag is inconsistent with explicit fields"
                    )
                if (
                    not self.lora.peft_version_was_declared
                    and self.lora.peft_version != PINNED_PEFT_VERSION
                ):
                    raise ValueError("an undeclared PEFT version must use the pinned default")
                required_profile_issues: set[tuple[str, str]] = set()
                if self.lora.bias is not LoraBiasMode.NONE:
                    required_profile_issues.add(
                        ("bias", ConfigFieldIssueKind.UNSUPPORTED_VALUE.value)
                    )
                if self.lora.initializer.kind is not LoraInitializerKind.DEFAULT:
                    required_profile_issues.add(
                        (
                            "init_lora_weights",
                            ConfigFieldIssueKind.UNSUPPORTED_VALUE.value,
                        )
                    )
                if not required_profile_issues.issubset(issue_pairs):
                    raise ValueError("nonordinary profile values require blocking field issues")
        elif self.method_status in {AdapterMethodStatus.MISSING, AdapterMethodStatus.INVALID}:
            if (self.method_status is AdapterMethodStatus.MISSING) == ("peft_type" in explicit):
                raise ValueError("method status is inconsistent with explicit peft_type presence")
            if self.declared_peft_type is not None:
                raise ValueError("missing or invalid method status must not retain peft_type")
            if self.lora is not None or self.issues:
                raise ValueError(
                    "unidentified methods must not carry a LoRA profile or field issues"
                )
        else:
            _require_nonempty_text("declared_peft_type", self.declared_peft_type)
            if "peft_type" not in explicit:
                raise ValueError("unsupported method status requires an explicit peft_type field")
            if self.declared_peft_type == "LORA":
                raise ValueError("unsupported method status cannot declare LORA")
            if self.lora is not None or self.issues:
                raise ValueError(
                    "unsupported methods must not carry a LoRA profile or field issues"
                )

    @property
    def closed_profile(self) -> bool:
        """Return whether every recognized LoRA field is ordinary and understood."""

        return (
            self.method_status is AdapterMethodStatus.SUPPORTED_LORA
            and self.lora is not None
            and self.lora.bias is LoraBiasMode.NONE
            and self.lora.initializer.kind is LoraInitializerKind.DEFAULT
            and not any(issue.blocking for issue in self.issues)
        )

    def __setstate__(self, _state: object) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is immutable")


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class _Decoded(Generic[T]):
    value: T
    valid: bool = True


def parse_adapter_config(
    document: bytes,
    *,
    limits: AdapterConfigLimits = DEFAULT_ADAPTER_CONFIG_LIMITS,
) -> AdapterConfigManifest:
    """Parse one complete ``adapter_config.json`` document without side effects."""

    if type(document) is not bytes:
        raise TypeError("adapter config document must be bytes")
    if type(limits) is not AdapterConfigLimits:
        raise TypeError("limits must be AdapterConfigLimits")
    limits = _validated_limits_copy(limits)
    if len(document) > limits.max_document_bytes:
        raise AdapterConfigLimitExceeded(
            AdapterConfigErrorCode.DOCUMENT_EXCEEDS_POLICY_LIMIT,
            limit=limits.max_document_bytes,
        )

    text: str | None = None
    try:
        text = document.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        pass
    if text is None:
        raise InvalidAdapterConfig(AdapterConfigErrorCode.DOCUMENT_UTF8)

    root: object | None = None
    json_error: BoundedJsonError | None = None
    try:
        root = decode_json(
            text,
            JsonLimits(
                max_document_chars=limits.max_document_bytes,
                max_depth=limits.max_json_depth,
                max_string_chars=limits.max_json_string_chars,
                max_tokens=limits.max_json_tokens,
                max_number_chars=limits.max_json_number_chars,
            ),
        )
    except BoundedJsonError as error:
        json_error = error
    if json_error is not None:
        _raise_adapter_json_error(json_error)
    if type(root) is not dict:
        raise InvalidAdapterConfig(AdapterConfigErrorCode.ROOT_NOT_OBJECT)
    members = cast(dict[str, object], root)
    if len(members) > limits.max_root_fields:
        raise AdapterConfigLimitExceeded(
            AdapterConfigErrorCode.ROOT_FIELD_COUNT_EXCEEDS_POLICY_LIMIT,
            limit=limits.max_root_fields,
        )
    _validate_json_tree(members, limits)

    explicit_fields = tuple(sorted(members))
    method_status, declared_peft_type = _classify_method(members, limits)
    if method_status is not AdapterMethodStatus.SUPPORTED_LORA:
        return AdapterConfigManifest(
            schema=ADAPTER_CONFIG_SCHEMA,
            method_status=method_status,
            declared_peft_type=declared_peft_type,
            explicit_fields=explicit_fields,
            lora=None,
            issues=(),
        )

    issues: list[ConfigFieldIssue] = [
        ConfigFieldIssue(field_name=name, kind=ConfigFieldIssueKind.UNKNOWN_FIELD)
        for name in members
        if name not in _KNOWN_FIELDS
    ]
    lora = _decode_lora_profile(members, limits, issues)
    ordered_issues = tuple(sorted(issues, key=lambda issue: issue.sort_key))
    return AdapterConfigManifest(
        schema=ADAPTER_CONFIG_SCHEMA,
        method_status=method_status,
        declared_peft_type=declared_peft_type,
        explicit_fields=explicit_fields,
        lora=lora,
        issues=ordered_issues,
    )


def _classify_method(
    members: dict[str, object],
    limits: AdapterConfigLimits,
) -> tuple[AdapterMethodStatus, str | None]:
    raw = members.get("peft_type", _MISSING)
    if raw is _MISSING:
        return AdapterMethodStatus.MISSING, None
    if type(raw) is not str or not raw:
        return AdapterMethodStatus.INVALID, None
    _check_retained_text(raw, limits)
    if raw == "LORA":
        return AdapterMethodStatus.SUPPORTED_LORA, raw
    return AdapterMethodStatus.UNSUPPORTED, raw


def _decode_lora_profile(
    members: dict[str, object],
    limits: AdapterConfigLimits,
    issues: list[ConfigFieldIssue],
) -> LoraConfigProfile | None:
    task_type = _decode_task_type(members, issues)
    auto_mapping = _decode_auto_mapping(members, limits, issues)
    peft_version, peft_version_was_declared = _decode_peft_version(members, limits, issues)
    base_model = _decode_optional_text(members, "base_model_name_or_path", None, limits, issues)
    revision = _decode_optional_text(members, "revision", None, limits, issues)
    inference_mode = _decode_bool(members, "inference_mode", False, issues)
    rank = _decode_integer(members, "r", 8, minimum=1, issues=issues)
    target_modules = _decode_selector(
        members, "target_modules", allow_all_linear=True, limits=limits, issues=issues
    )
    exclude_modules = _decode_selector(
        members, "exclude_modules", allow_all_linear=False, limits=limits, issues=issues
    )
    lora_alpha = _decode_decimal(
        members,
        "lora_alpha",
        Decimal(8),
        minimum=-_MAX_DECIMAL_MAGNITUDE,
        maximum=_MAX_DECIMAL_MAGNITUDE,
        issues=issues,
    )
    lora_dropout = _decode_decimal(
        members,
        "lora_dropout",
        Decimal(0),
        minimum=Decimal(0),
        maximum=Decimal(1),
        issues=issues,
    )
    fan_in_fan_out = _decode_bool(members, "fan_in_fan_out", False, issues)
    bias = _decode_bias(members, issues)
    use_rslora = _decode_bool(members, "use_rslora", False, issues)
    modules_to_save = _decode_optional_text_list(
        members, "modules_to_save", limits, issues, normalize=False
    )
    initializer = _decode_initializer(members, issues)
    layers_to_transform = _decode_layers_to_transform(members, issues)
    layers_pattern = _decode_optional_text_or_list(members, "layers_pattern", limits, issues)
    rank_pattern = _decode_rank_pattern(members, limits, issues)
    alpha_pattern = _decode_alpha_pattern(members, limits, issues)

    _classify_default_only_fields(members, issues)

    decoded = (
        task_type,
        auto_mapping,
        peft_version,
        base_model,
        revision,
        inference_mode,
        rank,
        target_modules,
        exclude_modules,
        lora_alpha,
        lora_dropout,
        fan_in_fan_out,
        bias,
        use_rslora,
        modules_to_save,
        initializer,
        layers_to_transform,
        layers_pattern,
        rank_pattern,
        alpha_pattern,
    )
    if not all(item.valid for item in decoded):
        return None
    if target_modules.value.kind in _STRING_SELECTOR_KINDS and (
        layers_to_transform.value is not None or layers_pattern.value is not None
    ):
        if layers_to_transform.value is not None:
            _add_issue(issues, "layers_to_transform", ConfigFieldIssueKind.INVALID_VALUE)
        if layers_pattern.value is not None:
            _add_issue(issues, "layers_pattern", ConfigFieldIssueKind.INVALID_VALUE)
        return None
    if target_modules.value.kind is ModuleSelectorKind.NAMES and not target_modules.value.values:
        target_parameters = members.get("target_parameters", _MISSING)
        if target_parameters is _MISSING or target_parameters is None:
            _add_issue(issues, "target_modules", ConfigFieldIssueKind.INVALID_VALUE)
        return None
    raw_layers_to_transform = members.get("layers_to_transform", _MISSING)
    scalar_zero_layer = type(
        raw_layers_to_transform
    ) is IntegerLexeme and raw_layers_to_transform.value in {"0", "-0"}
    if layers_pattern.value and (not layers_to_transform.value or scalar_zero_layer):
        _add_issue(issues, "layers_pattern", ConfigFieldIssueKind.INVALID_VALUE)
        return None

    return LoraConfigProfile(
        task_type=task_type.value,
        auto_mapping=auto_mapping.value,
        peft_version=peft_version.value,
        peft_version_was_declared=peft_version_was_declared,
        base_model_name_or_path=base_model.value,
        revision=revision.value,
        inference_mode=inference_mode.value,
        r=rank.value,
        target_modules=target_modules.value,
        exclude_modules=exclude_modules.value,
        lora_alpha=lora_alpha.value,
        lora_dropout=lora_dropout.value,
        fan_in_fan_out=fan_in_fan_out.value,
        bias=bias.value,
        use_rslora=use_rslora.value,
        modules_to_save=modules_to_save.value,
        initializer=initializer.value,
        layers_to_transform=layers_to_transform.value,
        layers_pattern=layers_pattern.value,
        rank_pattern=rank_pattern.value,
        alpha_pattern=alpha_pattern.value,
    )


def _decode_task_type(
    members: dict[str, object], issues: list[ConfigFieldIssue]
) -> _Decoded[PeftTaskType | None]:
    raw = members.get("task_type", _MISSING)
    if raw is _MISSING or raw is None:
        return _Decoded(None)
    if type(raw) is not str:
        _add_issue(issues, "task_type", ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded(None, False)
    try:
        return _Decoded(PeftTaskType(raw))
    except ValueError:
        _add_issue(issues, "task_type", ConfigFieldIssueKind.INVALID_VALUE)
        return _Decoded(None, False)


def _decode_auto_mapping(
    members: dict[str, object],
    limits: AdapterConfigLimits,
    issues: list[ConfigFieldIssue],
) -> _Decoded[AutoMapping | None]:
    raw = members.get("auto_mapping", _MISSING)
    if raw is _MISSING or raw is None:
        return _Decoded(None)
    if type(raw) is not dict:
        _add_issue(issues, "auto_mapping", ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded(None, False)
    mapping = cast(dict[str, object], raw)
    if set(mapping) != {"base_model_class", "parent_library"}:
        _add_issue(issues, "auto_mapping", ConfigFieldIssueKind.INVALID_VALUE)
        return _Decoded(None, False)
    base_model_class = mapping["base_model_class"]
    parent_library = mapping["parent_library"]
    if type(base_model_class) is not str or type(parent_library) is not str:
        _add_issue(issues, "auto_mapping", ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded(None, False)
    _check_retained_text(base_model_class, limits)
    _check_retained_text(parent_library, limits)
    if not base_model_class or not parent_library:
        _add_issue(issues, "auto_mapping", ConfigFieldIssueKind.INVALID_VALUE)
        return _Decoded(None, False)
    return _Decoded(AutoMapping(base_model_class, parent_library))


def _decode_peft_version(
    members: dict[str, object],
    limits: AdapterConfigLimits,
    issues: list[ConfigFieldIssue],
) -> tuple[_Decoded[str], bool]:
    raw = members.get("peft_version", _MISSING)
    if raw is _MISSING or raw is None:
        return _Decoded(PINNED_PEFT_VERSION), False
    if type(raw) is not str:
        _add_issue(issues, "peft_version", ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded(PINNED_PEFT_VERSION, False), False
    _check_retained_text(raw, limits)
    if not raw:
        _add_issue(issues, "peft_version", ConfigFieldIssueKind.INVALID_VALUE)
        return _Decoded(PINNED_PEFT_VERSION, False), False
    return _Decoded(raw), True


def _decode_optional_text(
    members: dict[str, object],
    name: str,
    default: str | None,
    limits: AdapterConfigLimits,
    issues: list[ConfigFieldIssue],
) -> _Decoded[str | None]:
    raw = members.get(name, _MISSING)
    if raw is _MISSING:
        return _Decoded(default)
    if raw is None:
        return _Decoded(None)
    if type(raw) is not str:
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded(default, False)
    _check_retained_text(raw, limits)
    return _Decoded(raw)


def _decode_bool(
    members: dict[str, object],
    name: str,
    default: bool,
    issues: list[ConfigFieldIssue],
) -> _Decoded[bool]:
    raw = members.get(name, _MISSING)
    if raw is _MISSING:
        return _Decoded(default)
    if type(raw) is not bool:
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded(default, False)
    return _Decoded(raw)


def _decode_integer(
    members: dict[str, object],
    name: str,
    default: int,
    *,
    minimum: int,
    issues: list[ConfigFieldIssue],
) -> _Decoded[int]:
    raw = members.get(name, _MISSING)
    if raw is _MISSING:
        return _Decoded(default)
    if type(raw) is not IntegerLexeme:
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded(default, False)
    value = _supported_integer_value(raw.value)
    if value is None or not minimum <= value:
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_VALUE)
        return _Decoded(default, False)
    return _Decoded(value)


def _decode_decimal(
    members: dict[str, object],
    name: str,
    default: Decimal,
    *,
    minimum: Decimal,
    maximum: Decimal,
    issues: list[ConfigFieldIssue],
) -> _Decoded[Decimal]:
    raw = members.get(name, _MISSING)
    if raw is _MISSING:
        return _Decoded(default)
    if type(raw) not in {IntegerLexeme, FloatLexeme}:
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded(default, False)
    lexeme = cast(IntegerLexeme | FloatLexeme, raw)
    try:
        value = Decimal(lexeme.value)
    except DecimalException:
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_VALUE)
        return _Decoded(default, False)
    if not _decimal_is_supported(value) or not minimum <= value <= maximum:
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_VALUE)
        return _Decoded(default, False)
    return _Decoded(value)


def _decode_selector(
    members: dict[str, object],
    name: str,
    *,
    allow_all_linear: bool,
    limits: AdapterConfigLimits,
    issues: list[ConfigFieldIssue],
) -> _Decoded[ModuleSelector]:
    raw = members.get(name, _MISSING)
    default = ModuleSelector(ModuleSelectorKind.DEFAULT)
    if raw is _MISSING or raw is None:
        return _Decoded(default)
    if type(raw) is str:
        _check_retained_text(raw, limits)
        if not raw:
            _add_issue(issues, name, ConfigFieldIssueKind.INVALID_VALUE)
            return _Decoded(default, False)
        if allow_all_linear and raw.lower() == "all-linear":
            return _Decoded(ModuleSelector(ModuleSelectorKind.ALL_LINEAR))
        return _Decoded(ModuleSelector(ModuleSelectorKind.REGEX, (raw,)))
    if type(raw) is not list:
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded(default, False)
    values = cast(list[object], raw)
    if not all(type(value) is str for value in values):
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded(default, False)
    names = cast(list[str], values)
    for value in names:
        _check_retained_text(value, limits)
    if any(not value for value in names):
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_VALUE)
        return _Decoded(default, False)
    return _Decoded(ModuleSelector(ModuleSelectorKind.NAMES, tuple(sorted(frozenset(names)))))


def _decode_bias(
    members: dict[str, object], issues: list[ConfigFieldIssue]
) -> _Decoded[LoraBiasMode]:
    raw = members.get("bias", _MISSING)
    if raw is _MISSING:
        return _Decoded(LoraBiasMode.NONE)
    if type(raw) is not str:
        _add_issue(issues, "bias", ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded(LoraBiasMode.NONE, False)
    try:
        value = LoraBiasMode(raw)
    except ValueError:
        _add_issue(issues, "bias", ConfigFieldIssueKind.INVALID_VALUE)
        return _Decoded(LoraBiasMode.NONE, False)
    if value is not LoraBiasMode.NONE:
        _add_issue(issues, "bias", ConfigFieldIssueKind.UNSUPPORTED_VALUE)
    return _Decoded(value)


def _decode_optional_text_list(
    members: dict[str, object],
    name: str,
    limits: AdapterConfigLimits,
    issues: list[ConfigFieldIssue],
    *,
    normalize: bool,
) -> _Decoded[tuple[str, ...] | None]:
    raw = members.get(name, _MISSING)
    if raw is _MISSING or raw is None:
        return _Decoded(None)
    if type(raw) is not list:
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded(None, False)
    values = cast(list[object], raw)
    if not all(type(value) is str for value in values):
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded(None, False)
    strings = cast(list[str], values)
    for value in strings:
        _check_retained_text(value, limits)
    if any(not value for value in strings):
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_VALUE)
        return _Decoded(None, False)
    result = tuple(sorted(frozenset(strings))) if normalize else tuple(strings)
    return _Decoded(result)


def _decode_initializer(
    members: dict[str, object], issues: list[ConfigFieldIssue]
) -> _Decoded[LoraInitializer]:
    raw = members.get("init_lora_weights", _MISSING)
    default = LoraInitializer(LoraInitializerKind.DEFAULT)
    if raw is _MISSING or raw is True:
        return _Decoded(default)
    if raw is False:
        random_initializer = LoraInitializer(LoraInitializerKind.RANDOM)
        _add_issue(issues, "init_lora_weights", ConfigFieldIssueKind.UNSUPPORTED_VALUE)
        return _Decoded(random_initializer)
    if type(raw) is not str:
        _add_issue(issues, "init_lora_weights", ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded(default, False)
    fixed = {
        "eva": LoraInitializerKind.EVA,
        "pissa": LoraInitializerKind.PISSA,
        "corda": LoraInitializerKind.CORDA,
        "loftq": LoraInitializerKind.LOFTQ,
        "orthogonal": LoraInitializerKind.ORTHOGONAL,
        "lora_ga": LoraInitializerKind.LORA_GA,
    }
    lowered = raw.lower()
    kind = (
        LoraInitializerKind.GAUSSIAN
        if lowered == "gaussian"
        else LoraInitializerKind.OLORA
        if lowered == "olora"
        else fixed.get(raw)
    )
    initializer: LoraInitializer | None = None
    if kind is not None:
        initializer = LoraInitializer(kind)
    elif raw.startswith("pissa_niter_"):
        suffix = raw.removeprefix("pissa_niter_")
        if (
            suffix
            and len(suffix) <= len(str(_MAX_INTEGER))
            and all("0" <= character <= "9" for character in suffix)
        ):
            iterations = int(suffix)
            if iterations <= _MAX_INTEGER:
                initializer = LoraInitializer(LoraInitializerKind.PISSA_NITER, iterations)
    if initializer is None:
        _add_issue(issues, "init_lora_weights", ConfigFieldIssueKind.INVALID_VALUE)
        return _Decoded(default, False)
    _add_issue(issues, "init_lora_weights", ConfigFieldIssueKind.UNSUPPORTED_VALUE)
    return _Decoded(initializer)


def _decode_layers_to_transform(
    members: dict[str, object], issues: list[ConfigFieldIssue]
) -> _Decoded[tuple[int, ...] | None]:
    raw = members.get("layers_to_transform", _MISSING)
    if raw is _MISSING or raw is None:
        return _Decoded(None)
    if type(raw) is IntegerLexeme:
        value = _supported_integer_value(raw.value)
        if value is not None and value >= 0:
            return _Decoded((value,))
        _add_issue(issues, "layers_to_transform", ConfigFieldIssueKind.INVALID_VALUE)
        return _Decoded(None, False)
    if type(raw) is not list:
        _add_issue(issues, "layers_to_transform", ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded(None, False)
    values = cast(list[object], raw)
    if not all(type(value) is IntegerLexeme for value in values):
        _add_issue(issues, "layers_to_transform", ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded(None, False)
    decoded_indices = tuple(
        _supported_integer_value(cast(IntegerLexeme, value).value) for value in values
    )
    if any(value is None or value < 0 for value in decoded_indices):
        _add_issue(issues, "layers_to_transform", ConfigFieldIssueKind.INVALID_VALUE)
        return _Decoded(None, False)
    indices = cast(tuple[int, ...], decoded_indices)
    return _Decoded(tuple(sorted(frozenset(indices))))


def _decode_optional_text_or_list(
    members: dict[str, object],
    name: str,
    limits: AdapterConfigLimits,
    issues: list[ConfigFieldIssue],
) -> _Decoded[tuple[str, ...] | None]:
    raw = members.get(name, _MISSING)
    if raw is _MISSING or raw is None:
        return _Decoded(None)
    if type(raw) is str:
        _check_retained_text(raw, limits)
        return _Decoded((raw,) if raw else ())
    if type(raw) is not list:
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded(None, False)
    values = cast(list[object], raw)
    if not all(type(value) is str for value in values):
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded(None, False)
    strings = cast(list[str], values)
    for value in strings:
        _check_retained_text(value, limits)
    if any(not value for value in strings):
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_VALUE)
        return _Decoded(None, False)
    return _Decoded(tuple(strings))


def _decode_rank_pattern(
    members: dict[str, object],
    limits: AdapterConfigLimits,
    issues: list[ConfigFieldIssue],
) -> _Decoded[tuple[RankPatternEntry, ...]]:
    raw = members.get("rank_pattern", _MISSING)
    if raw is _MISSING:
        return _Decoded(())
    if type(raw) is not dict:
        _add_issue(issues, "rank_pattern", ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded((), False)
    result: list[RankPatternEntry] = []
    for pattern, raw_rank in cast(dict[str, object], raw).items():
        _check_retained_text(pattern, limits)
        if not pattern:
            _add_issue(issues, "rank_pattern", ConfigFieldIssueKind.INVALID_VALUE)
            return _Decoded((), False)
        if type(raw_rank) is not IntegerLexeme:
            _add_issue(issues, "rank_pattern", ConfigFieldIssueKind.INVALID_TYPE)
            return _Decoded((), False)
        rank = _supported_integer_value(raw_rank.value)
        if rank is None or rank < 1:
            _add_issue(issues, "rank_pattern", ConfigFieldIssueKind.INVALID_VALUE)
            return _Decoded((), False)
        result.append(RankPatternEntry(pattern, rank))
    return _Decoded(tuple(result))


def _decode_alpha_pattern(
    members: dict[str, object],
    limits: AdapterConfigLimits,
    issues: list[ConfigFieldIssue],
) -> _Decoded[tuple[AlphaPatternEntry, ...]]:
    raw = members.get("alpha_pattern", _MISSING)
    if raw is _MISSING:
        return _Decoded(())
    if type(raw) is not dict:
        _add_issue(issues, "alpha_pattern", ConfigFieldIssueKind.INVALID_TYPE)
        return _Decoded((), False)
    result: list[AlphaPatternEntry] = []
    for pattern, raw_alpha in cast(dict[str, object], raw).items():
        _check_retained_text(pattern, limits)
        if not pattern:
            _add_issue(issues, "alpha_pattern", ConfigFieldIssueKind.INVALID_VALUE)
            return _Decoded((), False)
        if type(raw_alpha) not in {IntegerLexeme, FloatLexeme}:
            _add_issue(issues, "alpha_pattern", ConfigFieldIssueKind.INVALID_TYPE)
            return _Decoded((), False)
        lexeme = cast(IntegerLexeme | FloatLexeme, raw_alpha)
        try:
            alpha = Decimal(lexeme.value)
        except DecimalException:
            _add_issue(issues, "alpha_pattern", ConfigFieldIssueKind.INVALID_VALUE)
            return _Decoded((), False)
        if not _decimal_is_supported(alpha) or not (
            -_MAX_DECIMAL_MAGNITUDE <= alpha <= _MAX_DECIMAL_MAGNITUDE
        ):
            _add_issue(issues, "alpha_pattern", ConfigFieldIssueKind.INVALID_VALUE)
            return _Decoded((), False)
        result.append(AlphaPatternEntry(pattern, alpha))
    return _Decoded(tuple(result))


def _classify_default_only_fields(
    members: dict[str, object], issues: list[ConfigFieldIssue]
) -> None:
    _classify_null_or_type(members, "megatron_config", dict, issues)
    _classify_megatron_core(members, issues)
    _classify_null_or_types(members, "trainable_token_indices", (list, dict), issues)
    _classify_empty_object(members, "loftq_config", issues)
    _classify_null_or_type(members, "eva_config", dict, issues)
    _classify_null_or_type(members, "corda_config", dict, issues)
    _classify_null_or_type(members, "lora_ga_config", dict, issues)
    _classify_default_bool(members, "use_dora", False, issues)
    _classify_null_or_type(members, "alora_invocation_tokens", list, issues)
    _classify_default_bool(members, "use_qalora", False, issues)
    _classify_default_integer(members, "qalora_group_size", 16, minimum=1, issues=issues)
    _classify_null_or_type(members, "layer_replication", list, issues)
    if "runtime_config" in members:
        _add_issue(issues, "runtime_config", ConfigFieldIssueKind.IGNORED_FIELD)
    _classify_default_bool(members, "lora_bias", False, issues)
    _classify_null_or_type(members, "target_parameters", list, issues)
    _classify_null_or_type(members, "use_bdlora", dict, issues)
    _classify_null_or_type(members, "arrow_config", dict, issues)
    _classify_default_bool(members, "ensure_weight_tying", False, issues)


def _classify_default_bool(
    members: dict[str, object],
    name: str,
    default: bool,
    issues: list[ConfigFieldIssue],
) -> None:
    raw = members.get(name, _MISSING)
    if raw is _MISSING:
        return
    if type(raw) is not bool:
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_TYPE)
    elif raw is not default:
        _add_issue(issues, name, ConfigFieldIssueKind.UNSUPPORTED_VALUE)


def _classify_default_integer(
    members: dict[str, object],
    name: str,
    default: int,
    *,
    minimum: int,
    issues: list[ConfigFieldIssue],
) -> None:
    raw = members.get(name, _MISSING)
    if raw is _MISSING:
        return
    if type(raw) is not IntegerLexeme:
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_TYPE)
        return
    value = _supported_integer_value(raw.value)
    if value is None or value < minimum:
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_VALUE)
    elif value != default:
        _add_issue(issues, name, ConfigFieldIssueKind.UNSUPPORTED_VALUE)


def _classify_null_or_type(
    members: dict[str, object],
    name: str,
    accepted_type: type[object],
    issues: list[ConfigFieldIssue],
) -> None:
    _classify_null_or_types(members, name, (accepted_type,), issues)


def _classify_null_or_types(
    members: dict[str, object],
    name: str,
    accepted_types: tuple[type[object], ...],
    issues: list[ConfigFieldIssue],
) -> None:
    raw = members.get(name, _MISSING)
    if raw is _MISSING or raw is None:
        return
    if type(raw) not in accepted_types:
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_TYPE)
    else:
        _add_issue(issues, name, ConfigFieldIssueKind.UNSUPPORTED_VALUE)


def _classify_empty_object(
    members: dict[str, object], name: str, issues: list[ConfigFieldIssue]
) -> None:
    raw = members.get(name, _MISSING)
    if raw is _MISSING:
        return
    if type(raw) is not dict:
        _add_issue(issues, name, ConfigFieldIssueKind.INVALID_TYPE)
    elif raw:
        _add_issue(issues, name, ConfigFieldIssueKind.UNSUPPORTED_VALUE)


def _classify_megatron_core(members: dict[str, object], issues: list[ConfigFieldIssue]) -> None:
    raw = members.get("megatron_core", _MISSING)
    if raw is _MISSING or raw == "megatron.core":
        return
    if type(raw) is not str and raw is not None:
        _add_issue(issues, "megatron_core", ConfigFieldIssueKind.INVALID_TYPE)
    else:
        _add_issue(issues, "megatron_core", ConfigFieldIssueKind.UNSUPPORTED_VALUE)


def _add_issue(issues: list[ConfigFieldIssue], name: str, kind: ConfigFieldIssueKind) -> None:
    candidate = ConfigFieldIssue(field_name=name, kind=kind)
    if candidate not in issues:
        issues.append(candidate)


def _validate_json_tree(root: dict[str, object], limits: AdapterConfigLimits) -> None:
    stack: list[tuple[object, bool]] = [(root, True)]
    while stack:
        value, is_root = stack.pop()
        if type(value) is str:
            if _has_lone_surrogate(value):
                raise InvalidAdapterConfig(AdapterConfigErrorCode.INVALID_UNICODE_SCALAR)
        elif type(value) is list:
            sequence = cast(list[object], value)
            if len(sequence) > limits.max_collection_items:
                raise AdapterConfigLimitExceeded(
                    AdapterConfigErrorCode.COLLECTION_COUNT_EXCEEDS_POLICY_LIMIT,
                    limit=limits.max_collection_items,
                )
            stack.extend((item, False) for item in sequence)
        elif type(value) is dict:
            mapping = cast(dict[str, object], value)
            if not is_root and len(mapping) > limits.max_collection_items:
                raise AdapterConfigLimitExceeded(
                    AdapterConfigErrorCode.COLLECTION_COUNT_EXCEEDS_POLICY_LIMIT,
                    limit=limits.max_collection_items,
                )
            for key, item in mapping.items():
                if _has_lone_surrogate(key):
                    raise InvalidAdapterConfig(AdapterConfigErrorCode.INVALID_UNICODE_SCALAR)
                _check_retained_text(key, limits)
                stack.append((item, False))


def _has_lone_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _check_retained_text(value: str, limits: AdapterConfigLimits) -> None:
    if len(value.encode("utf-8")) > limits.max_name_bytes:
        raise AdapterConfigLimitExceeded(
            AdapterConfigErrorCode.NAME_EXCEEDS_POLICY_LIMIT,
            limit=limits.max_name_bytes,
        )


def _raise_adapter_json_error(error: BoundedJsonError) -> NoReturn:
    if type(error) is InvalidJson:
        code = (
            AdapterConfigErrorCode.DUPLICATE_JSON_KEY
            if error.code is BoundedJsonErrorCode.DUPLICATE_KEY
            else AdapterConfigErrorCode.DOCUMENT_JSON
        )
        raise InvalidAdapterConfig(code)
    if type(error) is JsonLimitExceeded:
        code = {
            BoundedJsonErrorCode.DOCUMENT_LIMIT: (
                AdapterConfigErrorCode.DOCUMENT_EXCEEDS_POLICY_LIMIT
            ),
            BoundedJsonErrorCode.DEPTH_LIMIT: (
                AdapterConfigErrorCode.JSON_DEPTH_EXCEEDS_POLICY_LIMIT
            ),
            BoundedJsonErrorCode.STRING_LIMIT: (
                AdapterConfigErrorCode.JSON_STRING_EXCEEDS_POLICY_LIMIT
            ),
            BoundedJsonErrorCode.TOKEN_LIMIT: (
                AdapterConfigErrorCode.JSON_TOKEN_EXCEEDS_POLICY_LIMIT
            ),
            BoundedJsonErrorCode.NUMBER_LIMIT: (
                AdapterConfigErrorCode.JSON_NUMBER_EXCEEDS_POLICY_LIMIT
            ),
        }[error.code]
        raise AdapterConfigLimitExceeded(code, limit=error.limit)
    raise TypeError("unsupported bounded JSON error type")


def _validated_limits_copy(limits: AdapterConfigLimits) -> AdapterConfigLimits:
    return AdapterConfigLimits(
        max_document_bytes=limits.max_document_bytes,
        max_json_depth=limits.max_json_depth,
        max_json_string_chars=limits.max_json_string_chars,
        max_json_tokens=limits.max_json_tokens,
        max_json_number_chars=limits.max_json_number_chars,
        max_root_fields=limits.max_root_fields,
        max_collection_items=limits.max_collection_items,
        max_name_bytes=limits.max_name_bytes,
    )


def _validated_lora_profile_copy(profile: LoraConfigProfile) -> LoraConfigProfile:
    return LoraConfigProfile(
        task_type=profile.task_type,
        auto_mapping=profile.auto_mapping,
        peft_version=profile.peft_version,
        peft_version_was_declared=profile.peft_version_was_declared,
        base_model_name_or_path=profile.base_model_name_or_path,
        revision=profile.revision,
        inference_mode=profile.inference_mode,
        r=profile.r,
        target_modules=profile.target_modules,
        exclude_modules=profile.exclude_modules,
        lora_alpha=profile.lora_alpha,
        lora_dropout=profile.lora_dropout,
        fan_in_fan_out=profile.fan_in_fan_out,
        bias=profile.bias,
        use_rslora=profile.use_rslora,
        modules_to_save=profile.modules_to_save,
        initializer=profile.initializer,
        layers_to_transform=profile.layers_to_transform,
        layers_pattern=profile.layers_pattern,
        rank_pattern=profile.rank_pattern,
        alpha_pattern=profile.alpha_pattern,
    )


def _validate_profile_field_provenance(
    profile: LoraConfigProfile,
    explicit_fields: frozenset[str],
) -> None:
    defaults: tuple[tuple[str, object, object], ...] = (
        ("task_type", profile.task_type, None),
        ("auto_mapping", profile.auto_mapping, None),
        ("peft_version", profile.peft_version, PINNED_PEFT_VERSION),
        ("base_model_name_or_path", profile.base_model_name_or_path, None),
        ("revision", profile.revision, None),
        ("inference_mode", profile.inference_mode, False),
        ("r", profile.r, 8),
        (
            "target_modules",
            profile.target_modules,
            ModuleSelector(ModuleSelectorKind.DEFAULT),
        ),
        (
            "exclude_modules",
            profile.exclude_modules,
            ModuleSelector(ModuleSelectorKind.DEFAULT),
        ),
        ("lora_alpha", profile.lora_alpha, Decimal(8)),
        ("lora_dropout", profile.lora_dropout, Decimal(0)),
        ("fan_in_fan_out", profile.fan_in_fan_out, False),
        ("bias", profile.bias, LoraBiasMode.NONE),
        ("use_rslora", profile.use_rslora, False),
        ("modules_to_save", profile.modules_to_save, None),
        (
            "init_lora_weights",
            profile.initializer,
            LoraInitializer(LoraInitializerKind.DEFAULT),
        ),
        ("layers_to_transform", profile.layers_to_transform, None),
        ("layers_pattern", profile.layers_pattern, None),
        ("rank_pattern", profile.rank_pattern, ()),
        ("alpha_pattern", profile.alpha_pattern, ()),
    )
    if any(
        field_name not in explicit_fields and value != default
        for field_name, value, default in defaults
    ):
        raise ValueError("nondefault profile values require explicit source fields")


def _require_exact_enum(name: str, value: object, enum_type: type[StrEnum]) -> None:
    if type(value) is not enum_type:
        raise TypeError(f"{name} must be {enum_type.__name__}")


def _require_text(name: str, value: object) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if _has_lone_surrogate(value):
        raise ValueError(f"{name} must contain only Unicode scalar values")


def _require_nonempty_text(name: str, value: object) -> None:
    _require_text(name, value)
    if not value:
        raise ValueError(f"{name} must not be empty")


def _require_optional_text(name: str, value: object) -> None:
    if value is not None and type(value) is not str:
        raise TypeError(f"{name} must be a string or None")
    if value is not None:
        _require_text(name, value)


def _require_bool(name: str, value: object) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")


def _require_integer(name: str, value: object, *, minimum: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= _MAX_INTEGER:
        raise ValueError(f"{name} is outside the supported integer range")


def _require_decimal(
    name: str,
    value: object,
    *,
    minimum: Decimal = -_MAX_DECIMAL_MAGNITUDE,
    maximum: Decimal = _MAX_DECIMAL_MAGNITUDE,
) -> None:
    if type(value) is not Decimal:
        raise TypeError(f"{name} must be Decimal")
    if not _decimal_is_supported(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside the supported numeric range")


def _require_text_tuple(name: str, value: object, *, allow_empty_items: bool) -> None:
    if type(value) is not tuple or not all(type(item) is str for item in value):
        raise TypeError(f"{name} must be a tuple of strings")
    for item in value:
        _require_text(name, item)
    if not allow_empty_items and any(not item for item in value):
        raise ValueError(f"{name} must not contain empty strings")


def _require_integer_tuple(name: str, value: object, *, minimum: int) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple of integers")
    for item in value:
        _require_integer(name, item, minimum=minimum)


def _require_exact_tuple(name: str, value: object, item_type: type[object]) -> None:
    if type(value) is not tuple or not all(type(item) is item_type for item in value):
        raise TypeError(f"{name} must be a tuple of {item_type.__name__} values")


def _decimal_is_supported(value: Decimal) -> bool:
    if not value.is_finite():
        return False
    exponent = value.as_tuple().exponent
    return type(exponent) is int and abs(exponent) <= _MAX_DECIMAL_EXPONENT


def _supported_integer_value(lexeme: str) -> int | None:
    digits = lexeme[1:] if lexeme.startswith("-") else lexeme
    if len(digits) > len(str(_MAX_INTEGER)):
        return None
    value = int(lexeme)
    return value if abs(value) <= _MAX_INTEGER else None


__all__ = [
    "ADAPTER_CONFIG_SCHEMA",
    "DEFAULT_ADAPTER_CONFIG_LIMITS",
    "PINNED_PEFT_VERSION",
    "AdapterConfigErrorCode",
    "AdapterConfigInspectionError",
    "AdapterConfigLimitExceeded",
    "AdapterConfigLimits",
    "AdapterConfigManifest",
    "AdapterMethodStatus",
    "AlphaPatternEntry",
    "AutoMapping",
    "ConfigFieldIssue",
    "ConfigFieldIssueKind",
    "InvalidAdapterConfig",
    "LoraBiasMode",
    "LoraConfigProfile",
    "LoraInitializer",
    "LoraInitializerKind",
    "ModuleSelector",
    "ModuleSelectorKind",
    "PeftTaskType",
    "RankPatternEntry",
    "parse_adapter_config",
]
