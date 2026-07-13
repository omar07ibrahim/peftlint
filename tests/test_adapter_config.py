from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from peftlint._bounded_json import BoundedJsonError, BoundedJsonErrorCode
from peftlint.adapter_config import (
    ADAPTER_CONFIG_SCHEMA,
    DEFAULT_ADAPTER_CONFIG_LIMITS,
    PINNED_PEFT_VERSION,
    AdapterConfigErrorCode,
    AdapterConfigInspectionError,
    AdapterConfigLimitExceeded,
    AdapterConfigLimits,
    AdapterConfigManifest,
    AdapterMethodStatus,
    AlphaPatternEntry,
    AutoMapping,
    ConfigFieldIssue,
    ConfigFieldIssueKind,
    InvalidAdapterConfig,
    LoraBiasMode,
    LoraConfigProfile,
    LoraInitializer,
    LoraInitializerKind,
    ModuleSelector,
    ModuleSelectorKind,
    PeftTaskType,
    RankPatternEntry,
    _add_issue,
    _raise_adapter_json_error,
    parse_adapter_config,
)
from peftlint.evidence import LORA_V1_RULESET


def encode_config(config: object) -> bytes:
    return json.dumps(config, ensure_ascii=True, separators=(",", ":")).encode()


def parse_config(
    config: object,
    *,
    limits: AdapterConfigLimits = DEFAULT_ADAPTER_CONFIG_LIMITS,
) -> AdapterConfigManifest:
    return parse_adapter_config(encode_config(config), limits=limits)


def require_lora(manifest: AdapterConfigManifest) -> LoraConfigProfile:
    assert manifest.lora is not None
    return manifest.lora


def issue_pairs(manifest: AdapterConfigManifest) -> tuple[tuple[str, ConfigFieldIssueKind], ...]:
    return tuple((issue.field_name, issue.kind) for issue in manifest.issues)


FULL_DEFAULT_CONFIG: dict[str, object] = {
    "task_type": None,
    "peft_type": "LORA",
    "auto_mapping": None,
    "peft_version": "0.19.1",
    "base_model_name_or_path": None,
    "revision": None,
    "inference_mode": False,
    "r": 8,
    "target_modules": None,
    "exclude_modules": None,
    "lora_alpha": 8,
    "lora_dropout": 0.0,
    "fan_in_fan_out": False,
    "bias": "none",
    "use_rslora": False,
    "modules_to_save": None,
    "init_lora_weights": True,
    "layers_to_transform": None,
    "layers_pattern": None,
    "rank_pattern": {},
    "alpha_pattern": {},
    "megatron_config": None,
    "megatron_core": "megatron.core",
    "trainable_token_indices": None,
    "loftq_config": {},
    "eva_config": None,
    "corda_config": None,
    "lora_ga_config": None,
    "use_dora": False,
    "alora_invocation_tokens": None,
    "use_qalora": False,
    "qalora_group_size": 16,
    "layer_replication": None,
    "runtime_config": {"ephemeral_gpu_offload": False},
    "lora_bias": False,
    "target_parameters": None,
    "use_bdlora": None,
    "arrow_config": None,
    "ensure_weight_tying": False,
}


def test_minimal_lora_config_expands_pinned_defaults() -> None:
    manifest = parse_config({"peft_type": "LORA"})
    profile = require_lora(manifest)

    assert manifest.schema == ADAPTER_CONFIG_SCHEMA
    assert ADAPTER_CONFIG_SCHEMA == LORA_V1_RULESET
    assert manifest.method_status is AdapterMethodStatus.SUPPORTED_LORA
    assert manifest.declared_peft_type == "LORA"
    assert manifest.explicit_fields == ("peft_type",)
    assert manifest.issues == ()
    assert manifest.closed_profile
    assert profile.task_type is None
    assert profile.auto_mapping is None
    assert profile.peft_version == PINNED_PEFT_VERSION
    assert not profile.peft_version_was_declared
    assert profile.base_model_name_or_path is None
    assert profile.revision is None
    assert not profile.inference_mode
    assert profile.r == 8
    assert profile.target_modules == ModuleSelector(ModuleSelectorKind.DEFAULT)
    assert profile.exclude_modules == ModuleSelector(ModuleSelectorKind.DEFAULT)
    assert profile.lora_alpha == Decimal(8)
    assert profile.lora_dropout == Decimal(0)
    assert not profile.fan_in_fan_out
    assert profile.bias is LoraBiasMode.NONE
    assert not profile.use_rslora
    assert profile.modules_to_save is None
    assert profile.initializer == LoraInitializer(LoraInitializerKind.DEFAULT)
    assert profile.layers_to_transform is None
    assert profile.layers_pattern is None
    assert profile.rank_pattern == ()
    assert profile.alpha_pattern == ()


def test_all_39_recognized_fields_close_without_unknowns() -> None:
    assert len(FULL_DEFAULT_CONFIG) == 39

    manifest = parse_config(FULL_DEFAULT_CONFIG)

    assert len(manifest.explicit_fields) == 39
    assert issue_pairs(manifest) == (("runtime_config", ConfigFieldIssueKind.IGNORED_FIELD),)
    assert manifest.closed_profile
    assert require_lora(manifest).peft_version_was_declared


def test_sparse_official_peft_fixture_uses_defaults() -> None:
    fixture = {
        "base_model_name_or_path": None,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": False,
        "init_lora_weights": True,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 8,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
    }

    manifest = parse_config(fixture)
    profile = require_lora(manifest)

    assert manifest.closed_profile
    assert profile.task_type is PeftTaskType.CAUSAL_LM
    assert profile.lora_alpha == Decimal(16)
    assert profile.lora_dropout == Decimal("0.1")
    assert profile.target_modules == ModuleSelector(ModuleSelectorKind.NAMES, ("q_proj", "v_proj"))
    assert profile.peft_version == PINNED_PEFT_VERSION
    assert not profile.peft_version_was_declared


@pytest.mark.parametrize("task_type", list(PeftTaskType))
def test_every_pinned_task_type_is_accepted(task_type: PeftTaskType) -> None:
    profile = require_lora(parse_config({"peft_type": "LORA", "task_type": task_type.value}))

    assert profile.task_type is task_type


@pytest.mark.parametrize(
    ("document", "status", "declared"),
    [
        ({}, AdapterMethodStatus.MISSING, None),
        ({"peft_type": None}, AdapterMethodStatus.INVALID, None),
        ({"peft_type": 1}, AdapterMethodStatus.INVALID, None),
        ({"peft_type": ""}, AdapterMethodStatus.INVALID, None),
        ({"peft_type": "IA3"}, AdapterMethodStatus.UNSUPPORTED, "IA3"),
    ],
)
def test_method_status_short_circuits_lora_schema(
    document: dict[str, object],
    status: AdapterMethodStatus,
    declared: str | None,
) -> None:
    document["future_method_field"] = {"opaque": [1, 2]}

    manifest = parse_config(document)

    assert manifest.method_status is status
    assert manifest.declared_peft_type == declared
    assert manifest.lora is None
    assert manifest.issues == ()
    assert not manifest.closed_profile


def test_empty_root_member_name_is_total_and_redacted_for_every_method_status() -> None:
    supported = parse_config({"peft_type": "LORA", "": None})
    missing = parse_config({"": None})
    unsupported = parse_config({"peft_type": "IA3", "": None})

    assert supported.explicit_fields == ("", "peft_type")
    assert require_lora(supported).r == 8
    assert issue_pairs(supported) == (("", ConfigFieldIssueKind.UNKNOWN_FIELD),)
    assert not supported.closed_profile
    assert missing.method_status is AdapterMethodStatus.MISSING
    assert missing.explicit_fields == ("",)
    assert missing.issues == ()
    assert unsupported.method_status is AdapterMethodStatus.UNSUPPORTED
    assert unsupported.explicit_fields == ("", "peft_type")
    assert unsupported.issues == ()


def test_auto_mapping_is_validated_but_never_imported() -> None:
    manifest = parse_config(
        {
            "peft_type": "LORA",
            "auto_mapping": {
                "base_model_class": "PrivateModel",
                "parent_library": "private_package.models",
            },
        }
    )

    mapping = require_lora(manifest).auto_mapping
    assert mapping == AutoMapping("PrivateModel", "private_package.models")
    assert manifest.closed_profile


def test_selector_forms_are_distinct_and_name_lists_are_canonical() -> None:
    named = require_lora(
        parse_config(
            {
                "peft_type": "LORA",
                "target_modules": ["v_proj", "q_proj", "v_proj"],
                "exclude_modules": ["lm_head", "embed_tokens", "lm_head"],
            }
        )
    )
    all_linear = require_lora(parse_config({"peft_type": "LORA", "target_modules": "ALL-LinEar"}))
    regex = require_lora(parse_config({"peft_type": "LORA", "exclude_modules": "^model\\.head$"}))

    assert named.target_modules == ModuleSelector(ModuleSelectorKind.NAMES, ("q_proj", "v_proj"))
    assert named.exclude_modules == ModuleSelector(
        ModuleSelectorKind.NAMES, ("embed_tokens", "lm_head")
    )
    assert all_linear.target_modules.kind is ModuleSelectorKind.ALL_LINEAR
    assert regex.exclude_modules == ModuleSelector(ModuleSelectorKind.REGEX, (r"^model\.head$",))


@pytest.mark.parametrize("target_parameters", [pytest.param(None, id="null")])
def test_empty_target_module_list_without_target_parameters_fails_closed(
    target_parameters: None,
) -> None:
    documents: tuple[dict[str, object], ...] = (
        {"peft_type": "LORA", "target_modules": []},
        {
            "peft_type": "LORA",
            "target_modules": [],
            "target_parameters": target_parameters,
        },
    )
    for document in documents:
        manifest = parse_config(document)

        assert manifest.lora is None
        assert ("target_modules", ConfigFieldIssueKind.INVALID_VALUE) in issue_pairs(manifest)
        assert not manifest.closed_profile


def test_empty_target_module_list_with_target_parameters_is_still_outside_profile() -> None:
    manifest = parse_config(
        {"peft_type": "LORA", "target_modules": [], "target_parameters": ["weight"]}
    )

    assert manifest.lora is None
    assert issue_pairs(manifest) == (("target_parameters", ConfigFieldIssueKind.UNSUPPORTED_VALUE),)


def test_numbers_are_exact_decimals_and_pattern_order_is_preserved() -> None:
    document = b"""{
      "peft_type":"LORA",
      "lora_alpha":8e-1,
      "lora_dropout":1e-1,
      "rank_pattern":{"second":16,"first":4},
      "alpha_pattern":{"later":2.50,"earlier":1e-2}
    }"""

    profile = require_lora(parse_adapter_config(document))

    assert profile.lora_alpha == Decimal("8e-1")
    assert profile.lora_dropout == Decimal("1e-1")
    assert profile.rank_pattern == (
        RankPatternEntry("second", 16),
        RankPatternEntry("first", 4),
    )
    assert profile.alpha_pattern == (
        AlphaPatternEntry("later", Decimal("2.50")),
        AlphaPatternEntry("earlier", Decimal("1e-2")),
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("lora_alpha", "0e" + "9" * 100),
        ("alpha_pattern", '{"pattern":0e' + "9" * 100 + "}"),
        ("lora_alpha", "1e-500000000"),
        ("alpha_pattern", '{"pattern":1e-500000000}'),
    ],
)
def test_decimal_exponent_outside_policy_is_a_redacted_field_issue(
    field_name: str, value: str
) -> None:
    member = f'"{field_name}":{value}'
    manifest = parse_adapter_config(f'{{"peft_type":"LORA",{member}}}'.encode())

    assert manifest.lora is None
    assert issue_pairs(manifest) == ((field_name, ConfigFieldIssueKind.INVALID_VALUE),)


@pytest.mark.parametrize(
    ("member_template", "field_name", "profile_survives"),
    [
        ('"r":{number}', "r", False),
        ('"layers_to_transform":{number}', "layers_to_transform", False),
        ('"rank_pattern":{{"x":{number}}}', "rank_pattern", False),
        ('"qalora_group_size":{number}', "qalora_group_size", True),
    ],
)
def test_large_integer_lexemes_remain_classified_when_custom_limit_exceeds_python_guard(
    member_template: str,
    field_name: str,
    profile_survives: bool,
) -> None:
    number = "9" * 5000
    member = member_template.format(number=number)
    limits = replace(DEFAULT_ADAPTER_CONFIG_LIMITS, max_json_number_chars=len(number))

    manifest = parse_adapter_config(f'{{"peft_type":"LORA",{member}}}'.encode(), limits=limits)

    assert (field_name, ConfigFieldIssueKind.INVALID_VALUE) in issue_pairs(manifest)
    assert (manifest.lora is not None) is profile_survives
    assert not manifest.closed_profile


def test_layer_indices_normalize_but_patterns_and_modules_preserve_order() -> None:
    profile = require_lora(
        parse_config(
            {
                "peft_type": "LORA",
                "target_modules": ["q_proj"],
                "layers_to_transform": [3, 1, 3],
                "layers_pattern": ["decoder", "layers"],
                "modules_to_save": ["z", "a", "z"],
            }
        )
    )

    assert profile.layers_to_transform == (1, 3)
    assert profile.layers_pattern == ("decoder", "layers")
    assert profile.modules_to_save == ("z", "a", "z")


def test_unknown_fields_are_redacted_blocking_issues() -> None:
    marker = "unknown-private-marker"

    manifest = parse_config({"peft_type": "LORA", marker: {"secret": marker}})

    assert require_lora(manifest).r == 8
    assert issue_pairs(manifest) == ((marker, ConfigFieldIssueKind.UNKNOWN_FIELD),)
    assert manifest.issues[0].blocking
    assert not manifest.closed_profile
    assert marker not in repr(manifest)
    assert marker not in repr(manifest.issues[0])


@pytest.mark.parametrize(
    ("field_name", "value", "kind"),
    [
        ("task_type", 1, ConfigFieldIssueKind.INVALID_TYPE),
        ("task_type", "UNKNOWN", ConfigFieldIssueKind.INVALID_VALUE),
        ("auto_mapping", [], ConfigFieldIssueKind.INVALID_TYPE),
        ("auto_mapping", {}, ConfigFieldIssueKind.INVALID_VALUE),
        (
            "auto_mapping",
            {"base_model_class": "", "parent_library": "x"},
            ConfigFieldIssueKind.INVALID_VALUE,
        ),
        (
            "auto_mapping",
            {"base_model_class": 1, "parent_library": "x"},
            ConfigFieldIssueKind.INVALID_TYPE,
        ),
        ("peft_version", 1, ConfigFieldIssueKind.INVALID_TYPE),
        ("peft_version", "", ConfigFieldIssueKind.INVALID_VALUE),
        ("base_model_name_or_path", 1, ConfigFieldIssueKind.INVALID_TYPE),
        ("revision", False, ConfigFieldIssueKind.INVALID_TYPE),
        ("inference_mode", 1, ConfigFieldIssueKind.INVALID_TYPE),
        ("r", 8.0, ConfigFieldIssueKind.INVALID_TYPE),
        ("r", 0, ConfigFieldIssueKind.INVALID_VALUE),
        ("target_modules", 1, ConfigFieldIssueKind.INVALID_TYPE),
        ("target_modules", [1], ConfigFieldIssueKind.INVALID_TYPE),
        ("target_modules", [""], ConfigFieldIssueKind.INVALID_VALUE),
        ("exclude_modules", "", ConfigFieldIssueKind.INVALID_VALUE),
        ("lora_alpha", True, ConfigFieldIssueKind.INVALID_TYPE),
        ("lora_alpha", 1e20, ConfigFieldIssueKind.INVALID_VALUE),
        ("lora_dropout", "0.1", ConfigFieldIssueKind.INVALID_TYPE),
        ("lora_dropout", -0.1, ConfigFieldIssueKind.INVALID_VALUE),
        ("fan_in_fan_out", 0, ConfigFieldIssueKind.INVALID_TYPE),
        ("bias", 1, ConfigFieldIssueKind.INVALID_TYPE),
        ("bias", "future", ConfigFieldIssueKind.INVALID_VALUE),
        ("use_rslora", 0, ConfigFieldIssueKind.INVALID_TYPE),
        ("modules_to_save", "head", ConfigFieldIssueKind.INVALID_TYPE),
        ("modules_to_save", [1], ConfigFieldIssueKind.INVALID_TYPE),
        ("modules_to_save", [""], ConfigFieldIssueKind.INVALID_VALUE),
        ("init_lora_weights", 1, ConfigFieldIssueKind.INVALID_TYPE),
        ("init_lora_weights", "future", ConfigFieldIssueKind.INVALID_VALUE),
        ("init_lora_weights", "pissa_niter_", ConfigFieldIssueKind.INVALID_VALUE),
        ("init_lora_weights", "pissa_niter_\u0661", ConfigFieldIssueKind.INVALID_VALUE),
        (
            "init_lora_weights",
            "pissa_niter_9223372036854775808",
            ConfigFieldIssueKind.INVALID_VALUE,
        ),
        (
            "init_lora_weights",
            "pissa_niter_" + "9" * 5000,
            ConfigFieldIssueKind.INVALID_VALUE,
        ),
        ("layers_to_transform", 1.0, ConfigFieldIssueKind.INVALID_TYPE),
        ("layers_to_transform", [1.0], ConfigFieldIssueKind.INVALID_TYPE),
        ("layers_to_transform", -1, ConfigFieldIssueKind.INVALID_VALUE),
        ("layers_to_transform", [-1], ConfigFieldIssueKind.INVALID_VALUE),
        ("layers_pattern", 1, ConfigFieldIssueKind.INVALID_TYPE),
        ("layers_pattern", [1], ConfigFieldIssueKind.INVALID_TYPE),
        ("layers_pattern", [""], ConfigFieldIssueKind.INVALID_VALUE),
        ("rank_pattern", None, ConfigFieldIssueKind.INVALID_TYPE),
        ("rank_pattern", {"x": 1.0}, ConfigFieldIssueKind.INVALID_TYPE),
        ("rank_pattern", {"x": 0}, ConfigFieldIssueKind.INVALID_VALUE),
        ("rank_pattern", {"": 1}, ConfigFieldIssueKind.INVALID_VALUE),
        ("alpha_pattern", None, ConfigFieldIssueKind.INVALID_TYPE),
        ("alpha_pattern", {"x": True}, ConfigFieldIssueKind.INVALID_TYPE),
        ("alpha_pattern", {"x": 1e20}, ConfigFieldIssueKind.INVALID_VALUE),
        ("alpha_pattern", {"": 1}, ConfigFieldIssueKind.INVALID_VALUE),
    ],
)
def test_invalid_profile_fields_are_classified_without_using_fallbacks(
    field_name: str,
    value: object,
    kind: ConfigFieldIssueKind,
) -> None:
    manifest = parse_config({"peft_type": "LORA", field_name: value})

    assert manifest.lora is None
    assert (field_name, kind) in issue_pairs(manifest)
    assert not manifest.closed_profile


@pytest.mark.parametrize("bias", ["all", "lora_only"])
def test_valid_nonordinary_bias_modes_are_retained_and_blocked(bias: str) -> None:
    manifest = parse_config({"peft_type": "LORA", "bias": bias})

    assert require_lora(manifest).bias is LoraBiasMode(bias)
    assert issue_pairs(manifest) == (("bias", ConfigFieldIssueKind.UNSUPPORTED_VALUE),)
    assert not manifest.closed_profile


@pytest.mark.parametrize(
    ("raw", "kind", "iterations"),
    [
        (False, LoraInitializerKind.RANDOM, None),
        ("gaussian", LoraInitializerKind.GAUSSIAN, None),
        ("GAUSSIAN", LoraInitializerKind.GAUSSIAN, None),
        ("eva", LoraInitializerKind.EVA, None),
        ("olora", LoraInitializerKind.OLORA, None),
        ("OlOrA", LoraInitializerKind.OLORA, None),
        ("pissa", LoraInitializerKind.PISSA, None),
        ("pissa_niter_0", LoraInitializerKind.PISSA_NITER, 0),
        ("corda", LoraInitializerKind.CORDA, None),
        ("loftq", LoraInitializerKind.LOFTQ, None),
        ("orthogonal", LoraInitializerKind.ORTHOGONAL, None),
        ("lora_ga", LoraInitializerKind.LORA_GA, None),
    ],
)
def test_valid_special_initializers_are_retained_and_blocked(
    raw: bool | str,
    kind: LoraInitializerKind,
    iterations: int | None,
) -> None:
    manifest = parse_config({"peft_type": "LORA", "init_lora_weights": raw})

    assert require_lora(manifest).initializer == LoraInitializer(kind, iterations)
    assert issue_pairs(manifest) == (("init_lora_weights", ConfigFieldIssueKind.UNSUPPORTED_VALUE),)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("megatron_config", {}),
        ("megatron_core", None),
        ("trainable_token_indices", []),
        ("loftq_config", {"loftq_bits": 4}),
        ("eva_config", {}),
        ("corda_config", {}),
        ("lora_ga_config", {}),
        ("use_dora", True),
        ("alora_invocation_tokens", []),
        ("use_qalora", True),
        ("qalora_group_size", 32),
        ("layer_replication", []),
        ("lora_bias", True),
        ("target_parameters", []),
        ("use_bdlora", {}),
        ("arrow_config", {}),
        ("ensure_weight_tying", True),
    ],
)
def test_valid_nondefault_variant_fields_are_unsupported_not_executed(
    field_name: str, value: object
) -> None:
    manifest = parse_config({"peft_type": "LORA", field_name: value})

    assert manifest.lora is not None
    assert issue_pairs(manifest) == ((field_name, ConfigFieldIssueKind.UNSUPPORTED_VALUE),)
    assert not manifest.closed_profile


@pytest.mark.parametrize(
    ("field_name", "value", "kind"),
    [
        ("megatron_config", [], ConfigFieldIssueKind.INVALID_TYPE),
        ("megatron_core", 1, ConfigFieldIssueKind.INVALID_TYPE),
        ("trainable_token_indices", "x", ConfigFieldIssueKind.INVALID_TYPE),
        ("loftq_config", None, ConfigFieldIssueKind.INVALID_TYPE),
        ("eva_config", [], ConfigFieldIssueKind.INVALID_TYPE),
        ("corda_config", [], ConfigFieldIssueKind.INVALID_TYPE),
        ("lora_ga_config", [], ConfigFieldIssueKind.INVALID_TYPE),
        ("use_dora", 1, ConfigFieldIssueKind.INVALID_TYPE),
        ("alora_invocation_tokens", {}, ConfigFieldIssueKind.INVALID_TYPE),
        ("use_qalora", 0, ConfigFieldIssueKind.INVALID_TYPE),
        ("qalora_group_size", 16.0, ConfigFieldIssueKind.INVALID_TYPE),
        ("qalora_group_size", 0, ConfigFieldIssueKind.INVALID_VALUE),
        ("layer_replication", {}, ConfigFieldIssueKind.INVALID_TYPE),
        ("lora_bias", 0, ConfigFieldIssueKind.INVALID_TYPE),
        ("target_parameters", {}, ConfigFieldIssueKind.INVALID_TYPE),
        ("use_bdlora", [], ConfigFieldIssueKind.INVALID_TYPE),
        ("arrow_config", [], ConfigFieldIssueKind.INVALID_TYPE),
        ("ensure_weight_tying", 1, ConfigFieldIssueKind.INVALID_TYPE),
    ],
)
def test_malformed_variant_fields_are_blocking_but_do_not_corrupt_core_profile(
    field_name: str,
    value: object,
    kind: ConfigFieldIssueKind,
) -> None:
    manifest = parse_config({"peft_type": "LORA", field_name: value})

    assert manifest.lora is not None
    assert issue_pairs(manifest) == ((field_name, kind),)
    assert not manifest.closed_profile


def test_runtime_config_is_ignored_regardless_of_shape() -> None:
    marker = "runtime-private-marker"

    manifest = parse_config({"peft_type": "LORA", "runtime_config": marker})

    assert issue_pairs(manifest) == (("runtime_config", ConfigFieldIssueKind.IGNORED_FIELD),)
    assert manifest.closed_profile
    assert marker not in repr(manifest)


@pytest.mark.parametrize(
    "document",
    [
        {"peft_type": "LORA", "target_modules": "x", "layers_to_transform": 1},
        {"peft_type": "LORA", "target_modules": "x", "layers_pattern": "layers"},
        {
            "peft_type": "LORA",
            "target_modules": "all-linear",
            "layers_to_transform": 1,
        },
        {
            "peft_type": "LORA",
            "target_modules": "ALL-LINEAR",
            "layers_pattern": "",
        },
        {"peft_type": "LORA", "target_modules": ["q"], "layers_pattern": "layers"},
        {
            "peft_type": "LORA",
            "target_modules": ["q"],
            "layers_to_transform": 0,
            "layers_pattern": "layers",
        },
        {
            "peft_type": "LORA",
            "target_modules": ["q"],
            "layers_to_transform": [],
            "layers_pattern": ["layers"],
        },
    ],
)
def test_invalid_layer_filter_combinations_do_not_produce_a_profile(
    document: dict[str, object],
) -> None:
    manifest = parse_config(document)

    assert manifest.lora is None
    assert any(issue.kind is ConfigFieldIssueKind.INVALID_VALUE for issue in manifest.issues)


def test_target_regex_without_layer_filters_is_a_valid_opaque_selector() -> None:
    profile = require_lora(
        parse_config({"peft_type": "LORA", "target_modules": r"^model\..*\.q_proj$"})
    )

    assert profile.target_modules.kind is ModuleSelectorKind.REGEX


def test_empty_scalar_layer_pattern_uses_peft_default_semantics() -> None:
    default_profile = require_lora(parse_config({"peft_type": "LORA", "layers_pattern": ""}))
    filtered_profile = require_lora(
        parse_config(
            {
                "peft_type": "LORA",
                "target_modules": ["q"],
                "layers_to_transform": [0],
                "layers_pattern": "",
            }
        )
    )

    assert default_profile.layers_pattern == ()
    assert filtered_profile.layers_to_transform == (0,)
    assert filtered_profile.layers_pattern == ()


@pytest.mark.parametrize(
    ("document", "code"),
    [
        (b"\xff", AdapterConfigErrorCode.DOCUMENT_UTF8),
        (b"{", AdapterConfigErrorCode.DOCUMENT_JSON),
        (b'{"peft_type":"LORA","peft_type":"LORA"}', AdapterConfigErrorCode.DUPLICATE_JSON_KEY),
        (rb'{"a":1,"\u0061":2}', AdapterConfigErrorCode.DUPLICATE_JSON_KEY),
        (b"[]", AdapterConfigErrorCode.ROOT_NOT_OBJECT),
        (rb'{"peft_type":"LORA","value":"\ud800"}', AdapterConfigErrorCode.INVALID_UNICODE_SCALAR),
        (rb'{"peft_type":"LORA","\ud800":null}', AdapterConfigErrorCode.INVALID_UNICODE_SCALAR),
    ],
)
def test_invalid_documents_have_static_classified_errors(
    document: bytes, code: AdapterConfigErrorCode
) -> None:
    with pytest.raises(InvalidAdapterConfig) as raised:
        parse_adapter_config(document)

    assert raised.value.code is code
    assert raised.value.rule_id == "PL001"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_parser_requires_exact_document_and_limit_types() -> None:
    with pytest.raises(TypeError, match=r"^adapter config document must be bytes$"):
        parse_adapter_config("{}")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"^limits must be AdapterConfigLimits$"):
        parse_adapter_config(b"{}", limits=object())  # type: ignore[arg-type]


def test_document_byte_limit_is_checked_at_the_exact_boundary() -> None:
    document = encode_config({"peft_type": "LORA"})

    parse_adapter_config(
        document,
        limits=replace(DEFAULT_ADAPTER_CONFIG_LIMITS, max_document_bytes=len(document)),
    )
    with pytest.raises(AdapterConfigLimitExceeded) as raised:
        parse_adapter_config(
            document,
            limits=replace(DEFAULT_ADAPTER_CONFIG_LIMITS, max_document_bytes=len(document) - 1),
        )

    assert raised.value.code is AdapterConfigErrorCode.DOCUMENT_EXCEEDS_POLICY_LIMIT
    assert raised.value.limit == len(document) - 1


@pytest.mark.parametrize(
    ("accepted", "rejected", "field", "limit", "code"),
    [
        (
            {"peft_type": "LORA"},
            {"peft_type": "LORA", "x": {}},
            "max_json_depth",
            1,
            AdapterConfigErrorCode.JSON_DEPTH_EXCEEDS_POLICY_LIMIT,
        ),
        (
            {"peft_type": "LORA", "123456789": None},
            {"peft_type": "LORA", "1234567890": None},
            "max_json_string_chars",
            9,
            AdapterConfigErrorCode.JSON_STRING_EXCEEDS_POLICY_LIMIT,
        ),
        (
            {},
            {"peft_type": "LORA"},
            "max_json_tokens",
            1,
            AdapterConfigErrorCode.JSON_TOKEN_EXCEEDS_POLICY_LIMIT,
        ),
        (
            {"peft_type": "LORA", "r": 12},
            {"peft_type": "LORA", "r": 123},
            "max_json_number_chars",
            2,
            AdapterConfigErrorCode.JSON_NUMBER_EXCEEDS_POLICY_LIMIT,
        ),
        (
            {"peft_type": "LORA"},
            {"peft_type": "LORA", "r": 8},
            "max_root_fields",
            1,
            AdapterConfigErrorCode.ROOT_FIELD_COUNT_EXCEEDS_POLICY_LIMIT,
        ),
        (
            {"peft_type": "LORA", "target_modules": ["q"]},
            {"peft_type": "LORA", "target_modules": ["q", "v"]},
            "max_collection_items",
            1,
            AdapterConfigErrorCode.COLLECTION_COUNT_EXCEEDS_POLICY_LIMIT,
        ),
        (
            {"peft_type": "LORA"},
            {"peft_type": "LORA", "1234567890": None},
            "max_name_bytes",
            9,
            AdapterConfigErrorCode.NAME_EXCEEDS_POLICY_LIMIT,
        ),
    ],
)
def test_each_json_and_structure_limit_has_an_exact_boundary(
    accepted: object,
    rejected: object,
    field: str,
    limit: int,
    code: AdapterConfigErrorCode,
) -> None:
    limits = replace(DEFAULT_ADAPTER_CONFIG_LIMITS, **{field: limit})

    parse_config(accepted, limits=limits)
    with pytest.raises(AdapterConfigLimitExceeded) as raised:
        parse_config(rejected, limits=limits)

    assert raised.value.code is code
    assert raised.value.limit == limit
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_nested_object_collection_limit_is_enforced() -> None:
    limits = replace(DEFAULT_ADAPTER_CONFIG_LIMITS, max_collection_items=1)

    parse_config({"opaque": {"a": None}}, limits=limits)
    with pytest.raises(AdapterConfigLimitExceeded) as raised:
        parse_config({"opaque": {"a": None, "b": None}}, limits=limits)

    assert raised.value.code is AdapterConfigErrorCode.COLLECTION_COUNT_EXCEEDS_POLICY_LIMIT
    assert raised.value.limit == 1


def test_retained_unsupported_method_obeys_name_limit() -> None:
    limits = replace(DEFAULT_ADAPTER_CONFIG_LIMITS, max_name_bytes=len("peft_type"))

    with pytest.raises(AdapterConfigLimitExceeded) as raised:
        parse_config({"peft_type": "UNSUPPORTED"}, limits=limits)

    assert raised.value.code is AdapterConfigErrorCode.NAME_EXCEEDS_POLICY_LIMIT
    assert raised.value.limit == len("peft_type")


@pytest.mark.parametrize(
    "field_name",
    [
        "max_document_bytes",
        "max_json_depth",
        "max_json_string_chars",
        "max_json_tokens",
        "max_json_number_chars",
        "max_root_fields",
        "max_collection_items",
        "max_name_bytes",
    ],
)
@pytest.mark.parametrize(("value", "error_type"), [(True, TypeError), (0, ValueError)])
def test_limits_reject_nonexact_or_nonpositive_values(
    field_name: str, value: object, error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        replace(DEFAULT_ADAPTER_CONFIG_LIMITS, **{field_name: value})  # type: ignore[arg-type]


def test_limits_and_public_models_are_frozen_and_reject_state_restore() -> None:
    manifest = parse_config({"peft_type": "LORA"})
    profile = require_lora(manifest)
    objects = [
        DEFAULT_ADAPTER_CONFIG_LIMITS,
        manifest,
        profile,
        profile.target_modules,
        profile.initializer,
        ConfigFieldIssue("x", ConfigFieldIssueKind.UNKNOWN_FIELD),
        AutoMapping("Model", "library"),
        RankPatternEntry("x", 1),
        AlphaPatternEntry("x", Decimal(1)),
    ]

    with pytest.raises(FrozenInstanceError):
        manifest.schema = "future"  # type: ignore[misc]
    for value in objects:
        with pytest.raises(TypeError, match="immutable"):
            value.__setstate__({})  # type: ignore[attr-defined]


def test_manifest_and_nested_repr_redact_all_user_controlled_text() -> None:
    marker = "private-marker"
    document = {
        "peft_type": "LORA",
        "base_model_name_or_path": marker,
        "revision": marker,
        "peft_version": marker,
        "target_modules": [marker],
        "exclude_modules": marker,
        "modules_to_save": [marker],
        "layers_to_transform": [0],
        "layers_pattern": [marker],
        "rank_pattern": {marker: 8},
        "alpha_pattern": {marker: 8},
        "auto_mapping": {"base_model_class": marker, "parent_library": marker},
    }

    manifest = parse_config(document)
    profile = require_lora(manifest)
    rendered = " ".join(
        repr(value)
        for value in (
            manifest,
            profile,
            profile.target_modules,
            profile.exclude_modules,
            profile.auto_mapping,
            profile.rank_pattern,
            profile.alpha_pattern,
        )
    )

    assert marker not in rendered
    assert profile.base_model_name_or_path == marker
    assert profile.target_modules.values == (marker,)


def test_errors_never_render_document_content() -> None:
    marker = "private-error-marker"

    with pytest.raises(InvalidAdapterConfig) as raised:
        parse_adapter_config(f'{{"{marker}":'.encode())

    rendered = f"{raised.value!s} {raised.value!r} {raised.value.args!r}"
    assert marker not in rendered


def test_error_categories_reject_invalid_manual_construction() -> None:
    class ConcreteInspectionError(AdapterConfigInspectionError):
        pass

    with pytest.raises(TypeError, match="abstract base class"):
        AdapterConfigInspectionError(AdapterConfigErrorCode.DOCUMENT_JSON)
    with pytest.raises(TypeError, match="error code must be"):
        ConcreteInspectionError("document_json")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="error code must be"):
        InvalidAdapterConfig("document_json")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not an invalid adapter config"):
        InvalidAdapterConfig(AdapterConfigErrorCode.JSON_DEPTH_EXCEEDS_POLICY_LIMIT)
    with pytest.raises(ValueError, match="not an inspection limit"):
        AdapterConfigLimitExceeded(AdapterConfigErrorCode.DOCUMENT_JSON, limit=1)
    with pytest.raises(TypeError, match="limit must be an integer"):
        AdapterConfigLimitExceeded(AdapterConfigErrorCode.DOCUMENT_EXCEEDS_POLICY_LIMIT, limit=True)
    with pytest.raises(ValueError, match="limit must not be negative"):
        AdapterConfigLimitExceeded(AdapterConfigErrorCode.DOCUMENT_EXCEEDS_POLICY_LIMIT, limit=-1)


def test_manual_selector_initializer_and_pattern_invariants() -> None:
    with pytest.raises(ValueError, match="must not contain values"):
        ModuleSelector(ModuleSelectorKind.DEFAULT, ("x",))
    with pytest.raises(ValueError, match="exactly one"):
        ModuleSelector(ModuleSelectorKind.REGEX, ())
    with pytest.raises(ValueError, match="sorted and unique"):
        ModuleSelector(ModuleSelectorKind.NAMES, ("z", "a"))
    with pytest.raises(ValueError, match="only valid"):
        LoraInitializer(LoraInitializerKind.DEFAULT, 1)
    with pytest.raises(ValueError, match="supported integer range"):
        LoraInitializer(LoraInitializerKind.PISSA_NITER, -1)
    with pytest.raises(ValueError, match="must not be empty"):
        RankPatternEntry("", 1)
    with pytest.raises(TypeError, match="must be Decimal"):
        AlphaPatternEntry("x", 1)  # type: ignore[arg-type]


def test_manual_profile_invariants_reject_forged_values() -> None:
    profile = require_lora(parse_config({"peft_type": "LORA"}))

    with pytest.raises(TypeError, match="r must be an integer"):
        replace(profile, r=True)
    with pytest.raises(ValueError, match="numeric range"):
        replace(profile, lora_dropout=Decimal(2))
    with pytest.raises(ValueError, match="string target_modules"):
        replace(
            profile,
            target_modules=ModuleSelector(ModuleSelectorKind.REGEX, ("x",)),
            layers_to_transform=(1,),
        )
    with pytest.raises(ValueError, match="non-empty named target_modules"):
        replace(profile, target_modules=ModuleSelector(ModuleSelectorKind.NAMES))
    with pytest.raises(ValueError, match="requires non-empty"):
        replace(profile, layers_pattern=("layers",), layers_to_transform=())
    with pytest.raises(ValueError, match="supported numeric range"):
        replace(profile, lora_alpha=Decimal("1e-10000"))
    with pytest.raises(ValueError, match="supported numeric range"):
        replace(profile, lora_alpha=Decimal("NaN"))
    with pytest.raises(TypeError, match="auto_mapping must be"):
        replace(profile, auto_mapping=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="target_modules must be"):
        replace(profile, target_modules=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exclude_modules must be"):
        replace(profile, exclude_modules=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="initializer must be"):
        replace(profile, initializer=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sorted and unique"):
        replace(profile, layers_to_transform=(2, 1))
    with pytest.raises(TypeError, match="revision must be a string or None"):
        replace(profile, revision=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="inference_mode must be a boolean"):
        replace(profile, inference_mode=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tuple of integers"):
        replace(profile, layers_to_transform=[1])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tuple of RankPatternEntry"):
        replace(profile, rank_pattern=[RankPatternEntry("x", 1)])  # type: ignore[arg-type]


def test_manual_text_and_enum_models_reject_forged_types() -> None:
    with pytest.raises(TypeError, match="kind must be ConfigFieldIssueKind"):
        ConfigFieldIssue("x", "unknown_field")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="field_name must be a string"):
        ConfigFieldIssue(1, ConfigFieldIssueKind.UNKNOWN_FIELD)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tuple of strings"):
        ModuleSelector(ModuleSelectorKind.NAMES, ["x"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty strings"):
        ModuleSelector(ModuleSelectorKind.NAMES, ("",))


def test_public_text_models_reject_lone_surrogates() -> None:
    surrogate = "\ud800"
    profile = require_lora(parse_config({"peft_type": "LORA"}))
    manifest = parse_config({"peft_type": "LORA"})

    with pytest.raises(ValueError, match="Unicode scalar"):
        AutoMapping(surrogate, "library")
    with pytest.raises(ValueError, match="Unicode scalar"):
        ModuleSelector(ModuleSelectorKind.REGEX, (surrogate,))
    with pytest.raises(ValueError, match="Unicode scalar"):
        RankPatternEntry(surrogate, 1)
    with pytest.raises(ValueError, match="Unicode scalar"):
        AlphaPatternEntry(surrogate, Decimal(1))
    with pytest.raises(ValueError, match="Unicode scalar"):
        ConfigFieldIssue(surrogate, ConfigFieldIssueKind.UNKNOWN_FIELD)
    with pytest.raises(ValueError, match="Unicode scalar"):
        replace(profile, base_model_name_or_path=surrogate)
    with pytest.raises(ValueError, match="Unicode scalar"):
        replace(manifest, explicit_fields=(surrogate, "peft_type"))


def test_aggregate_models_revalidate_and_copy_nested_evidence() -> None:
    manifest = parse_config({"peft_type": "LORA"})
    profile = require_lora(manifest)
    rebuilt = replace(manifest, lora=profile)

    assert rebuilt.lora is not profile
    assert require_lora(rebuilt).target_modules is not profile.target_modules
    assert require_lora(rebuilt).initializer is not profile.initializer

    forged_selector = ModuleSelector(ModuleSelectorKind.NAMES, ("q",))
    object.__setattr__(forged_selector, "values", ["q"])
    with pytest.raises(TypeError, match="tuple of strings"):
        replace(profile, target_modules=forged_selector)

    forged_profile = require_lora(parse_config({"peft_type": "LORA"}))
    object.__setattr__(forged_profile, "r", 0)
    with pytest.raises(ValueError, match="supported integer range"):
        replace(manifest, lora=forged_profile)

    unknown_manifest = parse_config({"peft_type": "LORA", "future": None})
    forged_issue = unknown_manifest.issues[0]
    object.__setattr__(forged_issue, "kind", "unknown_field")
    with pytest.raises(TypeError, match="kind must be ConfigFieldIssueKind"):
        replace(unknown_manifest, issues=(forged_issue,))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        pytest.param("task_type", PeftTaskType.CAUSAL_LM, id="task-type"),
        pytest.param("auto_mapping", AutoMapping("Model", "library"), id="auto-mapping"),
        pytest.param("peft_version", "future", id="peft-version"),
        pytest.param("base_model_name_or_path", "model", id="base-model"),
        pytest.param("revision", "main", id="revision"),
        pytest.param("inference_mode", True, id="inference-mode"),
        pytest.param("r", 16, id="rank"),
        pytest.param(
            "target_modules",
            ModuleSelector(ModuleSelectorKind.ALL_LINEAR),
            id="target-modules",
        ),
        pytest.param(
            "exclude_modules",
            ModuleSelector(ModuleSelectorKind.REGEX, ("head",)),
            id="exclude-modules",
        ),
        pytest.param("lora_alpha", Decimal(16), id="alpha"),
        pytest.param("lora_dropout", Decimal("0.5"), id="dropout"),
        pytest.param("fan_in_fan_out", True, id="fan-in-fan-out"),
        pytest.param("bias", LoraBiasMode.ALL, id="bias"),
        pytest.param("use_rslora", True, id="rslora"),
        pytest.param("modules_to_save", ("head",), id="modules-to-save"),
        pytest.param(
            "initializer",
            LoraInitializer(LoraInitializerKind.RANDOM),
            id="initializer",
        ),
        pytest.param("layers_to_transform", (0,), id="layers"),
        pytest.param("layers_pattern", (), id="layer-pattern"),
        pytest.param("rank_pattern", (RankPatternEntry("q", 16),), id="rank-pattern"),
        pytest.param(
            "alpha_pattern",
            (AlphaPatternEntry("q", Decimal(16)),),
            id="alpha-pattern",
        ),
    ],
)
def test_manifest_requires_source_fields_for_every_nondefault_profile_value(
    field_name: str,
    value: object,
) -> None:
    manifest = parse_config({"peft_type": "LORA"})
    profile = require_lora(manifest)
    object.__setattr__(profile, field_name, value)

    with pytest.raises(ValueError, match="require explicit source fields"):
        replace(manifest, lora=profile)


def test_manual_manifest_invariants_reject_inconsistent_evidence() -> None:
    manifest = parse_config({"peft_type": "LORA"})

    with pytest.raises(ValueError, match="schema must identify"):
        replace(manifest, schema="future")
    with pytest.raises(ValueError, match="requires declared"):
        replace(manifest, declared_peft_type="IA3")
    with pytest.raises(ValueError, match="canonical order"):
        replace(
            manifest,
            issues=(
                ConfigFieldIssue("z", ConfigFieldIssueKind.UNKNOWN_FIELD),
                ConfigFieldIssue("a", ConfigFieldIssueKind.UNKNOWN_FIELD),
            ),
        )
    with pytest.raises(ValueError, match="must be unique"):
        duplicate = ConfigFieldIssue("x", ConfigFieldIssueKind.UNKNOWN_FIELD)
        replace(manifest, issues=(duplicate, duplicate))
    with pytest.raises(ValueError, match="explicit_fields must be sorted"):
        replace(manifest, explicit_fields=("z", "a"))
    with pytest.raises(TypeError, match="lora must be"):
        replace(manifest, lora=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="explicit peft_type"):
        replace(manifest, explicit_fields=())
    with pytest.raises(ValueError, match="every issue must refer"):
        replace(
            manifest,
            issues=(ConfigFieldIssue("future", ConfigFieldIssueKind.UNKNOWN_FIELD),),
        )
    with pytest.raises(ValueError, match="exactly cover"):
        replace(manifest, explicit_fields=("future", "peft_type"))
    with pytest.raises(ValueError, match="unknown fields may only"):
        replace(
            manifest,
            explicit_fields=("future", "peft_type"),
            issues=(
                ConfigFieldIssue("future", ConfigFieldIssueKind.INVALID_VALUE),
                ConfigFieldIssue("future", ConfigFieldIssueKind.UNKNOWN_FIELD),
            ),
        )
    with pytest.raises(ValueError, match="runtime_config presence"):
        replace(manifest, explicit_fields=("peft_type", "runtime_config"))
    with pytest.raises(ValueError, match="only runtime_config"):
        replace(
            manifest,
            explicit_fields=("peft_type", "r"),
            issues=(ConfigFieldIssue("r", ConfigFieldIssueKind.IGNORED_FIELD),),
        )
    with pytest.raises(ValueError, match="requires a blocking"):
        replace(manifest, lora=None)
    with pytest.raises(ValueError, match="nonordinary profile"):
        replace(
            manifest,
            explicit_fields=("bias", "peft_type"),
            lora=replace(require_lora(manifest), bias=LoraBiasMode.ALL),
        )
    with pytest.raises(ValueError, match="nonordinary profile"):
        replace(
            manifest,
            explicit_fields=("init_lora_weights", "peft_type"),
            lora=replace(
                require_lora(manifest),
                initializer=LoraInitializer(LoraInitializerKind.RANDOM),
            ),
        )
    with pytest.raises(ValueError, match="declaration flag"):
        replace(
            manifest,
            lora=replace(require_lora(manifest), peft_version_was_declared=True),
        )
    with pytest.raises(ValueError, match="undeclared PEFT version"):
        replace(
            manifest,
            explicit_fields=("peft_type", "peft_version"),
            lora=replace(require_lora(manifest), peft_version="future"),
        )

    null_version = parse_config({"peft_type": "LORA", "peft_version": None})
    assert not require_lora(null_version).peft_version_was_declared
    assert null_version.closed_profile

    missing = parse_config({})
    with pytest.raises(ValueError, match="must not retain peft_type"):
        replace(missing, declared_peft_type="LORA")
    with pytest.raises(ValueError, match="must not carry"):
        replace(missing, lora=require_lora(manifest))
    with pytest.raises(ValueError, match="status is inconsistent"):
        replace(missing, explicit_fields=("peft_type",))

    invalid = parse_config({"peft_type": None})
    with pytest.raises(ValueError, match="status is inconsistent"):
        replace(invalid, explicit_fields=())

    unsupported = parse_config({"peft_type": "IA3"})
    with pytest.raises(ValueError, match="cannot declare LORA"):
        replace(unsupported, declared_peft_type="LORA")
    with pytest.raises(ValueError, match="must not carry"):
        replace(unsupported, lora=require_lora(manifest))
    with pytest.raises(ValueError, match="explicit peft_type"):
        replace(unsupported, explicit_fields=())


def test_issue_deduplication_and_unknown_internal_errors_fail_closed() -> None:
    issue = ConfigFieldIssue("x", ConfigFieldIssueKind.UNKNOWN_FIELD)
    issues = [issue]

    _add_issue(issues, "x", ConfigFieldIssueKind.UNKNOWN_FIELD)

    assert issues == [issue]

    class FutureBoundedJsonError(BoundedJsonError):
        pass

    with pytest.raises(TypeError, match="unsupported bounded JSON error type"):
        _raise_adapter_json_error(FutureBoundedJsonError(BoundedJsonErrorCode.SYNTAX))
