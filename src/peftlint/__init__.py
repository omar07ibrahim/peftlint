"""Static compatibility checks for LoRA adapters."""

from peftlint.evidence import (
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
from peftlint.safetensors import (
    DEFAULT_SAFETENSORS_LIMITS,
    HeaderNotice,
    HeaderReadPlan,
    InvalidSafetensors,
    MetadataForm,
    SafetensorsDtype,
    SafetensorsErrorCode,
    SafetensorsInspectionError,
    SafetensorsLimitExceeded,
    SafetensorsLimits,
    SafetensorsManifest,
    SafetensorsReadMismatch,
    TensorManifest,
    parse_safetensors_manifest,
)

__all__ = [
    "DEFAULT_SAFETENSORS_LIMITS",
    "LORA_V1_LOAD_RULES",
    "LORA_V1_RULESET",
    "EvidenceField",
    "HeaderNotice",
    "HeaderReadPlan",
    "InvalidSafetensors",
    "MetadataForm",
    "Profile",
    "ProfileSummary",
    "RuleOutcome",
    "RuleResult",
    "SafetensorsDtype",
    "SafetensorsErrorCode",
    "SafetensorsInspectionError",
    "SafetensorsLimitExceeded",
    "SafetensorsLimits",
    "SafetensorsManifest",
    "SafetensorsReadMismatch",
    "Severity",
    "TensorManifest",
    "Verdict",
    "parse_safetensors_manifest",
    "summarize_load",
]

__version__ = "0.1.0.dev0"
