"""Pure, bounded primitives for inspecting safetensors headers.

This module deliberately has no file, stream, network, or memory-mapping API.
Callers must inspect the eight-byte prefix first, use :func:`plan_header_read`
to obtain a bounded range, and only then supply exactly that range to
:func:`accept_header`. Plans are value objects, not provenance attestations:
source adapters remain responsible for binding both reads to one unchanged
object and for distinguishing transport failures from malformed artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import NoReturn, cast

from peftlint._bounded_json import (
    BoundedJsonError,
    BoundedJsonErrorCode,
    IntegerLexeme,
    InvalidJson,
    JsonLimitExceeded,
    JsonLimits,
    decode_json,
)

SAFETENSORS_HEADER_LIMIT = 100_000_000
"""Maximum header length accepted by the pinned safetensors v0.8 format."""

_U64_MAX = 2**64 - 1


class SafetensorsErrorCode(StrEnum):
    """Stable machine-readable failures produced by header inspection."""

    PREFIX_LENGTH = "prefix_length"
    FILE_TOO_SMALL = "file_too_small"
    HEADER_OUT_OF_BOUNDS = "header_out_of_bounds"
    HEADER_EXCEEDS_FORMAT_LIMIT = "header_exceeds_format_limit"
    HEADER_EXCEEDS_POLICY_LIMIT = "header_exceeds_policy_limit"
    HEADER_LENGTH_MISMATCH = "header_length_mismatch"
    HEADER_UTF8 = "header_utf8"
    HEADER_JSON = "header_json"
    JSON_DEPTH_EXCEEDS_POLICY_LIMIT = "json_depth_exceeds_policy_limit"
    JSON_STRING_EXCEEDS_POLICY_LIMIT = "json_string_exceeds_policy_limit"
    JSON_TOKEN_EXCEEDS_POLICY_LIMIT = "json_token_exceeds_policy_limit"
    DUPLICATE_JSON_KEY = "duplicate_json_key"
    INVALID_UNICODE_SCALAR = "invalid_unicode_scalar"
    HEADER_NOT_OBJECT = "header_not_object"
    INVALID_METADATA = "invalid_metadata"
    METADATA_COUNT_EXCEEDS_POLICY_LIMIT = "metadata_count_exceeds_policy_limit"
    TENSOR_COUNT_EXCEEDS_POLICY_LIMIT = "tensor_count_exceeds_policy_limit"
    TENSOR_NAME_EXCEEDS_POLICY_LIMIT = "tensor_name_exceeds_policy_limit"
    INVALID_TENSOR_RECORD = "invalid_tensor_record"
    MISSING_TENSOR_FIELD = "missing_tensor_field"
    INVALID_TENSOR_FIELD = "invalid_tensor_field"
    INTEGER_OUT_OF_RANGE = "integer_out_of_range"
    TENSOR_RANK_EXCEEDS_POLICY_LIMIT = "tensor_rank_exceeds_policy_limit"
    INVALID_DTYPE = "invalid_dtype"
    TENSOR_SIZE_OVERFLOW = "tensor_size_overflow"
    TENSOR_BYTE_MISALIGNED = "tensor_byte_misaligned"
    TENSOR_OFFSETS_REVERSED = "tensor_offsets_reversed"
    TENSOR_LAYOUT_GAP = "tensor_layout_gap"
    TENSOR_LAYOUT_OVERLAP = "tensor_layout_overlap"
    TENSOR_SIZE_MISMATCH = "tensor_size_mismatch"
    PAYLOAD_SIZE_MISMATCH = "payload_size_mismatch"


_ERROR_MESSAGES = {
    SafetensorsErrorCode.PREFIX_LENGTH: "safetensors prefix read must contain exactly 8 bytes",
    SafetensorsErrorCode.FILE_TOO_SMALL: "safetensors file is smaller than its prefix",
    SafetensorsErrorCode.HEADER_OUT_OF_BOUNDS: (
        "declared safetensors header does not fit inside the file"
    ),
    SafetensorsErrorCode.HEADER_EXCEEDS_FORMAT_LIMIT: (
        "declared safetensors header exceeds the format limit"
    ),
    SafetensorsErrorCode.HEADER_EXCEEDS_POLICY_LIMIT: (
        "declared safetensors header exceeds the inspection limit"
    ),
    SafetensorsErrorCode.HEADER_LENGTH_MISMATCH: (
        "provided safetensors header length does not match its prefix"
    ),
    SafetensorsErrorCode.HEADER_UTF8: "safetensors header is not valid UTF-8",
    SafetensorsErrorCode.HEADER_JSON: "safetensors header is not valid JSON",
    SafetensorsErrorCode.JSON_DEPTH_EXCEEDS_POLICY_LIMIT: (
        "safetensors header exceeds the JSON nesting limit"
    ),
    SafetensorsErrorCode.JSON_STRING_EXCEEDS_POLICY_LIMIT: (
        "safetensors header exceeds the JSON string limit"
    ),
    SafetensorsErrorCode.JSON_TOKEN_EXCEEDS_POLICY_LIMIT: (
        "safetensors header exceeds the JSON token limit"
    ),
    SafetensorsErrorCode.DUPLICATE_JSON_KEY: "safetensors header contains a duplicate JSON key",
    SafetensorsErrorCode.INVALID_UNICODE_SCALAR: (
        "safetensors header contains an invalid Unicode scalar"
    ),
    SafetensorsErrorCode.HEADER_NOT_OBJECT: "safetensors header root must be a JSON object",
    SafetensorsErrorCode.INVALID_METADATA: "safetensors metadata must contain only strings",
    SafetensorsErrorCode.METADATA_COUNT_EXCEEDS_POLICY_LIMIT: (
        "safetensors header exceeds the metadata entry limit"
    ),
    SafetensorsErrorCode.TENSOR_COUNT_EXCEEDS_POLICY_LIMIT: (
        "safetensors header exceeds the tensor count limit"
    ),
    SafetensorsErrorCode.TENSOR_NAME_EXCEEDS_POLICY_LIMIT: (
        "safetensors header exceeds the tensor name limit"
    ),
    SafetensorsErrorCode.INVALID_TENSOR_RECORD: ("safetensors tensor entry must be a JSON object"),
    SafetensorsErrorCode.MISSING_TENSOR_FIELD: (
        "safetensors tensor entry is missing a required field"
    ),
    SafetensorsErrorCode.INVALID_TENSOR_FIELD: (
        "safetensors tensor entry contains an invalid field"
    ),
    SafetensorsErrorCode.INTEGER_OUT_OF_RANGE: (
        "safetensors tensor entry contains an out-of-range integer"
    ),
    SafetensorsErrorCode.TENSOR_RANK_EXCEEDS_POLICY_LIMIT: (
        "safetensors header exceeds the tensor rank limit"
    ),
    SafetensorsErrorCode.INVALID_DTYPE: "safetensors tensor uses an invalid dtype",
    SafetensorsErrorCode.TENSOR_SIZE_OVERFLOW: (
        "safetensors tensor size exceeds unsigned 64-bit arithmetic"
    ),
    SafetensorsErrorCode.TENSOR_BYTE_MISALIGNED: (
        "safetensors sub-byte tensor does not end on a byte boundary"
    ),
    SafetensorsErrorCode.TENSOR_OFFSETS_REVERSED: ("safetensors tensor offsets are reversed"),
    SafetensorsErrorCode.TENSOR_LAYOUT_GAP: "safetensors tensor layout contains a gap",
    SafetensorsErrorCode.TENSOR_LAYOUT_OVERLAP: ("safetensors tensor layout contains an overlap"),
    SafetensorsErrorCode.TENSOR_SIZE_MISMATCH: (
        "safetensors tensor span does not match its dtype and shape"
    ),
    SafetensorsErrorCode.PAYLOAD_SIZE_MISMATCH: (
        "safetensors tensor layout does not cover the declared payload"
    ),
}

_PL101_ERROR_CODES = frozenset(
    {
        SafetensorsErrorCode.TENSOR_SIZE_OVERFLOW,
        SafetensorsErrorCode.TENSOR_BYTE_MISALIGNED,
        SafetensorsErrorCode.TENSOR_OFFSETS_REVERSED,
        SafetensorsErrorCode.TENSOR_LAYOUT_GAP,
        SafetensorsErrorCode.TENSOR_LAYOUT_OVERLAP,
        SafetensorsErrorCode.TENSOR_SIZE_MISMATCH,
        SafetensorsErrorCode.PAYLOAD_SIZE_MISMATCH,
    }
)

_INVALID_ERROR_CODES = frozenset(
    {
        SafetensorsErrorCode.FILE_TOO_SMALL,
        SafetensorsErrorCode.HEADER_OUT_OF_BOUNDS,
        SafetensorsErrorCode.HEADER_EXCEEDS_FORMAT_LIMIT,
        SafetensorsErrorCode.HEADER_UTF8,
        SafetensorsErrorCode.HEADER_JSON,
        SafetensorsErrorCode.DUPLICATE_JSON_KEY,
        SafetensorsErrorCode.INVALID_UNICODE_SCALAR,
        SafetensorsErrorCode.HEADER_NOT_OBJECT,
        SafetensorsErrorCode.INVALID_METADATA,
        SafetensorsErrorCode.INVALID_TENSOR_RECORD,
        SafetensorsErrorCode.MISSING_TENSOR_FIELD,
        SafetensorsErrorCode.INVALID_TENSOR_FIELD,
        SafetensorsErrorCode.INTEGER_OUT_OF_RANGE,
        SafetensorsErrorCode.INVALID_DTYPE,
        *_PL101_ERROR_CODES,
    }
)
_READ_MISMATCH_ERROR_CODES = frozenset(
    {
        SafetensorsErrorCode.PREFIX_LENGTH,
        SafetensorsErrorCode.HEADER_LENGTH_MISMATCH,
    }
)
_LIMIT_ERROR_CODES = frozenset(
    {
        SafetensorsErrorCode.HEADER_EXCEEDS_POLICY_LIMIT,
        SafetensorsErrorCode.JSON_DEPTH_EXCEEDS_POLICY_LIMIT,
        SafetensorsErrorCode.JSON_STRING_EXCEEDS_POLICY_LIMIT,
        SafetensorsErrorCode.JSON_TOKEN_EXCEEDS_POLICY_LIMIT,
        SafetensorsErrorCode.TENSOR_COUNT_EXCEEDS_POLICY_LIMIT,
        SafetensorsErrorCode.METADATA_COUNT_EXCEEDS_POLICY_LIMIT,
        SafetensorsErrorCode.TENSOR_NAME_EXCEEDS_POLICY_LIMIT,
        SafetensorsErrorCode.TENSOR_RANK_EXCEEDS_POLICY_LIMIT,
    }
)


class SafetensorsInspectionError(Exception):
    """Base class for safe, classified safetensors inspection failures."""

    code: SafetensorsErrorCode

    def __init__(self, code: SafetensorsErrorCode) -> None:
        if type(self) is SafetensorsInspectionError:
            raise TypeError("SafetensorsInspectionError is an abstract base class")
        if type(code) is not SafetensorsErrorCode:
            raise TypeError("safetensors error code must be SafetensorsErrorCode")
        self.code = code
        super().__init__(_ERROR_MESSAGES[code])


class InvalidSafetensors(SafetensorsInspectionError):
    """The supplied evidence contradicts the safetensors envelope format."""

    rule_id: str

    def __init__(self, code: SafetensorsErrorCode) -> None:
        _require_error_category(code, _INVALID_ERROR_CODES, "invalid safetensors")
        self.rule_id = "PL101" if code in _PL101_ERROR_CODES else "PL100"
        super().__init__(code)


class SafetensorsReadMismatch(SafetensorsInspectionError):
    """Acquired bytes do not satisfy a planned exact read."""

    def __init__(self, code: SafetensorsErrorCode) -> None:
        _require_error_category(code, _READ_MISMATCH_ERROR_CODES, "read mismatch")
        super().__init__(code)


class SafetensorsLimitExceeded(SafetensorsInspectionError):
    """Inspection stopped at a configured resource boundary."""

    limit: int

    def __init__(self, code: SafetensorsErrorCode, *, limit: int) -> None:
        _require_error_category(code, _LIMIT_ERROR_CODES, "inspection limit")
        if type(limit) is not int:
            raise TypeError("limit must be an integer")
        if limit < 0:
            raise ValueError("limit must not be negative")
        self.limit = limit
        super().__init__(code)


def _require_error_category(
    code: SafetensorsErrorCode,
    allowed: frozenset[SafetensorsErrorCode],
    category: str,
) -> None:
    if type(code) is not SafetensorsErrorCode:
        raise TypeError("safetensors error code must be SafetensorsErrorCode")
    if code not in allowed:
        raise ValueError(f"{code.value} is not an {category} error code")


@dataclass(frozen=True, slots=True)
class SafetensorsLimits:
    """Resource limits applied before untrusted header bytes are accepted."""

    max_header_bytes: int = 16 * 1024 * 1024
    max_json_depth: int = 32
    max_json_string_chars: int = 1024 * 1024
    max_json_tokens: int = 250_000
    max_tensors: int = 10_000
    max_tensor_rank: int = 32
    max_tensor_name_bytes: int = 4096
    max_metadata_entries: int = 10_000

    def __post_init__(self) -> None:
        if type(self.max_header_bytes) is not int:
            raise TypeError("max_header_bytes must be an integer")
        if not 1 <= self.max_header_bytes <= SAFETENSORS_HEADER_LIMIT:
            raise ValueError("max_header_bytes must be between 1 and the safetensors format limit")
        if type(self.max_json_depth) is not int:
            raise TypeError("max_json_depth must be an integer")
        if self.max_json_depth < 1:
            raise ValueError("max_json_depth must be positive")
        if type(self.max_json_tokens) is not int:
            raise TypeError("max_json_tokens must be an integer")
        if self.max_json_tokens < 1:
            raise ValueError("max_json_tokens must be positive")
        for name, value in (
            ("max_json_string_chars", self.max_json_string_chars),
            ("max_tensors", self.max_tensors),
            ("max_tensor_rank", self.max_tensor_rank),
            ("max_tensor_name_bytes", self.max_tensor_name_bytes),
            ("max_metadata_entries", self.max_metadata_entries),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must not be negative")

    def __setstate__(self, _state: object) -> NoReturn:
        """Reject state restoration that could bypass validation."""

        raise TypeError(f"{type(self).__name__} is immutable")


DEFAULT_SAFETENSORS_LIMITS = SafetensorsLimits()


@dataclass(frozen=True, slots=True)
class HeaderReadPlan:
    """The only byte range a source adapter may read after the prefix."""

    file_size: int
    header_size: int
    header_offset: int
    data_offset: int
    data_size: int
    limits: SafetensorsLimits

    def __post_init__(self) -> None:
        values = (
            self.file_size,
            self.header_size,
            self.header_offset,
            self.data_offset,
            self.data_size,
        )
        if not all(type(value) is int for value in values):
            raise TypeError("header read plan sizes must be integers")
        if type(self.limits) is not SafetensorsLimits:
            raise TypeError("header read plan limits must be SafetensorsLimits")
        object.__setattr__(self, "limits", _validated_limits_copy(self.limits))
        if self.file_size < 8:
            raise ValueError("header read plan file_size must be at least 8")
        if not 0 <= self.header_size <= self.limits.max_header_bytes:
            raise ValueError("header read plan exceeds its inspection limit")
        if self.header_offset != 8:
            raise ValueError("header read plan header_offset must be 8")
        if self.data_offset != 8 + self.header_size:
            raise ValueError("header read plan data_offset is inconsistent")
        if self.data_size < 0 or self.data_size != self.file_size - self.data_offset:
            raise ValueError("header read plan data_size is inconsistent")

    def __setstate__(self, _state: object) -> NoReturn:
        """Reject state restoration that could bypass validation."""

        raise TypeError(f"{type(self).__name__} is immutable")


@dataclass(frozen=True, slots=True)
class HeaderEnvelope:
    """A size-checked header and its payload boundary.

    Header bytes are intentionally omitted from ``repr`` so metadata cannot be
    copied into diagnostics accidentally.
    """

    plan: HeaderReadPlan
    header: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.plan) is not HeaderReadPlan:
            raise TypeError("plan must be HeaderReadPlan")
        if type(self.header) is not bytes:
            raise TypeError("header must be bytes")
        object.__setattr__(self, "plan", _validated_plan_copy(self.plan))
        if len(self.header) != self.plan.header_size:
            raise SafetensorsReadMismatch(SafetensorsErrorCode.HEADER_LENGTH_MISMATCH)

    def __setstate__(self, _state: object) -> NoReturn:
        """Reject state restoration that could bypass validation."""

        raise TypeError(f"{type(self).__name__} is immutable")


class MetadataForm(StrEnum):
    """How the optional ``__metadata__`` member appeared in the header."""

    ABSENT = "absent"
    NULL = "null"
    OBJECT = "object"


class HeaderNotice(StrEnum):
    """Non-fatal format details retained for deterministic reporting."""

    METADATA_NULL = "metadata_null"
    UNKNOWN_TENSOR_FIELDS = "unknown_tensor_fields"


class SafetensorsDtype(StrEnum):
    """The exact dtype tokens understood by safetensors v0.8."""

    F4 = "F4"
    F6_E2M3 = "F6_E2M3"
    F6_E3M2 = "F6_E3M2"
    BOOL = "BOOL"
    U8 = "U8"
    I8 = "I8"
    F8_E5M2 = "F8_E5M2"
    F8_E4M3 = "F8_E4M3"
    F8_E8M0 = "F8_E8M0"
    F8_E4M3FNUZ = "F8_E4M3FNUZ"
    F8_E5M2FNUZ = "F8_E5M2FNUZ"
    I16 = "I16"
    U16 = "U16"
    F16 = "F16"
    BF16 = "BF16"
    I32 = "I32"
    U32 = "U32"
    F32 = "F32"
    C64 = "C64"
    F64 = "F64"
    I64 = "I64"
    U64 = "U64"

    @property
    def bits_per_element(self) -> int:
        """Return the v0.8 storage width for one logical element."""

        return _DTYPE_BITS[self]


_DTYPE_BITS: Mapping[SafetensorsDtype, int] = MappingProxyType(
    {
        SafetensorsDtype.F4: 4,
        SafetensorsDtype.F6_E2M3: 6,
        SafetensorsDtype.F6_E3M2: 6,
        SafetensorsDtype.BOOL: 8,
        SafetensorsDtype.U8: 8,
        SafetensorsDtype.I8: 8,
        SafetensorsDtype.F8_E5M2: 8,
        SafetensorsDtype.F8_E4M3: 8,
        SafetensorsDtype.F8_E8M0: 8,
        SafetensorsDtype.F8_E4M3FNUZ: 8,
        SafetensorsDtype.F8_E5M2FNUZ: 8,
        SafetensorsDtype.I16: 16,
        SafetensorsDtype.U16: 16,
        SafetensorsDtype.F16: 16,
        SafetensorsDtype.BF16: 16,
        SafetensorsDtype.I32: 32,
        SafetensorsDtype.U32: 32,
        SafetensorsDtype.F32: 32,
        SafetensorsDtype.C64: 64,
        SafetensorsDtype.F64: 64,
        SafetensorsDtype.I64: 64,
        SafetensorsDtype.U64: 64,
    }
)


def _has_lone_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _validate_tensor_components(
    *,
    name: object,
    dtype: object,
    shape: object,
    data_offsets: object,
    unknown_fields: object,
) -> None:
    if type(name) is not str or type(dtype) is not str:
        raise TypeError("tensor name and dtype must be strings")
    if _has_lone_surrogate(name) or _has_lone_surrogate(dtype):
        raise ValueError("tensor name and dtype must contain valid Unicode scalars")
    if name == "__metadata__":
        raise ValueError("__metadata__ is reserved and cannot be a tensor name")
    if type(shape) is not tuple or not all(type(size) is int for size in shape):
        raise TypeError("tensor shape must be a tuple of integers")
    if any(not 0 <= size <= _U64_MAX for size in shape):
        raise ValueError("tensor shape integers must fit unsigned 64-bit values")
    if (
        type(data_offsets) is not tuple
        or len(data_offsets) != 2
        or not all(type(offset) is int for offset in data_offsets)
    ):
        raise TypeError("tensor data_offsets must contain exactly two integers")
    if any(not 0 <= offset <= _U64_MAX for offset in data_offsets):
        raise ValueError("tensor offsets must fit unsigned 64-bit values")
    if type(unknown_fields) is not tuple or not all(
        type(field_name) is str for field_name in unknown_fields
    ):
        raise TypeError("unknown_fields must be a tuple of strings")
    if any(_has_lone_surrogate(field_name) for field_name in unknown_fields):
        raise ValueError("unknown field names must contain valid Unicode scalars")
    if unknown_fields != tuple(sorted(set(unknown_fields))):
        raise ValueError("unknown field names must be sorted and unique")
    if frozenset(unknown_fields) & {"dtype", "shape", "data_offsets"}:
        raise ValueError("required tensor fields cannot be unknown fields")


@dataclass(frozen=True, slots=True)
class TensorHeader:
    """One schema-decoded tensor entry whose storage is not yet validated."""

    name: str = field(repr=False)
    dtype: str = field(repr=False)
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    unknown_fields: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        _validate_tensor_components(
            name=self.name,
            dtype=self.dtype,
            shape=self.shape,
            data_offsets=self.data_offsets,
            unknown_fields=self.unknown_fields,
        )

    def __setstate__(self, _state: object) -> NoReturn:
        """Reject state restoration that could bypass validation."""

        raise TypeError(f"{type(self).__name__} is immutable")


@dataclass(frozen=True, slots=True)
class DecodedSafetensorsHeader:
    """An order-normalized JSON header, before tensor storage validation."""

    plan: HeaderReadPlan
    tensors: tuple[TensorHeader, ...] = field(repr=False)
    metadata: tuple[tuple[str, str], ...] = field(repr=False)
    metadata_form: MetadataForm
    notices: tuple[HeaderNotice, ...]

    def __post_init__(self) -> None:
        if type(self.plan) is not HeaderReadPlan:
            raise TypeError("decoded header plan must be HeaderReadPlan")
        if type(self.tensors) is not tuple or not all(
            type(tensor) is TensorHeader for tensor in self.tensors
        ):
            raise TypeError("decoded header tensors must be a tuple of TensorHeader values")
        object.__setattr__(self, "plan", _validated_plan_copy(self.plan))
        object.__setattr__(
            self,
            "tensors",
            tuple(_validated_tensor_header_copy(tensor) for tensor in self.tensors),
        )
        tensor_names = tuple(tensor.name for tensor in self.tensors)
        if tensor_names != tuple(sorted(set(tensor_names))):
            raise ValueError("decoded header tensor names must be sorted and unique")
        if len(self.tensors) > self.plan.limits.max_tensors:
            raise ValueError("decoded header exceeds its tensor count limit")
        for tensor in self.tensors:
            if len(tensor.name.encode("utf-8")) > self.plan.limits.max_tensor_name_bytes:
                raise ValueError("decoded header exceeds its tensor name limit")
            if len(tensor.shape) > self.plan.limits.max_tensor_rank:
                raise ValueError("decoded header exceeds its tensor rank limit")

        if type(self.metadata) is not tuple or not all(
            type(entry) is tuple
            and len(entry) == 2
            and type(entry[0]) is str
            and type(entry[1]) is str
            for entry in self.metadata
        ):
            raise TypeError("decoded header metadata must contain string pairs")
        metadata_keys = tuple(entry[0] for entry in self.metadata)
        if metadata_keys != tuple(sorted(set(metadata_keys))):
            raise ValueError("decoded header metadata keys must be sorted and unique")
        if any(
            _has_lone_surrogate(key) or _has_lone_surrogate(value) for key, value in self.metadata
        ):
            raise ValueError("decoded header metadata must contain valid Unicode scalars")
        if len(self.metadata) > self.plan.limits.max_metadata_entries:
            raise ValueError("decoded header exceeds its metadata entry limit")
        if type(self.metadata_form) is not MetadataForm:
            raise TypeError("metadata_form must be MetadataForm")
        if self.metadata_form is not MetadataForm.OBJECT and self.metadata:
            raise ValueError("absent or null metadata cannot contain entries")

        if type(self.notices) is not tuple or not all(
            type(notice) is HeaderNotice for notice in self.notices
        ):
            raise TypeError("decoded header notices must be a tuple of HeaderNotice values")
        if self.notices != tuple(sorted(set(self.notices), key=lambda notice: notice.value)):
            raise ValueError("decoded header notices must be sorted and unique")
        expected_notices: set[HeaderNotice] = set()
        if self.metadata_form is MetadataForm.NULL:
            expected_notices.add(HeaderNotice.METADATA_NULL)
        if any(tensor.unknown_fields for tensor in self.tensors):
            expected_notices.add(HeaderNotice.UNKNOWN_TENSOR_FIELDS)
        if frozenset(self.notices) != expected_notices:
            raise ValueError("decoded header notices are inconsistent")

    def __setstate__(self, _state: object) -> NoReturn:
        """Reject state restoration that could bypass validation."""

        raise TypeError(f"{type(self).__name__} is immutable")


@dataclass(frozen=True, slots=True)
class TensorManifest:
    """One tensor with a dtype-, shape-, and span-consistent byte range."""

    name: str = field(repr=False)
    dtype: SafetensorsDtype
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    unknown_fields: tuple[str, ...] = field(default=(), repr=False)
    _element_count: int = field(init=False, repr=False)
    _nbytes: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.dtype) is not SafetensorsDtype:
            raise TypeError("tensor manifest dtype must be SafetensorsDtype")
        _validate_tensor_components(
            name=self.name,
            dtype=self.dtype.value,
            shape=self.shape,
            data_offsets=self.data_offsets,
            unknown_fields=self.unknown_fields,
        )
        begin, end = self.data_offsets
        if end < begin:
            raise InvalidSafetensors(SafetensorsErrorCode.TENSOR_OFFSETS_REVERSED)
        element_count, expected_bytes = _checked_tensor_size(self.shape, self.dtype)
        if end - begin != expected_bytes:
            raise InvalidSafetensors(SafetensorsErrorCode.TENSOR_SIZE_MISMATCH)
        object.__setattr__(self, "_element_count", element_count)
        object.__setattr__(self, "_nbytes", expected_bytes)

    def __setstate__(self, _state: object) -> NoReturn:
        """Reject state restoration that could forge derived values."""

        raise TypeError(f"{type(self).__name__} is immutable")

    @property
    def element_count(self) -> int:
        """Return the checked ordered product of the declared shape."""

        return self._element_count

    @property
    def nbytes(self) -> int:
        """Return the exact byte width proved for this tensor."""

        return self._nbytes


@dataclass(frozen=True, slots=True)
class SafetensorsManifest:
    """A schema- and storage-validated safetensors v0.8 manifest."""

    plan: HeaderReadPlan
    tensors: tuple[TensorManifest, ...] = field(repr=False)
    metadata: tuple[tuple[str, str], ...] = field(repr=False)
    metadata_form: MetadataForm
    notices: tuple[HeaderNotice, ...]

    def __post_init__(self) -> None:
        if type(self.plan) is not HeaderReadPlan:
            raise TypeError("safetensors manifest plan must be HeaderReadPlan")
        if type(self.tensors) is not tuple or not all(
            type(tensor) is TensorManifest for tensor in self.tensors
        ):
            raise TypeError("manifest tensors must be a tuple of TensorManifest values")

        rebuilt_tensors = tuple(
            TensorManifest(
                name=tensor.name,
                dtype=tensor.dtype,
                shape=tensor.shape,
                data_offsets=tensor.data_offsets,
                unknown_fields=tensor.unknown_fields,
            )
            for tensor in self.tensors
        )
        if any(
            type(tensor.element_count) is not int
            or tensor.element_count != rebuilt.element_count
            or type(tensor.nbytes) is not int
            or tensor.nbytes != rebuilt.nbytes
            for tensor, rebuilt in zip(self.tensors, rebuilt_tensors, strict=True)
        ):
            raise ValueError("manifest tensor derived state is inconsistent")
        object.__setattr__(self, "plan", _validated_plan_copy(self.plan))
        object.__setattr__(self, "tensors", rebuilt_tensors)

        decoded_tensors = tuple(
            sorted(
                (
                    TensorHeader(
                        name=tensor.name,
                        dtype=tensor.dtype.value,
                        shape=tensor.shape,
                        data_offsets=tensor.data_offsets,
                        unknown_fields=tensor.unknown_fields,
                    )
                    for tensor in self.tensors
                ),
                key=lambda tensor: tensor.name,
            )
        )
        DecodedSafetensorsHeader(
            plan=self.plan,
            tensors=decoded_tensors,
            metadata=self.metadata,
            metadata_form=self.metadata_form,
            notices=self.notices,
        )

        canonical = tuple(sorted(self.tensors, key=_manifest_storage_key))
        if self.tensors != canonical:
            raise ValueError("manifest tensors must use canonical storage order")
        cursor = 0
        for tensor in self.tensors:
            begin, end = tensor.data_offsets
            if begin != cursor:
                raise ValueError("manifest tensor layout must be hole-free")
            cursor = end
        if cursor != self.plan.data_size:
            raise ValueError("manifest tensor layout must cover the payload")

    def __setstate__(self, _state: object) -> NoReturn:
        """Reject state restoration that could bypass validation."""

        raise TypeError(f"{type(self).__name__} is immutable")


def _validated_limits_copy(limits: SafetensorsLimits) -> SafetensorsLimits:
    return SafetensorsLimits(
        max_header_bytes=limits.max_header_bytes,
        max_json_depth=limits.max_json_depth,
        max_json_string_chars=limits.max_json_string_chars,
        max_json_tokens=limits.max_json_tokens,
        max_tensors=limits.max_tensors,
        max_tensor_rank=limits.max_tensor_rank,
        max_tensor_name_bytes=limits.max_tensor_name_bytes,
        max_metadata_entries=limits.max_metadata_entries,
    )


def _validated_plan_copy(plan: HeaderReadPlan) -> HeaderReadPlan:
    return HeaderReadPlan(
        file_size=plan.file_size,
        header_size=plan.header_size,
        header_offset=plan.header_offset,
        data_offset=plan.data_offset,
        data_size=plan.data_size,
        limits=plan.limits,
    )


def _validated_envelope_copy(envelope: HeaderEnvelope) -> HeaderEnvelope:
    return HeaderEnvelope(plan=envelope.plan, header=envelope.header)


def _validated_tensor_header_copy(tensor: TensorHeader) -> TensorHeader:
    return TensorHeader(
        name=tensor.name,
        dtype=tensor.dtype,
        shape=tensor.shape,
        data_offsets=tensor.data_offsets,
        unknown_fields=tensor.unknown_fields,
    )


def _validated_decoded_copy(
    decoded: DecodedSafetensorsHeader,
) -> DecodedSafetensorsHeader:
    return DecodedSafetensorsHeader(
        plan=decoded.plan,
        tensors=decoded.tensors,
        metadata=decoded.metadata,
        metadata_form=decoded.metadata_form,
        notices=decoded.notices,
    )


def plan_header_read(
    prefix: bytes,
    *,
    file_size: int,
    limits: SafetensorsLimits = DEFAULT_SAFETENSORS_LIMITS,
) -> HeaderReadPlan:
    """Validate a safetensors prefix and return the bounded header range.

    The caller is expected to obtain ``prefix`` independently using an exact,
    bounded eight-byte read. No header or tensor data is accepted by this
    function.
    """

    if type(prefix) is not bytes:
        raise TypeError("prefix must be bytes")
    if type(file_size) is not int:
        raise TypeError("file_size must be an integer")
    if file_size < 0:
        raise ValueError("file_size must not be negative")
    if type(limits) is not SafetensorsLimits:
        raise TypeError("limits must be SafetensorsLimits")
    limits = _validated_limits_copy(limits)
    if file_size < 8:
        raise InvalidSafetensors(SafetensorsErrorCode.FILE_TOO_SMALL)
    if len(prefix) != 8:
        raise SafetensorsReadMismatch(SafetensorsErrorCode.PREFIX_LENGTH)

    header_size = int.from_bytes(prefix, byteorder="little", signed=False)
    if header_size > SAFETENSORS_HEADER_LIMIT:
        raise InvalidSafetensors(SafetensorsErrorCode.HEADER_EXCEEDS_FORMAT_LIMIT)
    available_after_prefix = file_size - 8
    if header_size > available_after_prefix:
        raise InvalidSafetensors(SafetensorsErrorCode.HEADER_OUT_OF_BOUNDS)
    if header_size > limits.max_header_bytes:
        raise SafetensorsLimitExceeded(
            SafetensorsErrorCode.HEADER_EXCEEDS_POLICY_LIMIT,
            limit=limits.max_header_bytes,
        )

    data_offset = 8 + header_size
    return HeaderReadPlan(
        file_size=file_size,
        header_size=header_size,
        header_offset=8,
        data_offset=data_offset,
        data_size=file_size - data_offset,
        limits=limits,
    )


def accept_header(plan: HeaderReadPlan, header: bytes) -> HeaderEnvelope:
    """Bind exactly the planned immutable header bytes to an envelope.

    A source adapter must establish that the prefix, size, and header came from
    one unchanged object before treating an exception as artifact evidence.
    """

    if type(plan) is not HeaderReadPlan:
        raise TypeError("plan must be HeaderReadPlan")
    return HeaderEnvelope(plan=plan, header=header)


def decode_header(envelope: HeaderEnvelope) -> DecodedSafetensorsHeader:
    """Decode UTF-8 JSON and the safetensors tensor-entry schema.

    This stage does not recognize dtype tokens, compute byte widths, compare
    shapes with spans, or prove a hole-free payload layout. Success therefore
    means ``schema decoded``, not ``valid safetensors storage``.
    """

    if type(envelope) is not HeaderEnvelope:
        raise TypeError("envelope must be HeaderEnvelope")
    envelope = _validated_envelope_copy(envelope)

    limits = envelope.plan.limits
    text: str | None = None
    try:
        text = envelope.header.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        pass
    if text is None:
        raise InvalidSafetensors(SafetensorsErrorCode.HEADER_UTF8)

    json_error: BoundedJsonError | None = None
    root: object | None = None
    try:
        root = decode_json(
            text,
            JsonLimits(
                max_document_chars=limits.max_header_bytes,
                max_depth=limits.max_json_depth,
                max_string_chars=limits.max_json_string_chars,
                max_tokens=limits.max_json_tokens,
            ),
        )
    except BoundedJsonError as error:
        json_error = error
    if json_error is not None:
        _raise_safetensors_json_error(json_error)
    if type(root) is not dict:
        raise InvalidSafetensors(SafetensorsErrorCode.HEADER_NOT_OBJECT)
    members = cast(dict[str, object], root)

    metadata, metadata_form, metadata_notices = _decode_metadata(members, limits)
    tensor_members = tuple(
        (name, value) for name, value in members.items() if name != "__metadata__"
    )
    if len(tensor_members) > limits.max_tensors:
        raise SafetensorsLimitExceeded(
            SafetensorsErrorCode.TENSOR_COUNT_EXCEEDS_POLICY_LIMIT,
            limit=limits.max_tensors,
        )

    tensors: list[TensorHeader] = []
    notices = set(metadata_notices)
    for name, value in tensor_members:
        if _has_lone_surrogate(name):
            raise InvalidSafetensors(SafetensorsErrorCode.INVALID_UNICODE_SCALAR)
        if len(name.encode("utf-8")) > limits.max_tensor_name_bytes:
            raise SafetensorsLimitExceeded(
                SafetensorsErrorCode.TENSOR_NAME_EXCEEDS_POLICY_LIMIT,
                limit=limits.max_tensor_name_bytes,
            )
        tensor = _decode_tensor(name, value, limits)
        if tensor.unknown_fields:
            notices.add(HeaderNotice.UNKNOWN_TENSOR_FIELDS)
        tensors.append(tensor)

    return DecodedSafetensorsHeader(
        plan=envelope.plan,
        tensors=tuple(sorted(tensors, key=lambda tensor: tensor.name)),
        metadata=metadata,
        metadata_form=metadata_form,
        notices=tuple(sorted(notices, key=lambda notice: notice.value)),
    )


def validate_storage(decoded: DecodedSafetensorsHeader) -> SafetensorsManifest:
    """Prove dtype, tensor-size, layout, and payload-coverage invariants."""

    if type(decoded) is not DecodedSafetensorsHeader:
        raise TypeError("decoded must be DecodedSafetensorsHeader")
    decoded = _validated_decoded_copy(decoded)

    dtypes: dict[str, SafetensorsDtype] = {}
    for tensor in decoded.tensors:
        dtype: SafetensorsDtype | None = None
        try:
            dtype = SafetensorsDtype(tensor.dtype)
        except ValueError:
            pass
        if dtype is None:
            raise InvalidSafetensors(SafetensorsErrorCode.INVALID_DTYPE)
        dtypes[tensor.name] = dtype

    ordered = tuple(
        sorted(
            decoded.tensors,
            key=lambda tensor: (tensor.data_offsets[0], tensor.data_offsets[1], tensor.name),
        )
    )
    manifests: list[TensorManifest] = []
    cursor = 0
    for tensor in ordered:
        begin, end = tensor.data_offsets
        if begin > cursor:
            raise InvalidSafetensors(SafetensorsErrorCode.TENSOR_LAYOUT_GAP)
        if begin < cursor:
            raise InvalidSafetensors(SafetensorsErrorCode.TENSOR_LAYOUT_OVERLAP)
        if end < begin:
            raise InvalidSafetensors(SafetensorsErrorCode.TENSOR_OFFSETS_REVERSED)

        dtype = dtypes[tensor.name]
        _, expected_bytes = _checked_tensor_size(tensor.shape, dtype)
        if end - begin != expected_bytes:
            raise InvalidSafetensors(SafetensorsErrorCode.TENSOR_SIZE_MISMATCH)
        manifests.append(
            TensorManifest(
                name=tensor.name,
                dtype=dtype,
                shape=tensor.shape,
                data_offsets=tensor.data_offsets,
                unknown_fields=tensor.unknown_fields,
            )
        )
        cursor = end

    if cursor != decoded.plan.data_size:
        raise InvalidSafetensors(SafetensorsErrorCode.PAYLOAD_SIZE_MISMATCH)

    return SafetensorsManifest(
        plan=decoded.plan,
        tensors=tuple(manifests),
        metadata=decoded.metadata,
        metadata_form=decoded.metadata_form,
        notices=decoded.notices,
    )


def parse_safetensors_manifest(
    prefix: bytes,
    header: bytes,
    *,
    file_size: int,
    limits: SafetensorsLimits = DEFAULT_SAFETENSORS_LIMITS,
) -> SafetensorsManifest:
    """Compose the pure envelope, schema, and storage-validation stages.

    This convenience API accepts already-acquired value evidence. A source
    adapter must still bind the prefix, file size, and header to one unchanged
    object and must use :func:`plan_header_read` before fetching the header.
    """

    plan = plan_header_read(prefix, file_size=file_size, limits=limits)
    envelope = accept_header(plan, header)
    decoded = decode_header(envelope)
    return validate_storage(decoded)


def _checked_tensor_size(
    shape: tuple[int, ...],
    dtype: SafetensorsDtype,
) -> tuple[int, int]:
    elements = 1
    for dimension in shape:
        if dimension != 0 and elements > _U64_MAX // dimension:
            raise InvalidSafetensors(SafetensorsErrorCode.TENSOR_SIZE_OVERFLOW)
        elements *= dimension

    bits = dtype.bits_per_element
    if elements != 0 and elements > _U64_MAX // bits:
        raise InvalidSafetensors(SafetensorsErrorCode.TENSOR_SIZE_OVERFLOW)
    total_bits = elements * bits
    if total_bits % 8 != 0:
        raise InvalidSafetensors(SafetensorsErrorCode.TENSOR_BYTE_MISALIGNED)
    return elements, total_bits // 8


def _manifest_storage_key(tensor: TensorManifest) -> tuple[int, int, str]:
    begin, end = tensor.data_offsets
    return begin, end, tensor.name


def _raise_safetensors_json_error(error: BoundedJsonError) -> NoReturn:
    if type(error) is InvalidJson:
        code = (
            SafetensorsErrorCode.DUPLICATE_JSON_KEY
            if error.code is BoundedJsonErrorCode.DUPLICATE_KEY
            else SafetensorsErrorCode.HEADER_JSON
        )
        raise InvalidSafetensors(code)

    if type(error) is JsonLimitExceeded:
        limit_code = {
            BoundedJsonErrorCode.DOCUMENT_LIMIT: (SafetensorsErrorCode.HEADER_EXCEEDS_POLICY_LIMIT),
            BoundedJsonErrorCode.DEPTH_LIMIT: (
                SafetensorsErrorCode.JSON_DEPTH_EXCEEDS_POLICY_LIMIT
            ),
            BoundedJsonErrorCode.STRING_LIMIT: (
                SafetensorsErrorCode.JSON_STRING_EXCEEDS_POLICY_LIMIT
            ),
            BoundedJsonErrorCode.TOKEN_LIMIT: (
                SafetensorsErrorCode.JSON_TOKEN_EXCEEDS_POLICY_LIMIT
            ),
        }.get(error.code)
        if limit_code is None:
            raise ValueError("unsupported bounded JSON limit code")
        raise SafetensorsLimitExceeded(limit_code, limit=error.limit)

    raise TypeError("unsupported bounded JSON error type")


def _decode_metadata(
    members: dict[str, object],
    limits: SafetensorsLimits,
) -> tuple[tuple[tuple[str, str], ...], MetadataForm, tuple[HeaderNotice, ...]]:
    missing = object()
    value = members.get("__metadata__", missing)
    if value is missing:
        return (), MetadataForm.ABSENT, ()
    if value is None:
        return (), MetadataForm.NULL, (HeaderNotice.METADATA_NULL,)
    if type(value) is not dict:
        raise InvalidSafetensors(SafetensorsErrorCode.INVALID_METADATA)

    metadata = cast(dict[str, object], value)
    if len(metadata) > limits.max_metadata_entries:
        raise SafetensorsLimitExceeded(
            SafetensorsErrorCode.METADATA_COUNT_EXCEEDS_POLICY_LIMIT,
            limit=limits.max_metadata_entries,
        )
    entries: list[tuple[str, str]] = []
    for key, metadata_value in metadata.items():
        if type(metadata_value) is not str:
            raise InvalidSafetensors(SafetensorsErrorCode.INVALID_METADATA)
        if _has_lone_surrogate(key) or _has_lone_surrogate(metadata_value):
            raise InvalidSafetensors(SafetensorsErrorCode.INVALID_UNICODE_SCALAR)
        entries.append((key, metadata_value))
    return tuple(sorted(entries)), MetadataForm.OBJECT, ()


def _decode_tensor(
    name: str,
    value: object,
    limits: SafetensorsLimits,
) -> TensorHeader:
    if type(value) is not dict:
        raise InvalidSafetensors(SafetensorsErrorCode.INVALID_TENSOR_RECORD)
    record = cast(dict[str, object], value)
    if any(_has_lone_surrogate(field_name) for field_name in record):
        raise InvalidSafetensors(SafetensorsErrorCode.INVALID_UNICODE_SCALAR)
    required = frozenset({"dtype", "shape", "data_offsets"})
    if not required.issubset(record):
        raise InvalidSafetensors(SafetensorsErrorCode.MISSING_TENSOR_FIELD)

    dtype = record["dtype"]
    if type(dtype) is not str:
        raise InvalidSafetensors(SafetensorsErrorCode.INVALID_TENSOR_FIELD)
    if _has_lone_surrogate(dtype):
        raise InvalidSafetensors(SafetensorsErrorCode.INVALID_UNICODE_SCALAR)
    shape = _decode_unsigned_sequence(record["shape"])
    if len(shape) > limits.max_tensor_rank:
        raise SafetensorsLimitExceeded(
            SafetensorsErrorCode.TENSOR_RANK_EXCEEDS_POLICY_LIMIT,
            limit=limits.max_tensor_rank,
        )
    offsets = _decode_unsigned_sequence(record["data_offsets"])
    if len(offsets) != 2:
        raise InvalidSafetensors(SafetensorsErrorCode.INVALID_TENSOR_FIELD)

    return TensorHeader(
        name=name,
        dtype=dtype,
        shape=shape,
        data_offsets=(offsets[0], offsets[1]),
        unknown_fields=tuple(sorted(record.keys() - required)),
    )


def _decode_unsigned_sequence(value: object) -> tuple[int, ...]:
    if type(value) is not list:
        raise InvalidSafetensors(SafetensorsErrorCode.INVALID_TENSOR_FIELD)
    return tuple(_decode_u64(member) for member in cast(list[object], value))


def _decode_u64(value: object) -> int:
    if type(value) is not IntegerLexeme:
        raise InvalidSafetensors(SafetensorsErrorCode.INVALID_TENSOR_FIELD)
    digits = value.value
    maximum = "18446744073709551615"
    if (
        not digits
        or not digits.isascii()
        or not digits.isdigit()
        or len(digits) > len(maximum)
        or (len(digits) == len(maximum) and digits > maximum)
    ):
        raise InvalidSafetensors(SafetensorsErrorCode.INTEGER_OUT_OF_RANGE)
    return int(digits)


__all__ = [
    "DEFAULT_SAFETENSORS_LIMITS",
    "SAFETENSORS_HEADER_LIMIT",
    "DecodedSafetensorsHeader",
    "HeaderEnvelope",
    "HeaderNotice",
    "HeaderReadPlan",
    "InvalidSafetensors",
    "MetadataForm",
    "SafetensorsDtype",
    "SafetensorsErrorCode",
    "SafetensorsInspectionError",
    "SafetensorsLimitExceeded",
    "SafetensorsLimits",
    "SafetensorsManifest",
    "SafetensorsReadMismatch",
    "TensorHeader",
    "TensorManifest",
    "accept_header",
    "decode_header",
    "parse_safetensors_manifest",
    "plan_header_read",
    "validate_storage",
]
