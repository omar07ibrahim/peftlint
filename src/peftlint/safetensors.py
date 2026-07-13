"""Pure, bounded primitives for inspecting safetensors headers.

This module deliberately has no file, stream, network, or memory-mapping API.
Callers must inspect the eight-byte prefix first, use :func:`plan_header_read`
to obtain a bounded range, and only then supply exactly that range to
:func:`accept_header`. Plans are value objects, not provenance attestations:
source adapters remain responsible for binding both reads to one unchanged
object and for distinguishing transport failures from malformed artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import NoReturn, cast

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
}

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

    def __init__(self, code: SafetensorsErrorCode) -> None:
        _require_error_category(code, _INVALID_ERROR_CODES, "invalid safetensors")
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
        if len(self.header) != self.plan.header_size:
            raise SafetensorsReadMismatch(SafetensorsErrorCode.HEADER_LENGTH_MISMATCH)


class MetadataForm(StrEnum):
    """How the optional ``__metadata__`` member appeared in the header."""

    ABSENT = "absent"
    NULL = "null"
    OBJECT = "object"


class HeaderNotice(StrEnum):
    """Non-fatal format details retained for deterministic reporting."""

    METADATA_NULL = "metadata_null"
    UNKNOWN_TENSOR_FIELDS = "unknown_tensor_fields"


@dataclass(frozen=True, slots=True)
class TensorHeader:
    """One schema-decoded tensor entry whose storage is not yet validated."""

    name: str = field(repr=False)
    dtype: str = field(repr=False)
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    unknown_fields: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if type(self.name) is not str or type(self.dtype) is not str:
            raise TypeError("tensor name and dtype must be strings")
        if _has_lone_surrogate(self.name) or _has_lone_surrogate(self.dtype):
            raise ValueError("tensor name and dtype must contain valid Unicode scalars")
        if self.name == "__metadata__":
            raise ValueError("__metadata__ is reserved and cannot be a tensor name")
        if type(self.shape) is not tuple or not all(type(size) is int for size in self.shape):
            raise TypeError("tensor shape must be a tuple of integers")
        if any(not 0 <= size <= _U64_MAX for size in self.shape):
            raise ValueError("tensor shape integers must fit unsigned 64-bit values")
        if (
            type(self.data_offsets) is not tuple
            or len(self.data_offsets) != 2
            or not all(type(offset) is int for offset in self.data_offsets)
        ):
            raise TypeError("tensor data_offsets must contain exactly two integers")
        if any(not 0 <= offset <= _U64_MAX for offset in self.data_offsets):
            raise ValueError("tensor offsets must fit unsigned 64-bit values")
        if type(self.unknown_fields) is not tuple or not all(
            type(name) is str for name in self.unknown_fields
        ):
            raise TypeError("unknown_fields must be a tuple of strings")
        if any(_has_lone_surrogate(name) for name in self.unknown_fields):
            raise ValueError("unknown field names must contain valid Unicode scalars")
        if self.unknown_fields != tuple(sorted(set(self.unknown_fields))):
            raise ValueError("unknown field names must be sorted and unique")
        if frozenset(self.unknown_fields) & {"dtype", "shape", "data_offsets"}:
            raise ValueError("required tensor fields cannot be unknown fields")


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


@dataclass(frozen=True, slots=True, repr=False)
class _IntegerLexeme:
    value: str


@dataclass(frozen=True, slots=True, repr=False)
class _FloatLexeme:
    value: str


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

    limits = envelope.plan.limits
    text: str | None = None
    try:
        text = envelope.header.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        pass
    if text is None:
        raise InvalidSafetensors(SafetensorsErrorCode.HEADER_UTF8)

    _prescan_json(text, limits)
    root = _load_json(text, limits)
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


def _prescan_json(text: str, limits: SafetensorsLimits) -> None:
    depth = 0
    tokens = 0
    in_string = False
    in_primitive = False
    escaped = False
    string_chars = 0

    def count_token() -> None:
        nonlocal tokens
        tokens += 1
        if tokens > limits.max_json_tokens:
            raise SafetensorsLimitExceeded(
                SafetensorsErrorCode.JSON_TOKEN_EXCEEDS_POLICY_LIMIT,
                limit=limits.max_json_tokens,
            )

    for character in text:
        if in_string:
            if character == '"' and not escaped:
                in_string = False
                continue
            string_chars += 1
            if string_chars > limits.max_json_string_chars:
                raise SafetensorsLimitExceeded(
                    SafetensorsErrorCode.JSON_STRING_EXCEEDS_POLICY_LIMIT,
                    limit=limits.max_json_string_chars,
                )
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            continue

        if character == '"':
            count_token()
            in_string = True
            in_primitive = False
            string_chars = 0
        elif character in "[{":
            count_token()
            depth += 1
            in_primitive = False
            if depth > limits.max_json_depth:
                raise SafetensorsLimitExceeded(
                    SafetensorsErrorCode.JSON_DEPTH_EXCEEDS_POLICY_LIMIT,
                    limit=limits.max_json_depth,
                )
        elif character in "]}":
            depth = max(0, depth - 1)
            in_primitive = False
        elif character == ",":
            count_token()
            in_primitive = False
        elif character == ":" or character in " \t\r\n":
            in_primitive = False
        elif not in_primitive:
            count_token()
            in_primitive = True


def _load_json(text: str, limits: SafetensorsLimits) -> object:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise InvalidSafetensors(SafetensorsErrorCode.DUPLICATE_JSON_KEY)
            result[key] = value
        return result

    def reject_constant(_value: str) -> NoReturn:
        raise InvalidSafetensors(SafetensorsErrorCode.HEADER_JSON)

    result: object | None = None
    json_failed = False
    recursion_failed = False
    try:
        result = cast(
            object,
            json.loads(
                text,
                object_pairs_hook=unique_object,
                parse_int=_IntegerLexeme,
                parse_float=_FloatLexeme,
                parse_constant=reject_constant,
            ),
        )
    except json.JSONDecodeError:
        json_failed = True
    except RecursionError:
        recursion_failed = True

    if json_failed:
        raise InvalidSafetensors(SafetensorsErrorCode.HEADER_JSON)
    if recursion_failed:
        raise SafetensorsLimitExceeded(
            SafetensorsErrorCode.JSON_DEPTH_EXCEEDS_POLICY_LIMIT,
            limit=limits.max_json_depth,
        )
    return result


def _has_lone_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


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
    if type(value) is not _IntegerLexeme:
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
    "SafetensorsErrorCode",
    "SafetensorsInspectionError",
    "SafetensorsLimitExceeded",
    "SafetensorsLimits",
    "SafetensorsReadMismatch",
    "TensorHeader",
    "accept_header",
    "decode_header",
    "plan_header_read",
]
