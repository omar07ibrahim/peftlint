"""Pure, bounded primitives for inspecting safetensors headers.

This module deliberately has no file, stream, network, or memory-mapping API.
Callers must inspect the eight-byte prefix first, use :func:`plan_header_read`
to obtain a bounded range, and only then supply exactly that range to
:func:`accept_header`. Plans are value objects, not provenance attestations:
source adapters remain responsible for binding both reads to one unchanged
object and for distinguishing transport failures from malformed artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

SAFETENSORS_HEADER_LIMIT = 100_000_000
"""Maximum header length accepted by the pinned safetensors v0.8 format."""


class SafetensorsErrorCode(StrEnum):
    """Stable machine-readable failures produced by header inspection."""

    PREFIX_LENGTH = "prefix_length"
    FILE_TOO_SMALL = "file_too_small"
    HEADER_OUT_OF_BOUNDS = "header_out_of_bounds"
    HEADER_EXCEEDS_FORMAT_LIMIT = "header_exceeds_format_limit"
    HEADER_EXCEEDS_POLICY_LIMIT = "header_exceeds_policy_limit"
    HEADER_LENGTH_MISMATCH = "header_length_mismatch"


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
}

_INVALID_ERROR_CODES = frozenset(
    {
        SafetensorsErrorCode.FILE_TOO_SMALL,
        SafetensorsErrorCode.HEADER_OUT_OF_BOUNDS,
        SafetensorsErrorCode.HEADER_EXCEEDS_FORMAT_LIMIT,
    }
)
_READ_MISMATCH_ERROR_CODES = frozenset(
    {
        SafetensorsErrorCode.PREFIX_LENGTH,
        SafetensorsErrorCode.HEADER_LENGTH_MISMATCH,
    }
)
_LIMIT_ERROR_CODES = frozenset({SafetensorsErrorCode.HEADER_EXCEEDS_POLICY_LIMIT})


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
        if limit < 1:
            raise ValueError("limit must be positive")
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

    def __post_init__(self) -> None:
        if type(self.max_header_bytes) is not int:
            raise TypeError("max_header_bytes must be an integer")
        if not 1 <= self.max_header_bytes <= SAFETENSORS_HEADER_LIMIT:
            raise ValueError("max_header_bytes must be between 1 and the safetensors format limit")


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


__all__ = [
    "DEFAULT_SAFETENSORS_LIMITS",
    "SAFETENSORS_HEADER_LIMIT",
    "HeaderEnvelope",
    "HeaderReadPlan",
    "InvalidSafetensors",
    "SafetensorsErrorCode",
    "SafetensorsInspectionError",
    "SafetensorsLimitExceeded",
    "SafetensorsLimits",
    "SafetensorsReadMismatch",
    "accept_header",
    "plan_header_read",
]
