from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import pytest

from peftlint.safetensors import (
    SAFETENSORS_HEADER_LIMIT,
    HeaderEnvelope,
    HeaderReadPlan,
    InvalidSafetensors,
    SafetensorsErrorCode,
    SafetensorsInspectionError,
    SafetensorsLimitExceeded,
    SafetensorsLimits,
    SafetensorsReadMismatch,
    accept_header,
    plan_header_read,
)


def _prefix(size: int) -> bytes:
    return size.to_bytes(length=8, byteorder="little", signed=False)


@pytest.mark.parametrize(
    ("header_size", "file_size", "data_size"),
    [
        (0, 8, 0),
        (2, 10, 0),
        (2, 13, 3),
        (16 * 1024 * 1024, 16 * 1024 * 1024 + 9, 1),
    ],
)
def test_plan_header_read_returns_only_the_bounded_range(
    header_size: int,
    file_size: int,
    data_size: int,
) -> None:
    plan = plan_header_read(_prefix(header_size), file_size=file_size)

    assert plan == HeaderReadPlan(
        file_size=file_size,
        header_size=header_size,
        header_offset=8,
        data_offset=8 + header_size,
        data_size=data_size,
        limits=SafetensorsLimits(),
    )


def test_plan_header_read_decodes_unsigned_little_endian() -> None:
    plan = plan_header_read(b"\x01\x00\x00\x00\x00\x00\x00\x00", file_size=9)

    assert plan.header_size == 1


@pytest.mark.parametrize("prefix", [b"", b"\x00", b"\x00" * 7, b"\x00" * 9])
def test_plan_header_read_rejects_non_eight_byte_prefix(prefix: bytes) -> None:
    with pytest.raises(SafetensorsReadMismatch) as raised:
        plan_header_read(prefix, file_size=8)

    assert raised.value.code is SafetensorsErrorCode.PREFIX_LENGTH
    assert str(raised.value) == "safetensors prefix read must contain exactly 8 bytes"


@pytest.mark.parametrize("prefix", [bytearray(8), memoryview(bytes(8)), "00000000"])
def test_plan_header_read_requires_immutable_bytes(prefix: object) -> None:
    with pytest.raises(TypeError, match=r"^prefix must be bytes$"):
        plan_header_read(prefix, file_size=8)  # type: ignore[arg-type]


@pytest.mark.parametrize("file_size", [True, False, 8.0, "8"])
def test_plan_header_read_requires_an_integer_file_size(file_size: object) -> None:
    with pytest.raises(TypeError, match=r"^file_size must be an integer$"):
        plan_header_read(bytes(8), file_size=file_size)  # type: ignore[arg-type]


def test_plan_header_read_rejects_negative_file_size() -> None:
    with pytest.raises(ValueError, match=r"^file_size must not be negative$"):
        plan_header_read(bytes(8), file_size=-1)


@pytest.mark.parametrize("file_size", range(8))
def test_plan_header_read_rejects_files_smaller_than_the_prefix(file_size: int) -> None:
    with pytest.raises(InvalidSafetensors) as raised:
        plan_header_read(bytes(file_size), file_size=file_size)

    assert raised.value.code is SafetensorsErrorCode.FILE_TOO_SMALL


def test_plan_header_read_checks_format_limit_before_declared_bounds() -> None:
    with pytest.raises(InvalidSafetensors) as raised:
        plan_header_read(_prefix(2**64 - 1), file_size=8)

    assert raised.value.code is SafetensorsErrorCode.HEADER_EXCEEDS_FORMAT_LIMIT


def test_plan_header_read_checks_declared_bounds_before_policy_limit() -> None:
    with pytest.raises(InvalidSafetensors) as raised:
        plan_header_read(_prefix(3), file_size=8, limits=SafetensorsLimits(max_header_bytes=2))

    assert raised.value.code is SafetensorsErrorCode.HEADER_OUT_OF_BOUNDS


def test_plan_header_read_enforces_the_format_header_limit() -> None:
    header_size = SAFETENSORS_HEADER_LIMIT + 1

    with pytest.raises(InvalidSafetensors) as raised:
        plan_header_read(
            _prefix(header_size),
            file_size=8 + header_size,
            limits=SafetensorsLimits(max_header_bytes=SAFETENSORS_HEADER_LIMIT),
        )

    assert raised.value.code is SafetensorsErrorCode.HEADER_EXCEEDS_FORMAT_LIMIT


def test_plan_header_read_accepts_the_exact_format_header_limit_without_allocating() -> None:
    plan = plan_header_read(
        _prefix(SAFETENSORS_HEADER_LIMIT),
        file_size=8 + SAFETENSORS_HEADER_LIMIT,
        limits=SafetensorsLimits(max_header_bytes=SAFETENSORS_HEADER_LIMIT),
    )

    assert plan.header_size == SAFETENSORS_HEADER_LIMIT
    assert plan.data_size == 0


def test_plan_header_read_enforces_policy_without_accepting_header_bytes() -> None:
    with pytest.raises(SafetensorsLimitExceeded) as raised:
        plan_header_read(_prefix(3), file_size=11, limits=SafetensorsLimits(max_header_bytes=2))

    assert raised.value.code is SafetensorsErrorCode.HEADER_EXCEEDS_POLICY_LIMIT
    assert raised.value.limit == 2
    assert str(raised.value) == "declared safetensors header exceeds the inspection limit"


@pytest.mark.parametrize("max_header_bytes", [True, False, 1.0, "1"])
def test_limits_require_an_integer(max_header_bytes: object) -> None:
    with pytest.raises(TypeError, match=r"^max_header_bytes must be an integer$"):
        SafetensorsLimits(max_header_bytes=max_header_bytes)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_header_bytes", [0, -1, SAFETENSORS_HEADER_LIMIT + 1])
def test_limits_stay_inside_the_format_boundary(max_header_bytes: int) -> None:
    with pytest.raises(
        ValueError,
        match=r"^max_header_bytes must be between 1 and the safetensors format limit$",
    ):
        SafetensorsLimits(max_header_bytes=max_header_bytes)


def test_plan_header_read_requires_its_limits_type() -> None:
    with pytest.raises(TypeError, match=r"^limits must be SafetensorsLimits$"):
        plan_header_read(bytes(8), file_size=8, limits=object())  # type: ignore[arg-type]


def test_accept_header_binds_exactly_the_planned_bytes() -> None:
    plan = plan_header_read(_prefix(2), file_size=13)

    envelope = accept_header(plan, b"{}")

    assert envelope == HeaderEnvelope(plan=plan, header=b"{}")
    assert envelope.header == b"{}"
    assert "{}" not in repr(envelope)


def test_header_envelope_repr_redacts_printable_header_content() -> None:
    header = b"private-marker"
    plan = plan_header_read(_prefix(len(header)), file_size=8 + len(header))

    assert "private-marker" not in repr(accept_header(plan, header))


@pytest.mark.parametrize("header", [b"", b"{", b"{} "])
def test_accept_header_rejects_a_length_mismatch(header: bytes) -> None:
    plan = plan_header_read(_prefix(2), file_size=10)

    with pytest.raises(SafetensorsReadMismatch) as raised:
        accept_header(plan, header)

    assert raised.value.code is SafetensorsErrorCode.HEADER_LENGTH_MISMATCH


@pytest.mark.parametrize("header", [bytearray(b"{}"), memoryview(b"{}"), "{}"])
def test_accept_header_requires_immutable_bytes(header: object) -> None:
    plan = plan_header_read(_prefix(2), file_size=10)

    with pytest.raises(TypeError, match=r"^header must be bytes$"):
        accept_header(plan, header)  # type: ignore[arg-type]


def test_accept_header_requires_a_read_plan() -> None:
    with pytest.raises(TypeError, match=r"^plan must be HeaderReadPlan$"):
        accept_header(object(), b"")  # type: ignore[arg-type]


def test_public_envelope_values_are_immutable() -> None:
    plan = plan_header_read(_prefix(2), file_size=10)
    envelope = accept_header(plan, b"{}")

    with pytest.raises(FrozenInstanceError):
        plan.header_size = 3  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        envelope.header = b"[]"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        (
            {"file_size": True},
            TypeError,
            "header read plan sizes must be integers",
        ),
        (
            {"limits": object()},
            TypeError,
            "header read plan limits must be SafetensorsLimits",
        ),
        (
            {"file_size": 7, "data_size": -3},
            ValueError,
            "header read plan file_size must be at least 8",
        ),
        (
            {"header_size": 3, "data_offset": 11, "data_size": -1},
            ValueError,
            "header read plan exceeds its inspection limit",
        ),
        (
            {"header_size": -1, "data_offset": 7, "data_size": 3},
            ValueError,
            "header read plan exceeds its inspection limit",
        ),
        (
            {"header_offset": 7},
            ValueError,
            "header read plan header_offset must be 8",
        ),
        (
            {"data_offset": 9},
            ValueError,
            "header read plan data_offset is inconsistent",
        ),
        (
            {"data_size": -1},
            ValueError,
            "header read plan data_size is inconsistent",
        ),
        (
            {"data_size": 1},
            ValueError,
            "header read plan data_size is inconsistent",
        ),
    ],
)
def test_header_read_plan_validates_manual_construction(
    overrides: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    values: dict[str, object] = {
        "file_size": 10,
        "header_size": 2,
        "header_offset": 8,
        "data_offset": 10,
        "data_size": 0,
        "limits": SafetensorsLimits(max_header_bytes=2),
    }
    values.update(overrides)

    with pytest.raises(error_type, match=f"^{message}$"):
        HeaderReadPlan(**values)  # type: ignore[arg-type]


def test_header_envelope_rejects_a_forged_plan_type() -> None:
    with pytest.raises(TypeError, match=r"^plan must be HeaderReadPlan$"):
        HeaderEnvelope(plan=object(), header=b"")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("error_type", "code", "category"),
    [
        (
            InvalidSafetensors,
            SafetensorsErrorCode.HEADER_EXCEEDS_POLICY_LIMIT,
            "invalid safetensors",
        ),
        (
            SafetensorsReadMismatch,
            SafetensorsErrorCode.FILE_TOO_SMALL,
            "read mismatch",
        ),
    ],
)
def test_error_subclasses_reject_codes_from_other_categories(
    error_type: type[SafetensorsInspectionError],
    code: SafetensorsErrorCode,
    category: str,
) -> None:
    with pytest.raises(ValueError, match=f"^{code.value} is not an {category} error code$"):
        error_type(code)


def test_limit_error_rejects_a_non_limit_code() -> None:
    with pytest.raises(
        ValueError,
        match=r"^prefix_length is not an inspection limit error code$",
    ):
        SafetensorsLimitExceeded(SafetensorsErrorCode.PREFIX_LENGTH, limit=1)


@pytest.mark.parametrize("limit", [True, 1.0, "1"])
def test_limit_error_requires_an_integer_limit(limit: object) -> None:
    with pytest.raises(TypeError, match=r"^limit must be an integer$"):
        SafetensorsLimitExceeded(
            SafetensorsErrorCode.HEADER_EXCEEDS_POLICY_LIMIT,
            limit=limit,  # type: ignore[arg-type]
        )


def test_limit_error_requires_a_positive_limit() -> None:
    with pytest.raises(ValueError, match=r"^limit must be positive$"):
        SafetensorsLimitExceeded(
            SafetensorsErrorCode.HEADER_EXCEEDS_POLICY_LIMIT,
            limit=0,
        )


def test_error_base_class_cannot_be_instantiated_directly() -> None:
    with pytest.raises(
        TypeError,
        match=r"^SafetensorsInspectionError is an abstract base class$",
    ):
        SafetensorsInspectionError(SafetensorsErrorCode.FILE_TOO_SMALL)


def test_error_codes_require_the_enum_type() -> None:
    with pytest.raises(
        TypeError,
        match=r"^safetensors error code must be SafetensorsErrorCode$",
    ):
        InvalidSafetensors("file_too_small")  # type: ignore[arg-type]


def test_envelope_api_cannot_receive_sources_or_payloads() -> None:
    assert tuple(inspect.signature(plan_header_read).parameters) == (
        "prefix",
        "file_size",
        "limits",
    )
    assert tuple(inspect.signature(accept_header).parameters) == ("plan", "header")

    with pytest.raises(TypeError, match="unexpected keyword argument 'payload'"):
        plan_header_read(bytes(8), file_size=8, payload=object())  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="unexpected keyword argument 'stream'"):
        plan_header_read(bytes(8), file_size=8, stream=object())  # type: ignore[call-arg]


def test_exception_text_never_contains_untrusted_header_bytes() -> None:
    marker = b"never-copy-this-value"
    plan = plan_header_read(_prefix(2), file_size=10)

    with pytest.raises(SafetensorsReadMismatch) as raised:
        accept_header(plan, marker)

    rendered = f"{raised.value!s} {raised.value!r}"
    assert marker.decode() not in rendered
