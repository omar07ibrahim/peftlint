from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import pytest

from peftlint.safetensors import (
    DEFAULT_SAFETENSORS_LIMITS,
    SAFETENSORS_HEADER_LIMIT,
    DecodedSafetensorsHeader,
    HeaderEnvelope,
    HeaderNotice,
    HeaderReadPlan,
    InvalidSafetensors,
    MetadataForm,
    SafetensorsErrorCode,
    SafetensorsInspectionError,
    SafetensorsLimitExceeded,
    SafetensorsLimits,
    SafetensorsReadMismatch,
    TensorHeader,
    accept_header,
    decode_header,
    plan_header_read,
)


def _prefix(size: int) -> bytes:
    return size.to_bytes(length=8, byteorder="little", signed=False)


def _decode_bytes(
    header: bytes,
    *,
    data_size: int = 0,
    limits: SafetensorsLimits = DEFAULT_SAFETENSORS_LIMITS,
) -> DecodedSafetensorsHeader:
    plan = plan_header_read(
        _prefix(len(header)),
        file_size=8 + len(header) + data_size,
        limits=limits,
    )
    return decode_header(accept_header(plan, header))


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


def test_limit_error_requires_a_nonnegative_limit() -> None:
    with pytest.raises(ValueError, match=r"^limit must not be negative$"):
        SafetensorsLimitExceeded(
            SafetensorsErrorCode.HEADER_EXCEEDS_POLICY_LIMIT,
            limit=-1,
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


def test_decode_header_normalizes_member_order_without_retaining_raw_bytes() -> None:
    header = (
        b'{"z":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]},'
        b'"__metadata__":{"z":"last","a":"first"},'
        b'"a":{"dtype":"BF16","shape":[],"data_offsets":[12,14]}}'
    )

    decoded = _decode_bytes(header, data_size=14)

    assert decoded.metadata == (("a", "first"), ("z", "last"))
    assert decoded.metadata_form is MetadataForm.OBJECT
    assert decoded.notices == ()
    assert decoded.tensors == (
        TensorHeader(name="a", dtype="BF16", shape=(), data_offsets=(12, 14)),
        TensorHeader(name="z", dtype="F16", shape=(2, 3), data_offsets=(0, 12)),
    )
    assert not hasattr(decoded, "header")


def test_decode_header_accepts_ascii_json_whitespace_and_writer_padding() -> None:
    decoded = _decode_bytes(b" \t\r\n{}    \n")

    assert decoded.tensors == ()
    assert decoded.metadata_form is MetadataForm.ABSENT


def test_json_prescan_ignores_escaped_quotes_backslashes_and_brackets() -> None:
    header = b'{"__metadata__":{"k":"quote: \\" and slash: \\\\ and [[["}}'

    decoded = _decode_bytes(header, limits=SafetensorsLimits(max_json_depth=2))

    assert decoded.metadata == (("k", 'quote: " and slash: \\ and [[['),)


@pytest.mark.parametrize(
    ("header", "form", "notices"),
    [
        (b"{}", MetadataForm.ABSENT, ()),
        (
            b'{"__metadata__":null}',
            MetadataForm.NULL,
            (HeaderNotice.METADATA_NULL,),
        ),
        (b'{"__metadata__":{}}', MetadataForm.OBJECT, ()),
    ],
)
def test_decode_header_accepts_all_v08_metadata_forms(
    header: bytes,
    form: MetadataForm,
    notices: tuple[HeaderNotice, ...],
) -> None:
    decoded = _decode_bytes(header)

    assert decoded.metadata == ()
    assert decoded.metadata_form is form
    assert decoded.notices == notices


@pytest.mark.parametrize(
    "header",
    [
        b'{"__metadata__":[]}',
        b'{"__metadata__":1}',
        b'{"__metadata__":{"key":1}}',
        b'{"__metadata__":{"key":null}}',
    ],
)
def test_decode_header_rejects_invalid_metadata(header: bytes) -> None:
    with pytest.raises(InvalidSafetensors) as raised:
        _decode_bytes(header)

    assert raised.value.code is SafetensorsErrorCode.INVALID_METADATA


def test_metadata_does_not_consume_the_tensor_count_budget() -> None:
    limits = SafetensorsLimits(max_tensors=0)

    decoded = _decode_bytes(b'{"__metadata__":{"key":"value"}}', limits=limits)

    assert decoded.metadata == (("key", "value"),)
    assert decoded.tensors == ()


def test_decode_header_enforces_the_metadata_entry_limit() -> None:
    limits = SafetensorsLimits(max_metadata_entries=0)

    with pytest.raises(SafetensorsLimitExceeded) as raised:
        _decode_bytes(b'{"__metadata__":{"key":"value"}}', limits=limits)

    assert raised.value.code is SafetensorsErrorCode.METADATA_COUNT_EXCEEDS_POLICY_LIMIT
    assert raised.value.limit == 0


def test_decode_header_accepts_the_exact_metadata_entry_limit() -> None:
    decoded = _decode_bytes(
        b'{"__metadata__":{"key":"value"}}',
        limits=SafetensorsLimits(max_metadata_entries=1),
    )

    assert decoded.metadata == (("key", "value"),)


@pytest.mark.parametrize(
    "header",
    [
        b'{"x":{"dtype":"F16","shape":[],"data_offsets":[0,0]},'
        b'"\\u0078":{"dtype":"F16","shape":[],"data_offsets":[0,0]}}',
        b'{"x":{"dtype":"F16","shape":[],"sh\\u0061pe":[],"data_offsets":[0,0]}}',
        b'{"__metadata__":{"key":"a","\\u006bey":"b"}}',
        b'{"x":{"dtype":"F16","shape":[],"data_offsets":[0,0],"extra":{"key":1,"\\u006bey":2}}}',
        b'{"\xf0\x9f\x98\x80":{"dtype":"F16","shape":[],"data_offsets":[0,0]},'
        b'"\\ud83d\\ude00":{"dtype":"F16","shape":[],"data_offsets":[0,0]}}',
    ],
)
def test_decode_header_rejects_duplicate_decoded_keys_at_every_depth(header: bytes) -> None:
    with pytest.raises(InvalidSafetensors) as raised:
        _decode_bytes(header)

    assert raised.value.code is SafetensorsErrorCode.DUPLICATE_JSON_KEY


def test_decode_header_does_not_normalize_unicode_names() -> None:
    header = (
        b'{"\xc3\xa9":{"dtype":"F16","shape":[],"data_offsets":[0,0]},'
        b'"e\\u0301":{"dtype":"F16","shape":[],"data_offsets":[0,0]}}'
    )

    decoded = _decode_bytes(header)

    assert {tensor.name for tensor in decoded.tensors} == {
        "\N{LATIN SMALL LETTER E WITH ACUTE}",
        "e\u0301",
    }


@pytest.mark.parametrize(
    ("header", "code"),
    [
        (b"\xff", SafetensorsErrorCode.HEADER_UTF8),
        (b"\xff\xfe{\x00}\x00", SafetensorsErrorCode.HEADER_UTF8),
        (b"\x00{\x00}", SafetensorsErrorCode.HEADER_JSON),
        (b"\x00\x00\x00{\x00\x00\x00}", SafetensorsErrorCode.HEADER_JSON),
        (b"\xef\xbb\xbf{}", SafetensorsErrorCode.HEADER_JSON),
        (b"{", SafetensorsErrorCode.HEADER_JSON),
        (b"{} trailing", SafetensorsErrorCode.HEADER_JSON),
        (b"{/* comment */}", SafetensorsErrorCode.HEADER_JSON),
        (b'{"x":1,}', SafetensorsErrorCode.HEADER_JSON),
        (b"{}\xc2\xa0", SafetensorsErrorCode.HEADER_JSON),
        (b'{"x":"raw\x00control"}', SafetensorsErrorCode.HEADER_JSON),
    ],
)
def test_decode_header_rejects_invalid_encoding_or_json(
    header: bytes,
    code: SafetensorsErrorCode,
) -> None:
    with pytest.raises(InvalidSafetensors) as raised:
        _decode_bytes(header)

    assert raised.value.code is code


@pytest.mark.parametrize("header", [b"[]", b"null", b'"text"', b"1", b"true"])
def test_decode_header_requires_an_object_root(header: bytes) -> None:
    with pytest.raises(InvalidSafetensors) as raised:
        _decode_bytes(header)

    assert raised.value.code is SafetensorsErrorCode.HEADER_NOT_OBJECT


@pytest.mark.parametrize(
    "constant",
    [b"NaN", b"Infinity", b"-Infinity"],
)
def test_decode_header_rejects_non_json_numeric_constants_even_when_ignored(
    constant: bytes,
) -> None:
    header = b'{"x":{"dtype":"F16","shape":[],"data_offsets":[0,0],"ignored":' + constant + b"}}"

    with pytest.raises(InvalidSafetensors) as raised:
        _decode_bytes(header)

    assert raised.value.code is SafetensorsErrorCode.HEADER_JSON


def test_decode_header_ignores_finite_and_huge_numeric_extensions_without_conversion() -> None:
    huge_integer = b"9" * 5000
    header = (
        b'{"x":{"dtype":"F16","shape":[],"data_offsets":[0,0],'
        b'"float":1e400,"integer":' + huge_integer + b"}}"
    )

    decoded = _decode_bytes(header)

    assert decoded.tensors[0].unknown_fields == ("float", "integer")
    assert decoded.notices == (HeaderNotice.UNKNOWN_TENSOR_FIELDS,)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        (b"shape", b"true", SafetensorsErrorCode.INVALID_TENSOR_FIELD),
        (b"shape", b"1.0", SafetensorsErrorCode.INVALID_TENSOR_FIELD),
        (b"shape", b"1e0", SafetensorsErrorCode.INVALID_TENSOR_FIELD),
        (b"shape", b"-1", SafetensorsErrorCode.INTEGER_OUT_OF_RANGE),
        (b"shape", b"-0", SafetensorsErrorCode.INTEGER_OUT_OF_RANGE),
        (
            b"shape",
            b"18446744073709551616",
            SafetensorsErrorCode.INTEGER_OUT_OF_RANGE,
        ),
        (b"data_offsets", b"true", SafetensorsErrorCode.INVALID_TENSOR_FIELD),
        (b"data_offsets", b"1.0", SafetensorsErrorCode.INVALID_TENSOR_FIELD),
        (b"data_offsets", b"-1", SafetensorsErrorCode.INTEGER_OUT_OF_RANGE),
    ],
)
def test_decode_header_rejects_non_u64_shape_and_offset_values(
    field: bytes,
    value: bytes,
    code: SafetensorsErrorCode,
) -> None:
    shape = b"[" + value + b"]" if field == b"shape" else b"[]"
    offsets = b"[0," + value + b"]" if field == b"data_offsets" else b"[0,0]"
    header = b'{"x":{"dtype":"F16","shape":' + shape + b',"data_offsets":' + offsets + b"}}"

    with pytest.raises(InvalidSafetensors) as raised:
        _decode_bytes(header)

    assert raised.value.code is code


def test_decode_header_accepts_scalar_and_maximum_u64_structurally() -> None:
    header = b'{"x":{"dtype":"future","shape":[],"data_offsets":[0,18446744073709551615]}}'

    decoded = _decode_bytes(header)

    assert decoded.tensors[0].shape == ()
    assert decoded.tensors[0].data_offsets == (0, 2**64 - 1)


@pytest.mark.parametrize(
    ("header", "code"),
    [
        (b'{"x":null}', SafetensorsErrorCode.INVALID_TENSOR_RECORD),
        (b'{"x":{}}', SafetensorsErrorCode.MISSING_TENSOR_FIELD),
        (
            b'{"x":{"dtype":"F16","shape":[],"data_offsets":[0,0]}}',
            SafetensorsErrorCode.MISSING_TENSOR_FIELD,
        ),
        (
            b'{"x":{"dtype":1,"shape":[],"data_offsets":[0,0]}}',
            SafetensorsErrorCode.INVALID_TENSOR_FIELD,
        ),
        (
            b'{"x":{"dtype":null,"shape":[],"data_offsets":[0,0]}}',
            SafetensorsErrorCode.INVALID_TENSOR_FIELD,
        ),
        (
            b'{"x":{"dtype":"F16","shape":{},"data_offsets":[0,0]}}',
            SafetensorsErrorCode.INVALID_TENSOR_FIELD,
        ),
        (
            b'{"x":{"dtype":"F16","shape":"[]","data_offsets":[0,0]}}',
            SafetensorsErrorCode.INVALID_TENSOR_FIELD,
        ),
        (
            b'{"x":{"dtype":"F16","shape":[],"data_offsets":null}}',
            SafetensorsErrorCode.INVALID_TENSOR_FIELD,
        ),
        (
            b'{"x":{"dtype":"F16","shape":[],"data_offsets":[0]}}',
            SafetensorsErrorCode.INVALID_TENSOR_FIELD,
        ),
        (
            b'{"x":{"dtype":"F16","shape":[],"data_offsets":[0,0,0]}}',
            SafetensorsErrorCode.INVALID_TENSOR_FIELD,
        ),
    ],
)
def test_decode_header_rejects_invalid_tensor_schema(
    header: bytes,
    code: SafetensorsErrorCode,
) -> None:
    if code is SafetensorsErrorCode.MISSING_TENSOR_FIELD and b'"dtype"' in header:
        header = header.replace(b'"dtype":"F16",', b"")

    with pytest.raises(InvalidSafetensors) as raised:
        _decode_bytes(header)

    assert raised.value.code is code


def test_decode_header_applies_tensor_count_limit_after_metadata() -> None:
    header = (
        b'{"__metadata__":{},"a":{"dtype":"F16","shape":[],"data_offsets":[0,0]},'
        b'"b":{"dtype":"F16","shape":[],"data_offsets":[0,0]}}'
    )

    with pytest.raises(SafetensorsLimitExceeded) as raised:
        _decode_bytes(header, limits=SafetensorsLimits(max_tensors=1))

    assert raised.value.code is SafetensorsErrorCode.TENSOR_COUNT_EXCEEDS_POLICY_LIMIT
    assert raised.value.limit == 1


def test_decode_header_accepts_exact_tensor_count_limit() -> None:
    header = b'{"x":{"dtype":"F16","shape":[],"data_offsets":[0,0]}}'

    assert len(_decode_bytes(header, limits=SafetensorsLimits(max_tensors=1)).tensors) == 1


def test_decode_header_enforces_rank_after_accepting_the_exact_limit() -> None:
    accepted = b'{"x":{"dtype":"F16","shape":[1,1],"data_offsets":[0,0]}}'
    rejected = b'{"x":{"dtype":"F16","shape":[1,1,1],"data_offsets":[0,0]}}'
    limits = SafetensorsLimits(max_tensor_rank=2)

    assert _decode_bytes(accepted, limits=limits).tensors[0].shape == (1, 1)
    with pytest.raises(SafetensorsLimitExceeded) as raised:
        _decode_bytes(rejected, limits=limits)

    assert raised.value.code is SafetensorsErrorCode.TENSOR_RANK_EXCEEDS_POLICY_LIMIT
    assert raised.value.limit == 2


def test_decode_header_allows_scalar_when_rank_limit_is_zero() -> None:
    header = b'{"x":{"dtype":"F16","shape":[],"data_offsets":[0,0]}}'

    decoded = _decode_bytes(header, limits=SafetensorsLimits(max_tensor_rank=0))

    assert decoded.tensors[0].shape == ()


def test_decode_header_measures_tensor_names_in_decoded_utf8_bytes() -> None:
    accepted = b'{"\\u00e9":{"dtype":"F16","shape":[],"data_offsets":[0,0]}}'
    rejected = b'{"\xe2\x82\xac":{"dtype":"F16","shape":[],"data_offsets":[0,0]}}'
    limits = SafetensorsLimits(max_tensor_name_bytes=2)

    assert (
        _decode_bytes(accepted, limits=limits).tensors[0].name
        == "\N{LATIN SMALL LETTER E WITH ACUTE}"
    )
    with pytest.raises(SafetensorsLimitExceeded) as raised:
        _decode_bytes(rejected, limits=limits)

    assert raised.value.code is SafetensorsErrorCode.TENSOR_NAME_EXCEEDS_POLICY_LIMIT
    assert raised.value.limit == 2


def test_decode_header_allows_empty_name_when_name_limit_is_zero() -> None:
    header = b'{"":{"dtype":"F16","shape":[],"data_offsets":[0,0]}}'

    decoded = _decode_bytes(header, limits=SafetensorsLimits(max_tensor_name_bytes=0))

    assert decoded.tensors[0].name == ""


def test_decode_header_enforces_depth_without_counting_brackets_in_strings() -> None:
    accepted = b'{"x":{"dtype":"F16","shape":[],"data_offsets":[0,0],"note":"[[[ not structure"}}'
    rejected = b'{"x":{"dtype":"F16","shape":[],"data_offsets":[0,0],"extra":[[]]}}'
    limits = SafetensorsLimits(max_json_depth=3)

    assert _decode_bytes(accepted, limits=limits).tensors[0].unknown_fields == ("note",)
    with pytest.raises(SafetensorsLimitExceeded) as raised:
        _decode_bytes(rejected, limits=limits)

    assert raised.value.code is SafetensorsErrorCode.JSON_DEPTH_EXCEEDS_POLICY_LIMIT
    assert raised.value.limit == 3


def test_decode_header_enforces_raw_json_string_limit_at_the_boundary() -> None:
    accepted = b'{"__metadata__":{"k":"123456789012"}}'
    rejected = b'{"__metadata__":{"k":"1234567890123"}}'
    limits = SafetensorsLimits(max_json_string_chars=12)

    assert _decode_bytes(accepted, limits=limits).metadata == (("k", "123456789012"),)
    with pytest.raises(SafetensorsLimitExceeded) as raised:
        _decode_bytes(rejected, limits=limits)

    assert raised.value.code is SafetensorsErrorCode.JSON_STRING_EXCEEDS_POLICY_LIMIT
    assert raised.value.limit == 12


def test_decode_header_enforces_json_token_limit_at_the_boundary() -> None:
    header = b'{"__metadata__":null}'

    decoded = _decode_bytes(header, limits=SafetensorsLimits(max_json_tokens=3))
    assert decoded.metadata_form is MetadataForm.NULL

    with pytest.raises(SafetensorsLimitExceeded) as raised:
        _decode_bytes(header, limits=SafetensorsLimits(max_json_tokens=2))

    assert raised.value.code is SafetensorsErrorCode.JSON_TOKEN_EXCEEDS_POLICY_LIMIT
    assert raised.value.limit == 2


@pytest.mark.parametrize(
    "wide_value",
    [
        b"[" + b",".join([b"0"] * 1000) + b"]",
        b"[" + b",".join([b"[]"] * 1000) + b"]",
    ],
)
def test_json_token_limit_stops_wide_values_before_materialization(
    wide_value: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    header = b'{"x":{"dtype":"F16","shape":[],"data_offsets":[0,0],"extra":' + wide_value + b"}}"

    def forbidden_json_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("json.loads must not run after the token limit")

    monkeypatch.setattr("peftlint.safetensors.json.loads", forbidden_json_loads)

    with pytest.raises(SafetensorsLimitExceeded) as raised:
        _decode_bytes(header, limits=SafetensorsLimits(max_json_tokens=100))

    assert raised.value.code is SafetensorsErrorCode.JSON_TOKEN_EXCEEDS_POLICY_LIMIT


@pytest.mark.parametrize(
    "header",
    [
        b'{"\\ud800":{"dtype":"F16","shape":[],"data_offsets":[0,0]}}',
        b'{"\\udc00":{"dtype":"F16","shape":[],"data_offsets":[0,0]}}',
        b'{"x":{"dtype":"\\ud800","shape":[],"data_offsets":[0,0]}}',
        b'{"x":{"dtype":"F16","shape":[],"data_offsets":[0,0],"\\ud800":null}}',
        b'{"__metadata__":{"\\ud800":"value"}}',
        b'{"__metadata__":{"key":"\\udc00"}}',
    ],
)
def test_decode_header_rejects_lone_surrogates_in_manifest_values(header: bytes) -> None:
    with pytest.raises(InvalidSafetensors) as raised:
        _decode_bytes(header)

    assert raised.value.code is SafetensorsErrorCode.INVALID_UNICODE_SCALAR


def test_decode_header_accepts_a_valid_surrogate_pair() -> None:
    header = b'{"\\ud83d\\ude00":{"dtype":"F16","shape":[],"data_offsets":[0,0]}}'

    assert _decode_bytes(header).tensors[0].name == "\N{GRINNING FACE}"


@pytest.mark.parametrize(
    "ignored_value",
    [
        b'"\\ud800"',
        b'{"value":"\\udc00"}',
        b'["\\ud800",{"nested":"\\udc00"}]',
    ],
)
def test_decode_header_matches_v08_ignored_surrogate_values(ignored_value: bytes) -> None:
    header = b'{"x":{"dtype":"F16","shape":[],"data_offsets":[0,0],"extra":' + ignored_value + b"}}"

    decoded = _decode_bytes(header)

    assert decoded.tensors[0].unknown_fields == ("extra",)


def test_decode_header_deliberately_defers_storage_validation() -> None:
    header = (
        b'{"x":{"dtype":"","shape":[2],"data_offsets":[9,1]},'
        b'"y":{"dtype":"future","shape":[99],"data_offsets":[1000,2000]}}'
    )

    decoded = _decode_bytes(header, data_size=1)

    assert tuple(tensor.dtype for tensor in decoded.tensors) == ("", "future")
    assert decoded.tensors[0].data_offsets == (9, 1)
    assert decoded.tensors[1].data_offsets == (1000, 2000)


def test_decode_header_defers_overlap_and_shape_span_mismatch() -> None:
    header = (
        b'{"a":{"dtype":"F32","shape":[100],"data_offsets":[0,4]},'
        b'"b":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}'
    )

    decoded = _decode_bytes(header, data_size=6)

    assert tuple(tensor.data_offsets for tensor in decoded.tensors) == ((0, 4), (2, 6))


def test_decode_header_allows_empty_manifest_with_unindexed_payload_at_schema_stage() -> None:
    decoded = _decode_bytes(b"{}", data_size=1024)

    assert decoded.tensors == ()
    assert decoded.plan.data_size == 1024


def test_decoded_header_repr_redacts_all_header_derived_strings() -> None:
    marker = "private-marker"
    header = (
        b'{"private-marker":{"dtype":"private-marker","shape":[],"data_offsets":[0,0],'
        b'"private-marker-extra":null},"__metadata__":{"private-marker":"private-marker"}}'
    )

    decoded = _decode_bytes(header)

    assert marker not in repr(decoded)
    assert marker not in repr(decoded.tensors[0])


def test_tensor_header_validates_manual_construction() -> None:
    cases: list[tuple[dict[str, object], type[Exception], str]] = [
        ({"name": 1}, TypeError, "tensor name and dtype must be strings"),
        ({"dtype": 1}, TypeError, "tensor name and dtype must be strings"),
        (
            {"name": "\ud800"},
            ValueError,
            "tensor name and dtype must contain valid Unicode scalars",
        ),
        (
            {"dtype": "\udc00"},
            ValueError,
            "tensor name and dtype must contain valid Unicode scalars",
        ),
        (
            {"name": "__metadata__"},
            ValueError,
            "__metadata__ is reserved and cannot be a tensor name",
        ),
        ({"shape": []}, TypeError, "tensor shape must be a tuple of integers"),
        ({"shape": (True,)}, TypeError, "tensor shape must be a tuple of integers"),
        (
            {"shape": (-1,)},
            ValueError,
            "tensor shape integers must fit unsigned 64-bit values",
        ),
        (
            {"shape": (2**64,)},
            ValueError,
            "tensor shape integers must fit unsigned 64-bit values",
        ),
        (
            {"data_offsets": [0, 0]},
            TypeError,
            "tensor data_offsets must contain exactly two integers",
        ),
        (
            {"data_offsets": (0,)},
            TypeError,
            "tensor data_offsets must contain exactly two integers",
        ),
        (
            {"data_offsets": (0, True)},
            TypeError,
            "tensor data_offsets must contain exactly two integers",
        ),
        (
            {"data_offsets": (-1, 0)},
            ValueError,
            "tensor offsets must fit unsigned 64-bit values",
        ),
        (
            {"unknown_fields": ["extra"]},
            TypeError,
            "unknown_fields must be a tuple of strings",
        ),
        (
            {"unknown_fields": (1,)},
            TypeError,
            "unknown_fields must be a tuple of strings",
        ),
        (
            {"unknown_fields": ("\ud800",)},
            ValueError,
            "unknown field names must contain valid Unicode scalars",
        ),
        (
            {"unknown_fields": ("z", "a")},
            ValueError,
            "unknown field names must be sorted and unique",
        ),
        (
            {"unknown_fields": ("shape",)},
            ValueError,
            "required tensor fields cannot be unknown fields",
        ),
    ]

    for overrides, error_type, message in cases:
        values: dict[str, object] = {
            "name": "x",
            "dtype": "F16",
            "shape": (),
            "data_offsets": (0, 0),
            "unknown_fields": (),
        }
        values.update(overrides)
        with pytest.raises(error_type, match=f"^{message}$"):
            TensorHeader(**values)  # type: ignore[arg-type]


def test_decoded_header_validates_manual_construction() -> None:
    default_plan = plan_header_read(bytes(8), file_size=8)
    tensor_a = TensorHeader(name="a", dtype="F16", shape=(), data_offsets=(0, 0))
    tensor_b = TensorHeader(name="b", dtype="F16", shape=(), data_offsets=(0, 0))
    tensor_with_extra = TensorHeader(
        name="a",
        dtype="F16",
        shape=(),
        data_offsets=(0, 0),
        unknown_fields=("extra",),
    )
    cases: list[tuple[dict[str, object], type[Exception], str]] = [
        ({"plan": object()}, TypeError, "decoded header plan must be HeaderReadPlan"),
        (
            {"tensors": [tensor_a]},
            TypeError,
            "decoded header tensors must be a tuple of TensorHeader values",
        ),
        (
            {"tensors": (object(),)},
            TypeError,
            "decoded header tensors must be a tuple of TensorHeader values",
        ),
        (
            {"tensors": (tensor_b, tensor_a)},
            ValueError,
            "decoded header tensor names must be sorted and unique",
        ),
        (
            {"tensors": (tensor_a, tensor_a)},
            ValueError,
            "decoded header tensor names must be sorted and unique",
        ),
        (
            {
                "plan": plan_header_read(
                    bytes(8),
                    file_size=8,
                    limits=SafetensorsLimits(max_tensors=0),
                ),
                "tensors": (tensor_a,),
            },
            ValueError,
            "decoded header exceeds its tensor count limit",
        ),
        (
            {
                "plan": plan_header_read(
                    bytes(8),
                    file_size=8,
                    limits=SafetensorsLimits(max_tensor_name_bytes=0),
                ),
                "tensors": (tensor_a,),
            },
            ValueError,
            "decoded header exceeds its tensor name limit",
        ),
        (
            {
                "plan": plan_header_read(
                    bytes(8),
                    file_size=8,
                    limits=SafetensorsLimits(max_tensor_rank=0),
                ),
                "tensors": (TensorHeader(name="a", dtype="F16", shape=(1,), data_offsets=(0, 0)),),
            },
            ValueError,
            "decoded header exceeds its tensor rank limit",
        ),
        (
            {"metadata": []},
            TypeError,
            "decoded header metadata must contain string pairs",
        ),
        (
            {"metadata": (("key",),)},
            TypeError,
            "decoded header metadata must contain string pairs",
        ),
        (
            {"metadata": ((1, "value"),)},
            TypeError,
            "decoded header metadata must contain string pairs",
        ),
        (
            {"metadata": (("z", "1"), ("a", "2"))},
            ValueError,
            "decoded header metadata keys must be sorted and unique",
        ),
        (
            {"metadata": (("\ud800", "value"),)},
            ValueError,
            "decoded header metadata must contain valid Unicode scalars",
        ),
        (
            {
                "plan": plan_header_read(
                    bytes(8),
                    file_size=8,
                    limits=SafetensorsLimits(max_metadata_entries=0),
                ),
                "metadata": (("key", "value"),),
            },
            ValueError,
            "decoded header exceeds its metadata entry limit",
        ),
        ({"metadata_form": "object"}, TypeError, "metadata_form must be MetadataForm"),
        (
            {"metadata": (("key", "value"),)},
            ValueError,
            "absent or null metadata cannot contain entries",
        ),
        (
            {"notices": [HeaderNotice.METADATA_NULL]},
            TypeError,
            "decoded header notices must be a tuple of HeaderNotice values",
        ),
        (
            {"notices": ("metadata_null",)},
            TypeError,
            "decoded header notices must be a tuple of HeaderNotice values",
        ),
        (
            {"notices": (HeaderNotice.METADATA_NULL, HeaderNotice.METADATA_NULL)},
            ValueError,
            "decoded header notices must be sorted and unique",
        ),
        (
            {"metadata_form": MetadataForm.NULL},
            ValueError,
            "decoded header notices are inconsistent",
        ),
        (
            {"tensors": (tensor_with_extra,)},
            ValueError,
            "decoded header notices are inconsistent",
        ),
    ]

    for overrides, error_type, message in cases:
        values: dict[str, object] = {
            "plan": default_plan,
            "tensors": (),
            "metadata": (),
            "metadata_form": MetadataForm.ABSENT,
            "notices": (),
        }
        values.update(overrides)
        with pytest.raises(error_type, match=f"^{message}$"):
            DecodedSafetensorsHeader(**values)  # type: ignore[arg-type]


def test_decode_error_never_renders_header_values_or_chained_decoder_exceptions() -> None:
    marker = "never-copy-private-marker"
    header = b'{"__metadata__":{"never-copy-private-marker":1}}'

    with pytest.raises(InvalidSafetensors) as raised:
        _decode_bytes(header)

    rendered = f"{raised.value!s} {raised.value!r} {raised.value.args!r}"
    assert marker not in rendered
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("header", [b"\xff-private-marker", b"{private-marker"])
def test_decoder_errors_do_not_retain_codec_or_json_context(header: bytes) -> None:
    with pytest.raises(InvalidSafetensors) as raised:
        _decode_bytes(header)

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_decode_header_api_cannot_receive_a_source_or_payload() -> None:
    assert tuple(inspect.signature(decode_header).parameters) == ("envelope",)

    with pytest.raises(TypeError, match="unexpected keyword argument 'payload'"):
        decode_header(object(), payload=object())  # type: ignore[call-arg,arg-type]


def test_decode_header_requires_an_envelope() -> None:
    with pytest.raises(TypeError, match=r"^envelope must be HeaderEnvelope$"):
        decode_header(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_json_depth", True, "max_json_depth must be an integer"),
        ("max_json_depth", 0, "max_json_depth must be positive"),
        ("max_json_string_chars", True, "max_json_string_chars must be an integer"),
        ("max_json_string_chars", -1, "max_json_string_chars must not be negative"),
        ("max_json_tokens", True, "max_json_tokens must be an integer"),
        ("max_json_tokens", 0, "max_json_tokens must be positive"),
        ("max_tensors", True, "max_tensors must be an integer"),
        ("max_tensors", -1, "max_tensors must not be negative"),
        ("max_tensor_rank", True, "max_tensor_rank must be an integer"),
        ("max_tensor_rank", -1, "max_tensor_rank must not be negative"),
        ("max_tensor_name_bytes", True, "max_tensor_name_bytes must be an integer"),
        ("max_tensor_name_bytes", -1, "max_tensor_name_bytes must not be negative"),
        ("max_metadata_entries", True, "max_metadata_entries must be an integer"),
        ("max_metadata_entries", -1, "max_metadata_entries must not be negative"),
    ],
)
def test_extended_limits_validate_exact_types_and_ranges(
    field: str,
    value: object,
    message: str,
) -> None:
    arguments: dict[str, object] = {field: value}

    with pytest.raises((TypeError, ValueError), match=f"^{message}$"):
        SafetensorsLimits(**arguments)  # type: ignore[arg-type]
