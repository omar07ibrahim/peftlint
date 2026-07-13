from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace

import pytest

import peftlint
from peftlint.safetensors import (
    DEFAULT_SAFETENSORS_LIMITS,
    HeaderNotice,
    HeaderReadPlan,
    InvalidSafetensors,
    MetadataForm,
    SafetensorsDtype,
    SafetensorsErrorCode,
    SafetensorsLimits,
    SafetensorsManifest,
    TensorManifest,
    accept_header,
    decode_header,
    parse_safetensors_manifest,
    plan_header_read,
    validate_storage,
)

_U64_MAX = 2**64 - 1


def _encode(root: Mapping[str, object]) -> bytes:
    return json.dumps(root, ensure_ascii=True, separators=(",", ":")).encode()


def _tensor(
    dtype: str,
    shape: list[int],
    begin: int,
    end: int,
    **extensions: object,
) -> dict[str, object]:
    return {
        "dtype": dtype,
        "shape": shape,
        "data_offsets": [begin, end],
        **extensions,
    }


def _parse_root(
    root: Mapping[str, object],
    *,
    payload_size: int,
    limits: SafetensorsLimits = DEFAULT_SAFETENSORS_LIMITS,
) -> SafetensorsManifest:
    header = _encode(root)
    prefix = len(header).to_bytes(8, "little")
    return parse_safetensors_manifest(
        prefix,
        header,
        file_size=8 + len(header) + payload_size,
        limits=limits,
    )


_DTYPE_CASES = [
    (SafetensorsDtype.F4, 4, 2, 1),
    (SafetensorsDtype.F6_E2M3, 6, 4, 3),
    (SafetensorsDtype.F6_E3M2, 6, 4, 3),
    (SafetensorsDtype.BOOL, 8, 1, 1),
    (SafetensorsDtype.U8, 8, 1, 1),
    (SafetensorsDtype.I8, 8, 1, 1),
    (SafetensorsDtype.F8_E5M2, 8, 1, 1),
    (SafetensorsDtype.F8_E4M3, 8, 1, 1),
    (SafetensorsDtype.F8_E8M0, 8, 1, 1),
    (SafetensorsDtype.F8_E4M3FNUZ, 8, 1, 1),
    (SafetensorsDtype.F8_E5M2FNUZ, 8, 1, 1),
    (SafetensorsDtype.I16, 16, 1, 2),
    (SafetensorsDtype.U16, 16, 1, 2),
    (SafetensorsDtype.F16, 16, 1, 2),
    (SafetensorsDtype.BF16, 16, 1, 2),
    (SafetensorsDtype.I32, 32, 1, 4),
    (SafetensorsDtype.U32, 32, 1, 4),
    (SafetensorsDtype.F32, 32, 1, 4),
    (SafetensorsDtype.C64, 64, 1, 8),
    (SafetensorsDtype.F64, 64, 1, 8),
    (SafetensorsDtype.I64, 64, 1, 8),
    (SafetensorsDtype.U64, 64, 1, 8),
]


def test_dtype_vocabulary_is_pinned_to_literal_v08_tokens() -> None:
    assert tuple(dtype.value for dtype in SafetensorsDtype) == (
        "F4",
        "F6_E2M3",
        "F6_E3M2",
        "BOOL",
        "U8",
        "I8",
        "F8_E5M2",
        "F8_E4M3",
        "F8_E8M0",
        "F8_E4M3FNUZ",
        "F8_E5M2FNUZ",
        "I16",
        "U16",
        "F16",
        "BF16",
        "I32",
        "U32",
        "F32",
        "C64",
        "F64",
        "I64",
        "U64",
    )


@pytest.mark.parametrize(("dtype", "bits", "elements", "nbytes"), _DTYPE_CASES)
def test_validate_storage_supports_every_v08_dtype(
    dtype: SafetensorsDtype,
    bits: int,
    elements: int,
    nbytes: int,
) -> None:
    manifest = _parse_root(
        {"tensor": _tensor(dtype.value, [elements], 0, nbytes)},
        payload_size=nbytes,
    )
    tensor = manifest.tensors[0]

    assert len(SafetensorsDtype) == 22
    assert dtype.bits_per_element == bits
    assert tensor.dtype is dtype
    assert tensor.element_count == elements
    assert tensor.nbytes == nbytes


@pytest.mark.parametrize("dtype", ["", "f16", "F16 ", "FLOAT32", "F8_E4M3FN"])
def test_validate_storage_rejects_every_non_v08_dtype(dtype: str) -> None:
    with pytest.raises(InvalidSafetensors) as raised:
        _parse_root({"tensor": _tensor(dtype, [], 0, 0)}, payload_size=0)

    assert raised.value.code is SafetensorsErrorCode.INVALID_DTYPE
    assert raised.value.rule_id == "PL100"


def test_scalar_shape_has_one_element() -> None:
    manifest = _parse_root(
        {"scalar": _tensor("F16", [], 0, 2)},
        payload_size=2,
    )

    assert manifest.tensors[0].element_count == 1
    assert manifest.tensors[0].nbytes == 2


@pytest.mark.parametrize(
    ("dtype", "count", "nbytes"),
    [
        ("F4", 0, 0),
        ("F4", 2, 1),
        ("F6_E2M3", 0, 0),
        ("F6_E2M3", 4, 3),
        ("F6_E3M2", 0, 0),
        ("F6_E3M2", 4, 3),
    ],
)
def test_subbyte_dtypes_accept_only_byte_aligned_counts(
    dtype: str,
    count: int,
    nbytes: int,
) -> None:
    manifest = _parse_root(
        {"tensor": _tensor(dtype, [count], 0, nbytes)},
        payload_size=nbytes,
    )

    assert manifest.tensors[0].element_count == count
    assert manifest.tensors[0].nbytes == nbytes


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        ("F4", []),
        ("F4", [1]),
        ("F4", [3]),
        ("F6_E2M3", []),
        ("F6_E2M3", [1]),
        ("F6_E2M3", [2]),
        ("F6_E2M3", [3]),
        ("F6_E3M2", [1]),
        ("F6_E3M2", [2]),
        ("F6_E3M2", [3]),
    ],
)
def test_subbyte_dtypes_reject_partial_bytes(dtype: str, shape: list[int]) -> None:
    with pytest.raises(InvalidSafetensors) as raised:
        _parse_root({"tensor": _tensor(dtype, shape, 0, 0)}, payload_size=0)

    assert raised.value.code is SafetensorsErrorCode.TENSOR_BYTE_MISALIGNED


def test_ordered_shape_product_preserves_v08_zero_and_overflow_semantics() -> None:
    valid = _parse_root(
        {"tensor": _tensor("F64", [0, 2**63, 2], 0, 0)},
        payload_size=0,
    )
    assert valid.tensors[0].element_count == 0

    with pytest.raises(InvalidSafetensors) as raised:
        _parse_root(
            {"tensor": _tensor("F64", [2**63, 2, 0], 0, 0)},
            payload_size=0,
        )

    assert raised.value.code is SafetensorsErrorCode.TENSOR_SIZE_OVERFLOW


def test_checked_bit_multiplication_is_not_optimized_away_for_u8() -> None:
    too_many_elements = _U64_MAX // 8 + 1

    with pytest.raises(InvalidSafetensors) as raised:
        _parse_root(
            {"tensor": _tensor("U8", [too_many_elements], 0, 0)},
            payload_size=0,
        )

    assert raised.value.code is SafetensorsErrorCode.TENSOR_SIZE_OVERFLOW


def test_checked_bit_multiplication_accepts_its_exact_u64_boundary_without_payload() -> None:
    elements = _U64_MAX // 8
    manifest = _parse_root(
        {"tensor": _tensor("U8", [elements], 0, elements)},
        payload_size=elements,
    )

    assert manifest.tensors[0].nbytes == elements
    assert manifest.plan.data_size == elements


@pytest.mark.parametrize(
    ("root", "payload_size", "code"),
    [
        (
            {"tensor": _tensor("U8", [1], 1, 2)},
            2,
            SafetensorsErrorCode.TENSOR_LAYOUT_GAP,
        ),
        (
            {
                "a": _tensor("U8", [1], 0, 1),
                "b": _tensor("U8", [1], 2, 3),
            },
            3,
            SafetensorsErrorCode.TENSOR_LAYOUT_GAP,
        ),
        (
            {
                "a": _tensor("U8", [2], 0, 2),
                "b": _tensor("U8", [1], 1, 2),
            },
            2,
            SafetensorsErrorCode.TENSOR_LAYOUT_OVERLAP,
        ),
        (
            {
                "a": _tensor("U8", [1], 0, 1),
                "tensor": _tensor("U8", [0], 1, 0),
            },
            1,
            SafetensorsErrorCode.TENSOR_OFFSETS_REVERSED,
        ),
        (
            {"tensor": _tensor("U8", [2], 0, 1)},
            1,
            SafetensorsErrorCode.TENSOR_SIZE_MISMATCH,
        ),
        (
            {"tensor": _tensor("U8", [1], 0, 2)},
            2,
            SafetensorsErrorCode.TENSOR_SIZE_MISMATCH,
        ),
        (
            {"tensor": _tensor("U8", [1], 0, 1)},
            2,
            SafetensorsErrorCode.PAYLOAD_SIZE_MISMATCH,
        ),
        (
            {"tensor": _tensor("U8", [2], 0, 2)},
            1,
            SafetensorsErrorCode.PAYLOAD_SIZE_MISMATCH,
        ),
        ({}, 1, SafetensorsErrorCode.PAYLOAD_SIZE_MISMATCH),
    ],
)
def test_validate_storage_rejects_invalid_layouts(
    root: dict[str, object],
    payload_size: int,
    code: SafetensorsErrorCode,
) -> None:
    with pytest.raises(InvalidSafetensors) as raised:
        _parse_root(root, payload_size=payload_size)

    assert raised.value.code is code
    assert raised.value.rule_id == "PL101"


def test_empty_manifest_is_storage_valid_only_for_an_empty_payload() -> None:
    manifest = _parse_root({}, payload_size=0)

    assert manifest.tensors == ()
    assert manifest.plan.data_size == 0


def test_storage_order_is_numeric_and_independent_of_json_name_order() -> None:
    root = {
        "z-last-name": _tensor("U8", [1], 1, 2),
        "a-first-name": _tensor("U8", [1], 0, 1),
    }

    manifest = _parse_root(root, payload_size=2)

    assert tuple(tensor.name for tensor in manifest.tensors) == (
        "a-first-name",
        "z-last-name",
    )


def test_zero_spans_sort_before_positive_spans_at_the_same_boundary() -> None:
    root = {
        "a-positive": _tensor("U8", [1], 0, 1),
        "z-zero-origin": _tensor("U8", [0], 0, 0),
        "z-zero-end": _tensor("U8", [0], 1, 1),
        "b-f64-no-natural-alignment": _tensor("F64", [1], 1, 9),
    }

    manifest = _parse_root(root, payload_size=9)

    assert tuple(tensor.name for tensor in manifest.tensors) == (
        "z-zero-origin",
        "a-positive",
        "z-zero-end",
        "b-f64-no-natural-alignment",
    )


def test_multiple_zero_tensors_share_a_boundary_with_name_tiebreaking() -> None:
    root = {
        "z": _tensor("F4", [0], 0, 0),
        "a": _tensor("F6_E2M3", [0], 0, 0),
        "m": _tensor("U8", [1], 0, 1),
    }

    manifest = _parse_root(root, payload_size=1)

    assert tuple(tensor.name for tensor in manifest.tensors) == ("a", "z", "m")


def test_zero_tensor_inside_a_nonempty_span_is_an_overlap() -> None:
    root = {
        "a": _tensor("U8", [2], 0, 2),
        "z": _tensor("U8", [0], 1, 1),
    }

    with pytest.raises(InvalidSafetensors) as raised:
        _parse_root(root, payload_size=2)

    assert raised.value.code is SafetensorsErrorCode.TENSOR_LAYOUT_OVERLAP


def test_validation_precedence_is_fail_closed_and_deterministic() -> None:
    with pytest.raises(InvalidSafetensors) as unknown_dtype:
        _parse_root(
            {"tensor": _tensor("future", [2**63, 2], 1, 0)},
            payload_size=0,
        )
    assert unknown_dtype.value.code is SafetensorsErrorCode.INVALID_DTYPE

    with pytest.raises(InvalidSafetensors) as offsets:
        _parse_root(
            {
                "a": _tensor("U8", [1], 0, 1),
                "tensor": _tensor("F4", [2**63, 2], 1, 0),
            },
            payload_size=1,
        )
    assert offsets.value.code is SafetensorsErrorCode.TENSOR_OFFSETS_REVERSED

    with pytest.raises(InvalidSafetensors) as overflow:
        _parse_root(
            {"tensor": _tensor("F4", [2**63, 2], 0, 0)},
            payload_size=0,
        )
    assert overflow.value.code is SafetensorsErrorCode.TENSOR_SIZE_OVERFLOW

    with pytest.raises(InvalidSafetensors) as alignment:
        _parse_root(
            {"tensor": _tensor("F4", [1], 0, 7)},
            payload_size=7,
        )
    assert alignment.value.code is SafetensorsErrorCode.TENSOR_BYTE_MISALIGNED

    with pytest.raises(InvalidSafetensors) as span:
        _parse_root(
            {"tensor": _tensor("U8", [1], 0, 7)},
            payload_size=8,
        )
    assert span.value.code is SafetensorsErrorCode.TENSOR_SIZE_MISMATCH


def test_sorted_tensor_precedence_does_not_let_late_reversal_mask_early_failure() -> None:
    root = {
        "a-overflow": _tensor("U8", [2**63, 2], 0, 0),
        "z-reversed": _tensor("U8", [0], 1, 0),
    }

    with pytest.raises(InvalidSafetensors) as raised:
        _parse_root(root, payload_size=0)

    assert raised.value.code is SafetensorsErrorCode.TENSOR_SIZE_OVERFLOW


def test_sorted_gap_precedes_a_later_reversed_range() -> None:
    root = {
        "a-gap": _tensor("U8", [1], 1, 2),
        "z-reversed": _tensor("U8", [0], 3, 2),
    }

    with pytest.raises(InvalidSafetensors) as raised:
        _parse_root(root, payload_size=2)

    assert raised.value.code is SafetensorsErrorCode.TENSOR_LAYOUT_GAP


def test_manifest_preserves_metadata_extensions_and_notices() -> None:
    root = {
        "__metadata__": {"format": "pt"},
        "tensor": _tensor("U8", [1], 0, 1, future={"ignored": True}),
    }

    manifest = _parse_root(root, payload_size=1)

    assert manifest.metadata == (("format", "pt"),)
    assert manifest.metadata_form is MetadataForm.OBJECT
    assert manifest.notices == (HeaderNotice.UNKNOWN_TENSOR_FIELDS,)
    assert manifest.tensors[0].unknown_fields == ("future",)


def test_pure_convenience_api_equals_the_explicit_four_stage_pipeline() -> None:
    header = _encode({"tensor": _tensor("F32", [2], 0, 8)})
    prefix = len(header).to_bytes(8, "little")
    file_size = 8 + len(header) + 8

    plan = plan_header_read(prefix, file_size=file_size)
    explicit = validate_storage(decode_header(accept_header(plan, header)))
    composed = parse_safetensors_manifest(prefix, header, file_size=file_size)

    assert composed == explicit
    assert peftlint.parse_safetensors_manifest is parse_safetensors_manifest


def test_root_manifest_api_exports_the_types_visible_in_its_result() -> None:
    assert peftlint.DEFAULT_SAFETENSORS_LIMITS is DEFAULT_SAFETENSORS_LIMITS
    assert peftlint.HeaderNotice is HeaderNotice
    assert peftlint.HeaderReadPlan is HeaderReadPlan
    assert peftlint.MetadataForm is MetadataForm


def test_pure_convenience_api_has_no_source_or_payload_surface() -> None:
    assert tuple(inspect.signature(parse_safetensors_manifest).parameters) == (
        "prefix",
        "header",
        "file_size",
        "limits",
    )

    with pytest.raises(TypeError, match="unexpected keyword argument 'payload'"):
        parse_safetensors_manifest(
            bytes(8),
            b"",
            file_size=8,
            payload=object(),  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError, match="unexpected keyword argument 'path'"):
        parse_safetensors_manifest(
            bytes(8),
            b"",
            file_size=8,
            path=object(),  # type: ignore[call-arg]
        )


def test_validate_storage_requires_a_decoded_header() -> None:
    with pytest.raises(TypeError, match=r"^decoded must be DecodedSafetensorsHeader$"):
        validate_storage(object())  # type: ignore[arg-type]


def test_validate_storage_revalidates_nested_tensor_headers() -> None:
    header = _encode({"tensor": _tensor("U8", [1], 0, 1)})
    plan = plan_header_read(
        len(header).to_bytes(8, "little"),
        file_size=8 + len(header) + 1,
    )
    decoded = decode_header(accept_header(plan, header))
    object.__setattr__(decoded.tensors[0], "shape", [1])

    with pytest.raises(TypeError, match=r"^tensor shape must be a tuple of integers$"):
        validate_storage(decoded)


def test_storage_errors_are_redacted_and_mapped_to_pl101() -> None:
    marker = "private-tensor-marker"

    with pytest.raises(InvalidSafetensors) as raised:
        _parse_root(
            {marker: _tensor("U8", [1], 0, 0, private_extension_marker=marker)},
            payload_size=0,
        )

    rendered = f"{raised.value!s} {raised.value!r} {raised.value.args!r}"
    assert marker not in rendered
    assert raised.value.rule_id == "PL101"


def test_schema_errors_are_mapped_to_pl100() -> None:
    header = b"not-json"
    prefix = len(header).to_bytes(8, "little")

    with pytest.raises(InvalidSafetensors) as raised:
        parse_safetensors_manifest(prefix, header, file_size=8 + len(header))

    assert raised.value.code is SafetensorsErrorCode.HEADER_JSON
    assert raised.value.rule_id == "PL100"


def test_manifest_repr_redacts_names_metadata_and_extension_fields() -> None:
    marker = "private-marker"
    manifest = _parse_root(
        {
            "__metadata__": {marker: marker},
            marker: _tensor("U8", [1], 0, 1, private_marker=True),
        },
        payload_size=1,
    )

    assert marker not in repr(manifest)
    assert marker not in repr(manifest.tensors[0])


def test_manifest_values_are_immutable() -> None:
    manifest = _parse_root(
        {"tensor": _tensor("U8", [1], 0, 1)},
        payload_size=1,
    )

    with pytest.raises(FrozenInstanceError):
        manifest.tensors = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        manifest.tensors[0].shape = ()  # type: ignore[misc]


def test_public_header_values_reject_pickle_style_state_mutation() -> None:
    header = _encode({"tensor": _tensor("U8", [1], 0, 1)})
    plan = plan_header_read(
        len(header).to_bytes(8, "little"),
        file_size=8 + len(header) + 1,
    )
    envelope = accept_header(plan, header)
    decoded = decode_header(envelope)
    manifest = validate_storage(decoded)
    values = (
        DEFAULT_SAFETENSORS_LIMITS,
        plan,
        envelope,
        decoded.tensors[0],
        decoded,
        manifest.tensors[0],
        manifest,
    )

    for value in values:
        with pytest.raises(TypeError, match=rf"^{type(value).__name__} is immutable$"):
            value.__setstate__(object())

    assert manifest.tensors[0].element_count == 1
    assert manifest.tensors[0].nbytes == 1


def test_replacing_a_tensor_manifest_recomputes_derived_values() -> None:
    original = TensorManifest(
        name="tensor",
        dtype=SafetensorsDtype.F32,
        shape=(2,),
        data_offsets=(0, 8),
    )

    replaced = replace(
        original,
        dtype=SafetensorsDtype.U8,
        shape=(8,),
        data_offsets=(0, 8),
    )

    assert replaced.element_count == 8
    assert replaced.nbytes == 8


def test_tensor_manifest_rejects_forged_or_inconsistent_values() -> None:
    cases: list[tuple[dict[str, object], type[Exception], str]] = [
        (
            {"dtype": "U8"},
            TypeError,
            "tensor manifest dtype must be SafetensorsDtype",
        ),
        (
            {"shape": [1]},
            TypeError,
            "tensor shape must be a tuple of integers",
        ),
        (
            {"data_offsets": (1, 0)},
            InvalidSafetensors,
            "safetensors tensor offsets are reversed",
        ),
        (
            {"data_offsets": (0, 2)},
            InvalidSafetensors,
            "safetensors tensor span does not match its dtype and shape",
        ),
        (
            {
                "dtype": SafetensorsDtype.F4,
                "shape": (1,),
                "data_offsets": (0, 0),
            },
            InvalidSafetensors,
            "safetensors sub-byte tensor does not end on a byte boundary",
        ),
        (
            {
                "shape": (2**63, 2),
                "data_offsets": (0, 0),
            },
            InvalidSafetensors,
            "safetensors tensor size exceeds unsigned 64-bit arithmetic",
        ),
    ]

    for overrides, error_type, message in cases:
        values: dict[str, object] = {
            "name": "tensor",
            "dtype": SafetensorsDtype.U8,
            "shape": (1,),
            "data_offsets": (0, 1),
            "unknown_fields": (),
        }
        values.update(overrides)
        with pytest.raises(error_type, match=f"^{message}$"):
            TensorManifest(**values)  # type: ignore[arg-type]


def test_tensor_manifest_derived_values_cannot_be_supplied_or_mutated() -> None:
    tensor = TensorManifest(
        name="tensor",
        dtype=SafetensorsDtype.F32,
        shape=(2,),
        data_offsets=(0, 8),
    )

    assert tensor.element_count == 2
    assert tensor.nbytes == 8
    with pytest.raises(TypeError, match="unexpected keyword argument '_element_count'"):
        TensorManifest(
            name="tensor",
            dtype=SafetensorsDtype.F32,
            shape=(2,),
            data_offsets=(0, 8),
            _element_count=99,  # type: ignore[call-arg]
        )


def test_safetensors_manifest_rejects_forged_aggregate_state() -> None:
    plan_two = plan_header_read(bytes(8), file_size=10)
    first = TensorManifest(
        name="first",
        dtype=SafetensorsDtype.U8,
        shape=(1,),
        data_offsets=(0, 1),
    )
    second = TensorManifest(
        name="second",
        dtype=SafetensorsDtype.U8,
        shape=(1,),
        data_offsets=(1, 2),
    )
    duplicate_name = TensorManifest(
        name="first",
        dtype=SafetensorsDtype.U8,
        shape=(1,),
        data_offsets=(1, 2),
    )
    gap = TensorManifest(
        name="gap",
        dtype=SafetensorsDtype.U8,
        shape=(1,),
        data_offsets=(1, 2),
    )
    overlap = TensorManifest(
        name="overlap",
        dtype=SafetensorsDtype.U8,
        shape=(1,),
        data_offsets=(0, 1),
    )
    forged_derived_state = TensorManifest(
        name="first",
        dtype=SafetensorsDtype.U8,
        shape=(1,),
        data_offsets=(0, 1),
    )
    object.__setattr__(forged_derived_state, "_nbytes", 99)
    cases: list[tuple[dict[str, object], type[Exception], str]] = [
        (
            {"plan": object()},
            TypeError,
            "safetensors manifest plan must be HeaderReadPlan",
        ),
        (
            {"tensors": [first, second]},
            TypeError,
            "manifest tensors must be a tuple of TensorManifest values",
        ),
        (
            {"tensors": (object(),)},
            TypeError,
            "manifest tensors must be a tuple of TensorManifest values",
        ),
        (
            {"tensors": (forged_derived_state, second)},
            ValueError,
            "manifest tensor derived state is inconsistent",
        ),
        (
            {"tensors": (second, first)},
            ValueError,
            "manifest tensors must use canonical storage order",
        ),
        (
            {"tensors": (first, duplicate_name)},
            ValueError,
            "decoded header tensor names must be sorted and unique",
        ),
        (
            {"tensors": (gap,)},
            ValueError,
            "manifest tensor layout must be hole-free",
        ),
        (
            {"tensors": (first, overlap)},
            ValueError,
            "manifest tensor layout must be hole-free",
        ),
        (
            {"tensors": (first,)},
            ValueError,
            "manifest tensor layout must cover the payload",
        ),
        (
            {"metadata": (("key", "value"),)},
            ValueError,
            "absent or null metadata cannot contain entries",
        ),
    ]

    for overrides, error_type, message in cases:
        values: dict[str, object] = {
            "plan": plan_two,
            "tensors": (first, second),
            "metadata": (),
            "metadata_form": MetadataForm.ABSENT,
            "notices": (),
        }
        values.update(overrides)
        with pytest.raises(error_type, match=f"^{message}$"):
            SafetensorsManifest(**values)  # type: ignore[arg-type]


def test_manifest_revalidates_an_internally_inconsistent_plan() -> None:
    plan = plan_header_read(bytes(8), file_size=9)
    tensor = TensorManifest(
        name="tensor",
        dtype=SafetensorsDtype.U8,
        shape=(1,),
        data_offsets=(0, 1),
    )
    object.__setattr__(plan, "file_size", 999)

    with pytest.raises(ValueError, match=r"^header read plan data_size is inconsistent$"):
        SafetensorsManifest(
            plan=plan,
            tensors=(tensor,),
            metadata=(),
            metadata_form=MetadataForm.ABSENT,
            notices=(),
        )


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    [
        ("_element_count", True),
        ("_element_count", 1.0),
        ("_nbytes", True),
        ("_nbytes", 1.0),
    ],
)
def test_manifest_rejects_equality_coercing_derived_state(
    field_name: str,
    forged_value: object,
) -> None:
    plan = plan_header_read(bytes(8), file_size=9)
    tensor = TensorManifest(
        name="tensor",
        dtype=SafetensorsDtype.U8,
        shape=(1,),
        data_offsets=(0, 1),
    )
    object.__setattr__(tensor, field_name, forged_value)

    with pytest.raises(ValueError, match=r"^manifest tensor derived state is inconsistent$"):
        SafetensorsManifest(
            plan=plan,
            tensors=(tensor,),
            metadata=(),
            metadata_form=MetadataForm.ABSENT,
            notices=(),
        )
