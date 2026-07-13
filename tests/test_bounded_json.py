from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

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

LIMITS = JsonLimits(
    max_document_chars=1_024,
    max_depth=8,
    max_string_chars=64,
    max_tokens=64,
    max_number_chars=16,
)


def test_decode_json_preserves_number_lexemes_and_json_scalars() -> None:
    decoded = decode_json(
        '{"integer":-12,"float":1.25e+2,"boolean":true,"null":null}',
        LIMITS,
    )
    assert type(decoded) is dict
    document = cast(dict[str, object], decoded)

    assert document == {
        "integer": IntegerLexeme("-12"),
        "float": FloatLexeme("1.25e+2"),
        "boolean": True,
        "null": None,
    }
    assert "-12" not in repr(document["integer"])
    assert "1.25e+2" not in repr(document["float"])


@pytest.mark.parametrize("text", ["{", "[1,]", '"unterminated', "nul"])
def test_decode_json_rejects_invalid_syntax_without_decoder_context(text: str) -> None:
    with pytest.raises(InvalidJson) as raised:
        decode_json(text, LIMITS)

    assert raised.value.code is BoundedJsonErrorCode.SYNTAX
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_decode_json_rejects_non_json_numeric_constants(constant: str) -> None:
    with pytest.raises(InvalidJson) as raised:
        decode_json(f'{{"value":{constant}}}', LIMITS)

    assert raised.value.code is BoundedJsonErrorCode.SYNTAX


@pytest.mark.parametrize(
    "text",
    [
        '{"key":1,"key":2}',
        '{"outer":{"key":1,"key":2}}',
        '[{"key":1,"key":2}]',
        r'{"a":1,"\u0061":2}',
    ],
)
def test_decode_json_rejects_duplicate_keys_at_every_depth(text: str) -> None:
    with pytest.raises(InvalidJson) as raised:
        decode_json(text, LIMITS)

    assert raised.value.code is BoundedJsonErrorCode.DUPLICATE_KEY


@pytest.mark.parametrize(
    ("accepted", "rejected", "limits", "code", "limit"),
    [
        (
            "null",
            " null",
            JsonLimits(
                max_document_chars=4,
                max_depth=8,
                max_string_chars=64,
                max_tokens=64,
            ),
            BoundedJsonErrorCode.DOCUMENT_LIMIT,
            4,
        ),
        (
            "[]",
            "[[]]",
            JsonLimits(
                max_document_chars=64,
                max_depth=1,
                max_string_chars=64,
                max_tokens=64,
            ),
            BoundedJsonErrorCode.DEPTH_LIMIT,
            1,
        ),
        (
            '"abc"',
            '"abcd"',
            JsonLimits(
                max_document_chars=64,
                max_depth=8,
                max_string_chars=3,
                max_tokens=64,
            ),
            BoundedJsonErrorCode.STRING_LIMIT,
            3,
        ),
        (
            "[]",
            "[0]",
            JsonLimits(
                max_document_chars=64,
                max_depth=8,
                max_string_chars=64,
                max_tokens=1,
            ),
            BoundedJsonErrorCode.TOKEN_LIMIT,
            1,
        ),
        (
            "123",
            "1234",
            JsonLimits(
                max_document_chars=64,
                max_depth=8,
                max_string_chars=64,
                max_tokens=64,
                max_number_chars=3,
            ),
            BoundedJsonErrorCode.NUMBER_LIMIT,
            3,
        ),
        (
            "1.2",
            "1.23",
            JsonLimits(
                max_document_chars=64,
                max_depth=8,
                max_string_chars=64,
                max_tokens=64,
                max_number_chars=3,
            ),
            BoundedJsonErrorCode.NUMBER_LIMIT,
            3,
        ),
        (
            "-1.2e+3",
            "-1.2e+30",
            JsonLimits(
                max_document_chars=64,
                max_depth=8,
                max_string_chars=64,
                max_tokens=64,
                max_number_chars=7,
            ),
            BoundedJsonErrorCode.NUMBER_LIMIT,
            7,
        ),
    ],
)
def test_decode_json_enforces_each_limit_at_the_exact_boundary(
    accepted: str,
    rejected: str,
    limits: JsonLimits,
    code: BoundedJsonErrorCode,
    limit: int,
) -> None:
    decode_json(accepted, limits)

    with pytest.raises(JsonLimitExceeded) as raised:
        decode_json(rejected, limits)

    assert raised.value.code is code
    assert raised.value.limit == limit


def test_decode_json_counts_raw_escaped_string_characters() -> None:
    text = r'"\u0061"'

    assert (
        decode_json(
            text,
            JsonLimits(
                max_document_chars=64,
                max_depth=8,
                max_string_chars=6,
                max_tokens=64,
            ),
        )
        == "a"
    )
    with pytest.raises(JsonLimitExceeded) as raised:
        decode_json(
            text,
            JsonLimits(
                max_document_chars=64,
                max_depth=8,
                max_string_chars=5,
                max_tokens=64,
            ),
        )

    assert raised.value.code is BoundedJsonErrorCode.STRING_LIMIT
    assert raised.value.limit == 5


def test_decode_json_classifies_runtime_recursion_as_a_depth_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def recursive_load(*_args: object, **_kwargs: object) -> object:
        raise RecursionError

    monkeypatch.setattr("peftlint._bounded_json.json.loads", recursive_load)

    with pytest.raises(JsonLimitExceeded) as raised:
        decode_json("null", LIMITS)

    assert raised.value.code is BoundedJsonErrorCode.DEPTH_LIMIT
    assert raised.value.limit == LIMITS.max_depth
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        (
            {"max_document_chars": True},
            TypeError,
            "max_document_chars must be an integer",
        ),
        (
            {"max_document_chars": -1},
            ValueError,
            "max_document_chars must not be negative",
        ),
        ({"max_depth": True}, TypeError, "max_depth must be an integer"),
        ({"max_depth": -1}, ValueError, "max_depth must not be negative"),
        (
            {"max_string_chars": 1.0},
            TypeError,
            "max_string_chars must be an integer",
        ),
        (
            {"max_string_chars": -1},
            ValueError,
            "max_string_chars must not be negative",
        ),
        ({"max_tokens": None}, TypeError, "max_tokens must be an integer"),
        ({"max_tokens": -1}, ValueError, "max_tokens must not be negative"),
        (
            {"max_number_chars": False},
            TypeError,
            "max_number_chars must be an integer or None",
        ),
        (
            {"max_number_chars": -1},
            ValueError,
            "max_number_chars must not be negative",
        ),
    ],
)
def test_json_limits_validate_manual_construction(
    overrides: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    values: dict[str, object] = {
        "max_document_chars": 1_024,
        "max_depth": 8,
        "max_string_chars": 64,
        "max_tokens": 64,
        "max_number_chars": None,
    }
    values.update(overrides)

    with pytest.raises(error_type, match=f"^{message}$"):
        JsonLimits(**values)  # type: ignore[arg-type]


def test_json_limits_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        LIMITS.max_depth = 9  # type: ignore[misc]


def test_decode_json_requires_exact_input_types() -> None:
    with pytest.raises(TypeError, match=r"^bounded JSON input must be a string$"):
        decode_json(b"{}", LIMITS)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"^bounded JSON limits must be JsonLimits$"):
        decode_json("{}", object())  # type: ignore[arg-type]


def test_number_lexemes_require_exact_string_values() -> None:
    with pytest.raises(TypeError, match=r"^integer lexeme value must be a string$"):
        IntegerLexeme(1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"^float lexeme value must be a string$"):
        FloatLexeme(1.0)  # type: ignore[arg-type]


def test_bounded_json_error_categories_reject_invalid_construction() -> None:
    class ConcreteBoundedJsonError(BoundedJsonError):
        pass

    with pytest.raises(TypeError, match=r"^BoundedJsonError is an abstract base class$"):
        BoundedJsonError(BoundedJsonErrorCode.SYNTAX)
    with pytest.raises(TypeError, match=r"^bounded JSON error code must be"):
        ConcreteBoundedJsonError("syntax")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"^bounded JSON error code must be"):
        InvalidJson("syntax")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"^depth_limit is not an invalid JSON error code$"):
        InvalidJson(BoundedJsonErrorCode.DEPTH_LIMIT)
    with pytest.raises(TypeError, match=r"^bounded JSON error code must be"):
        JsonLimitExceeded("depth_limit", limit=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"^syntax is not a JSON limit error code$"):
        JsonLimitExceeded(BoundedJsonErrorCode.SYNTAX, limit=1)
    with pytest.raises(TypeError, match=r"^bounded JSON limit must be an integer$"):
        JsonLimitExceeded(BoundedJsonErrorCode.DEPTH_LIMIT, limit=True)
    with pytest.raises(ValueError, match=r"^bounded JSON limit must not be negative$"):
        JsonLimitExceeded(BoundedJsonErrorCode.DEPTH_LIMIT, limit=-1)


def test_bounded_json_errors_never_render_document_content() -> None:
    marker = "private-json-marker"

    with pytest.raises(InvalidJson) as raised:
        decode_json(f'{{"{marker}":', LIMITS)

    rendered = f"{raised.value!s} {raised.value!r} {raised.value.args!r}"
    assert marker not in rendered
