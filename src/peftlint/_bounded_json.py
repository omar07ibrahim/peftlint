"""Private duplicate-safe JSON decoding under explicit resource limits.

The decoder preserves JSON number lexemes instead of converting attacker-
controlled integers or floats. Format-specific modules remain responsible for
interpreting those lexemes and translating these private failures into their
public error taxonomies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import NoReturn, cast


class BoundedJsonErrorCode(StrEnum):
    """Stable internal reasons why bounded JSON decoding stopped."""

    SYNTAX = "syntax"
    DUPLICATE_KEY = "duplicate_key"
    DOCUMENT_LIMIT = "document_limit"
    DEPTH_LIMIT = "depth_limit"
    STRING_LIMIT = "string_limit"
    TOKEN_LIMIT = "token_limit"
    NUMBER_LIMIT = "number_limit"


_ERROR_MESSAGES = {
    BoundedJsonErrorCode.SYNTAX: "document is not valid JSON",
    BoundedJsonErrorCode.DUPLICATE_KEY: "document contains a duplicate JSON key",
    BoundedJsonErrorCode.DOCUMENT_LIMIT: "document exceeds the JSON document limit",
    BoundedJsonErrorCode.DEPTH_LIMIT: "document exceeds the JSON nesting limit",
    BoundedJsonErrorCode.STRING_LIMIT: "document exceeds the JSON string limit",
    BoundedJsonErrorCode.TOKEN_LIMIT: "document exceeds the JSON token limit",
    BoundedJsonErrorCode.NUMBER_LIMIT: "document exceeds the JSON number limit",
}

_LIMIT_CODES = frozenset(
    {
        BoundedJsonErrorCode.DOCUMENT_LIMIT,
        BoundedJsonErrorCode.DEPTH_LIMIT,
        BoundedJsonErrorCode.STRING_LIMIT,
        BoundedJsonErrorCode.TOKEN_LIMIT,
        BoundedJsonErrorCode.NUMBER_LIMIT,
    }
)


class BoundedJsonError(Exception):
    """Base class for private, content-redacted JSON failures."""

    code: BoundedJsonErrorCode

    def __init__(self, code: BoundedJsonErrorCode) -> None:
        if type(self) is BoundedJsonError:
            raise TypeError("BoundedJsonError is an abstract base class")
        if type(code) is not BoundedJsonErrorCode:
            raise TypeError("bounded JSON error code must be BoundedJsonErrorCode")
        self.code = code
        super().__init__(_ERROR_MESSAGES[code])


class InvalidJson(BoundedJsonError):
    """The supplied text contradicts the JSON grammar or key-uniqueness rule."""

    def __init__(self, code: BoundedJsonErrorCode) -> None:
        if type(code) is not BoundedJsonErrorCode:
            raise TypeError("bounded JSON error code must be BoundedJsonErrorCode")
        if code not in {
            BoundedJsonErrorCode.SYNTAX,
            BoundedJsonErrorCode.DUPLICATE_KEY,
        }:
            raise ValueError(f"{code.value} is not an invalid JSON error code")
        super().__init__(code)


class JsonLimitExceeded(BoundedJsonError):
    """Decoding stopped at a configured local resource boundary."""

    limit: int

    def __init__(self, code: BoundedJsonErrorCode, *, limit: int) -> None:
        if type(code) is not BoundedJsonErrorCode:
            raise TypeError("bounded JSON error code must be BoundedJsonErrorCode")
        if code not in _LIMIT_CODES:
            raise ValueError(f"{code.value} is not a JSON limit error code")
        if type(limit) is not int:
            raise TypeError("bounded JSON limit must be an integer")
        if limit < 0:
            raise ValueError("bounded JSON limit must not be negative")
        self.limit = limit
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class JsonLimits:
    """Limits enforced before and during ordinary JSON materialization."""

    max_document_chars: int
    max_depth: int
    max_string_chars: int
    max_tokens: int
    max_number_chars: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_document_chars", self.max_document_chars),
            ("max_depth", self.max_depth),
            ("max_string_chars", self.max_string_chars),
            ("max_tokens", self.max_tokens),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must not be negative")
        if self.max_number_chars is not None:
            if type(self.max_number_chars) is not int:
                raise TypeError("max_number_chars must be an integer or None")
            if self.max_number_chars < 0:
                raise ValueError("max_number_chars must not be negative")


@dataclass(frozen=True, slots=True, repr=False)
class IntegerLexeme:
    """An unconverted JSON integer token."""

    value: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.value) is not str:
            raise TypeError("integer lexeme value must be a string")


@dataclass(frozen=True, slots=True, repr=False)
class FloatLexeme:
    """An unconverted JSON floating-point token."""

    value: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.value) is not str:
            raise TypeError("float lexeme value must be a string")


def decode_json(text: str, limits: JsonLimits) -> object:
    """Decode one JSON value after enforcing deterministic resource limits."""

    if type(text) is not str:
        raise TypeError("bounded JSON input must be a string")
    if type(limits) is not JsonLimits:
        raise TypeError("bounded JSON limits must be JsonLimits")
    if len(text) > limits.max_document_chars:
        raise JsonLimitExceeded(
            BoundedJsonErrorCode.DOCUMENT_LIMIT,
            limit=limits.max_document_chars,
        )

    _prescan(text, limits)

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise InvalidJson(BoundedJsonErrorCode.DUPLICATE_KEY)
            result[key] = value
        return result

    def reject_constant(_value: str) -> NoReturn:
        raise InvalidJson(BoundedJsonErrorCode.SYNTAX)

    def integer_lexeme(value: str) -> IntegerLexeme:
        _check_number_length(value, limits)
        return IntegerLexeme(value)

    def float_lexeme(value: str) -> FloatLexeme:
        _check_number_length(value, limits)
        return FloatLexeme(value)

    result: object | None = None
    json_failed = False
    recursion_failed = False
    try:
        result = cast(
            object,
            json.loads(
                text,
                object_pairs_hook=unique_object,
                parse_int=integer_lexeme,
                parse_float=float_lexeme,
                parse_constant=reject_constant,
            ),
        )
    except json.JSONDecodeError:
        json_failed = True
    except RecursionError:
        recursion_failed = True

    if json_failed:
        raise InvalidJson(BoundedJsonErrorCode.SYNTAX)
    if recursion_failed:
        raise JsonLimitExceeded(
            BoundedJsonErrorCode.DEPTH_LIMIT,
            limit=limits.max_depth,
        )
    return result


def _prescan(text: str, limits: JsonLimits) -> None:
    depth = 0
    tokens = 0
    in_string = False
    in_primitive = False
    escaped = False
    string_chars = 0

    def count_token() -> None:
        nonlocal tokens
        tokens += 1
        if tokens > limits.max_tokens:
            raise JsonLimitExceeded(
                BoundedJsonErrorCode.TOKEN_LIMIT,
                limit=limits.max_tokens,
            )

    for character in text:
        if in_string:
            if character == '"' and not escaped:
                in_string = False
                continue
            string_chars += 1
            if string_chars > limits.max_string_chars:
                raise JsonLimitExceeded(
                    BoundedJsonErrorCode.STRING_LIMIT,
                    limit=limits.max_string_chars,
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
            if depth > limits.max_depth:
                raise JsonLimitExceeded(
                    BoundedJsonErrorCode.DEPTH_LIMIT,
                    limit=limits.max_depth,
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


def _check_number_length(value: str, limits: JsonLimits) -> None:
    if limits.max_number_chars is not None and len(value) > limits.max_number_chars:
        raise JsonLimitExceeded(
            BoundedJsonErrorCode.NUMBER_LIMIT,
            limit=limits.max_number_chars,
        )


__all__ = [
    "BoundedJsonError",
    "BoundedJsonErrorCode",
    "FloatLexeme",
    "IntegerLexeme",
    "InvalidJson",
    "JsonLimitExceeded",
    "JsonLimits",
    "decode_json",
]
