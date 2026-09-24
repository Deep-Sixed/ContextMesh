"""Fail-closed LLM provider adapters for Context Mesh.

The provider layer is below execution authority. A worker may return structured
output, but an LLM audit returns :class:`AuditProposal` only. It is deliberately
not a native ``Verdict``: deployment-owned code must decide whether a validated
proposal is sufficient to call ``ctx.ok``, ``ctx.fail`` or the PR #8
evidence-bound ``ctx.disproved`` path.

There is no provider auto-detection and no live-to-simulation fallback. Live
configuration names one provider, one model, and that provider's own credential.
Simulation is a separate explicit mode and requires an explicit simulator.

The core remains dependency-free: HTTP uses urllib and structured output is
validated locally after provider-native JSON-schema controls are requested.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .execute import AuditContext, RunContext
from .model import NodeType


class LLMError(Exception):
    """Base error for provider configuration, transport, or response failure."""


class LLMConfigurationError(LLMError):
    """The requested provider or mode cannot be configured safely."""


class LLMTransportError(LLMError):
    """No HTTP response was obtained from the configured provider."""


class LLMRequestTooLargeError(LLMError):
    """The assembled request body exceeds this client's own outbound limit.

    Raised before the request is ever sent: nothing here truncates a prompt
    to fit, because a worker's ``inputs`` or an audit's ``available_evidence``
    are graph content a caller reasons about, and silently dropping some of
    it would change what the model is actually asked without telling anyone.
    The one honest response to an oversized request is to refuse it and say
    so, the same way :func:`_bounded_read` refuses an oversized response
    rather than truncating what a provider sends back.
    """


class LLMProviderError(LLMError):
    """The configured provider returned a failure."""

    def __init__(self, message: str, *, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


class LLMResponseTooLargeError(LLMProviderError):
    """A successful provider response exceeded ``max_response_bytes``.

    Deliberately not an :class:`LLMTransportError`: an HTTP response *was*
    obtained, and the same request will draw the same oversized reply, so
    retrying it only re-bills the completion. It fails closed on the first
    attempt, with the status and the limit in the message.
    """


class LLMResponseError(LLMError):
    """A provider response cannot be interpreted as requested output."""


class LLMSchemaError(LLMResponseError):
    """Structured output does not satisfy the locally enforced schema."""


LIVE_PROVIDERS = ("openai", "gemini", "anthropic", "openrouter")
_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/responses",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/interactions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}
_SCHEMA_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_JSON_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}
_SCHEMA_KEYS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "description",
    "title",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
}
_RESERVED_OUTPUT_KEY = "_contextmesh_llm"


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Optional[int]]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class LLMProvenance:
    provider: str
    model: str
    mode: str
    attempts: int
    usage: TokenUsage = field(default_factory=TokenUsage)
    response_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode,
            "attempts": self.attempts,
            "usage": self.usage.to_dict(),
            "response_id": self.response_id,
        }


@dataclass(frozen=True)
class LLMResult:
    data: Dict[str, Any]
    provenance: LLMProvenance


@dataclass(frozen=True)
class HTTPRequest:
    method: str
    url: str
    headers: Dict[str, str] = field(repr=False)
    body: bytes = field(repr=False)
    timeout: float = field(repr=False)
    max_response_bytes: int = field(repr=False)


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    headers: Dict[str, str]
    body: bytes


Transport = Callable[[HTTPRequest], HTTPResponse]
Sleeper = Callable[[float], None]
Simulator = Callable[[str, Mapping[str, Any], Optional[str]], Dict[str, Any]]


@dataclass(frozen=True)
class LLMConfig:
    """One explicit provider identity; credentials are never included in repr."""

    provider: str
    model: str
    api_key: Optional[str] = field(default=None, repr=False)
    mode: str = "live"
    timeout: float = 30.0
    max_attempts: int = 3
    base_delay: float = 0.25
    max_tokens: int = 1024
    max_response_bytes: int = 1_048_576
    max_request_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider:
            raise LLMConfigurationError("provider must be a non-empty string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise LLMConfigurationError("model must be a non-empty string")
        if self.mode not in ("live", "simulation"):
            raise LLMConfigurationError("mode must be exactly 'live' or 'simulation'")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise LLMConfigurationError("timeout must be a positive number")
        if not math.isfinite(float(self.timeout)) or self.timeout <= 0:
            raise LLMConfigurationError("timeout must be a positive finite number")
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise LLMConfigurationError("max_attempts must be a positive integer")
        if self.max_attempts < 1:
            raise LLMConfigurationError("max_attempts must be a positive integer")
        if isinstance(self.base_delay, bool) or not isinstance(self.base_delay, (int, float)):
            raise LLMConfigurationError("base_delay must be a non-negative number")
        if not math.isfinite(float(self.base_delay)) or self.base_delay < 0:
            raise LLMConfigurationError("base_delay must be a non-negative finite number")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise LLMConfigurationError("max_tokens must be a positive integer")
        if self.max_tokens < 1:
            raise LLMConfigurationError("max_tokens must be a positive integer")
        if isinstance(self.max_response_bytes, bool) or not isinstance(
            self.max_response_bytes, int
        ):
            raise LLMConfigurationError("max_response_bytes must be a positive integer")
        if self.max_response_bytes < 1:
            raise LLMConfigurationError("max_response_bytes must be a positive integer")
        if isinstance(self.max_request_bytes, bool) or not isinstance(
            self.max_request_bytes, int
        ):
            raise LLMConfigurationError("max_request_bytes must be a positive integer")
        if self.max_request_bytes < 1:
            raise LLMConfigurationError("max_request_bytes must be a positive integer")

        if self.mode == "live":
            if self.provider not in LIVE_PROVIDERS:
                raise LLMConfigurationError(
                    f"live provider must be one of {', '.join(LIVE_PROVIDERS)}"
                )
            if not isinstance(self.api_key, str) or not self.api_key.strip():
                raise LLMConfigurationError(
                    f"live provider {self.provider!r} requires its own API key"
                )
        else:
            if self.provider != "simulation":
                raise LLMConfigurationError(
                    "simulation mode must use provider='simulation' so provenance "
                    "cannot pretend a live provider ran"
                )
            if self.api_key is not None:
                raise LLMConfigurationError("simulation mode must not carry an API key")

    @classmethod
    def live_from_env(
        cls,
        provider: str,
        model: str,
        *,
        environ: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> "LLMConfig":
        """Read exactly the selected provider's credential, with no fallback chain."""
        if provider not in LIVE_PROVIDERS:
            raise LLMConfigurationError(
                f"live provider must be one of {', '.join(LIVE_PROVIDERS)}"
            )
        env = os.environ if environ is None else environ
        name = _ENV_KEYS[provider]
        key = env.get(name)
        if not key:
            raise LLMConfigurationError(
                f"provider {provider!r} requires {name}; another provider's key is not used"
            )
        return cls(provider=provider, model=model, api_key=key, mode="live", **kwargs)

    @classmethod
    def simulation(cls, model: str = "deterministic-v1", **kwargs: Any) -> "LLMConfig":
        return cls(provider="simulation", model=model, mode="simulation", **kwargs)


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect out of a provider request.

    ``urllib`` copies a request's headers onto the redirected request and strips
    only the content headers, so following a 30x hands ``Authorization`` and
    ``x-api-key`` to whatever host the response names — including a different
    one — and returns 200 as though nothing happened. Nothing else in this
    module lets the credential out: it is ``repr=False`` on the config, on the
    request headers and on the body, and transport failures name no header.

    Provider endpoints are a fixed HTTPS allowlist (``_ENDPOINTS``), so there is
    no legitimate redirect to follow and no sanitising to get right. Returning
    ``None`` here leaves the 30x unhandled, which surfaces it as an ordinary
    non-2xx response and fails the request closed with its status intact.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def _bounded_read(readable: Any, max_bytes: int) -> Optional[bytes]:
    """Read at most ``max_bytes`` and refuse whatever comes after.

    ``.read()`` with no argument buffers the whole body regardless of how
    large the provider's reply turns out to be, so a provider or an
    intermediary between us and it can force an unbounded allocation just by
    sending an unbounded response. Reading ``max_bytes + 1`` costs one extra
    byte and tells us, without ever holding more than the limit plus one in
    memory, whether the body would have exceeded it.
    """
    data = readable.read(max_bytes + 1)
    if len(data) > max_bytes:
        # What an oversized body means depends on the status, so the caller
        # decides; this only guarantees nothing past limit + 1 is buffered.
        return None
    return data


class UrllibTransport:
    """Small HTTP transport whose failures expose no request headers or body."""

    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(_RefuseRedirects())

    def __call__(self, request: HTTPRequest) -> HTTPResponse:
        raw = urllib.request.Request(
            request.url,
            data=request.body,
            headers=request.headers,
            method=request.method,
        )
        try:
            with self._opener.open(raw, timeout=request.timeout) as response:
                status = int(response.status)
                body = _bounded_read(response, request.max_response_bytes)
                if body is None:
                    raise LLMResponseTooLargeError(
                        f"provider response (HTTP {status}) exceeded the "
                        f"{request.max_response_bytes}-byte limit; refusing to buffer it",
                        status=status,
                    )
                return HTTPResponse(
                    status=status,
                    headers={str(k): str(v) for k, v in response.headers.items()},
                    body=body,
                )
        except urllib.error.HTTPError as exc:
            # Nothing reads an error body -- the client acts on the status
            # alone -- so an oversized one is dropped after the bounded read
            # and the status still decides: a 4xx fails closed at once, a
            # 429/5xx is retried. Raising here turned every oversized error
            # page into a retried "transport" failure that hid the status.
            body = _bounded_read(exc, request.max_response_bytes) or b""
            headers = {str(k): str(v) for k, v in exc.headers.items()} if exc.headers else {}
            return HTTPResponse(status=int(exc.code), headers=headers, body=body)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMTransportError("provider request did not return an HTTP response") from exc


def _reject_constant(value: str) -> Any:
    raise LLMResponseError(f"JSON contains non-finite constant {value!r}")


def _strict_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LLMResponseError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(text: str, *, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except LLMResponseError:
        raise
    except (TypeError, ValueError) as exc:
        raise LLMResponseError(f"{label} is not valid JSON") from exc


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LLMConfigurationError("request contains a value JSON cannot represent") from exc


@dataclass
class _JsonArray:
    """An internal lazy array, so auditing need not copy all evidence first."""

    items: Iterable[Any]


def _bounded_json_bytes(value: Any, limit: int, *, sort_keys: bool = False) -> bytes:
    """Encode compact UTF-8 JSON with storage bounded by the byte budget.

    JSONEncoder.iterencode alone is insufficient: it encodes a whole string
    into a single chunk. Strings here are escaped in small pieces, containers
    are visited incrementally, and the output never grows beyond ``limit``.
    """
    output = bytearray()
    active = set()

    def too_large() -> None:
        raise LLMRequestTooLargeError(f"request exceeds the {limit}-byte limit")

    def emit(chunk: bytes) -> None:
        if len(chunk) > limit - len(output):
            too_large()
        output.extend(chunk)

    def string(text: str) -> None:
        # Every code point needs at least one byte, plus the quotes. Check
        # before slicing, escaping, or encoding a possibly enormous value.
        if len(text) + 2 > limit - len(output):
            too_large()
        emit(b'"')
        for start in range(0, len(text), 1024):
            chunk = json.dumps(text[start:start + 1024], ensure_ascii=False)[1:-1]
            emit(chunk.encode("utf-8"))
        emit(b'"')

    def scalar(item: Any) -> str:
        if isinstance(item, int) and not isinstance(item, bool):
            # 2**4 > 10: this conservative lower bound avoids converting an
            # enormous integer to decimal just to discover it cannot fit.
            if item.bit_length() // 4 > limit - len(output):
                too_large()
        return json.dumps(item, ensure_ascii=False, allow_nan=False)

    def encode(item: Any) -> None:
        if isinstance(item, str):
            string(item)
        elif item is None or isinstance(item, (bool, int, float)):
            emit(scalar(item).encode("utf-8"))
        elif isinstance(item, (dict, list, tuple, _JsonArray)):
            identity = id(item)
            if identity in active:
                raise ValueError("circular reference")
            active.add(identity)
            try:
                # Each member requires at least one byte. In particular this
                # bounds the temporary key list needed for deterministic sort.
                if not isinstance(item, _JsonArray) and len(item) > limit - len(output):
                    too_large()
                if isinstance(item, dict):
                    emit(b"{")
                    keys = sorted(item) if sort_keys else item
                    for index, key in enumerate(keys):
                        if index:
                            emit(b",")
                        if not isinstance(key, str):
                            if key is not None and not isinstance(key, (bool, int, float)):
                                raise TypeError("unsupported JSON key")
                            key_text = scalar(key)
                        else:
                            key_text = key
                        string(key_text)
                        emit(b":")
                        encode(item[key])
                    emit(b"}")
                else:
                    emit(b"[")
                    items = item.items if isinstance(item, _JsonArray) else item
                    for index, member in enumerate(items):
                        if index:
                            emit(b",")
                        encode(member)
                    emit(b"]")
            finally:
                active.remove(identity)
        else:
            raise TypeError("unsupported JSON value")

    try:
        encode(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise LLMConfigurationError("request contains a value JSON cannot represent") from exc
    return bytes(output)


def _bounded_prompt(prefix: str, payload: Any, limit: int) -> str:
    if len(prefix) > limit:
        raise LLMRequestTooLargeError(f"request exceeds the {limit}-byte limit")
    prefix_bytes = prefix.encode("utf-8")
    if len(prefix_bytes) > limit:
        raise LLMRequestTooLargeError(f"request exceeds the {limit}-byte limit")
    try:
        body = _bounded_json_bytes(payload, limit - len(prefix_bytes), sort_keys=True)
    except LLMRequestTooLargeError:
        raise LLMRequestTooLargeError(f"request exceeds the {limit}-byte limit") from None
    return prefix + body.decode("utf-8")


def _type_list(schema: Mapping[str, Any], path: str) -> List[str]:
    declared = schema.get("type")
    if isinstance(declared, str):
        values = [declared]
    elif isinstance(declared, list) and declared:
        if not all(isinstance(value, str) for value in declared):
            raise LLMConfigurationError(f"{path}.type must contain only strings")
        if len(set(declared)) != len(declared):
            raise LLMConfigurationError(f"{path}.type contains duplicate types")
        values = list(declared)
    else:
        raise LLMConfigurationError(f"{path}.type must be a string or non-empty list")
    unknown = [value for value in values if value not in _JSON_TYPES]
    if unknown:
        raise LLMConfigurationError(f"{path}.type contains unsupported JSON types {unknown!r}")
    return values


def validate_schema_definition(schema: Mapping[str, Any], path: str = "$schema") -> None:
    """Accept one conservative strict-schema subset shared by all providers."""
    if not isinstance(schema, Mapping):
        raise LLMConfigurationError(f"{path} must be an object")
    unknown = sorted(set(schema) - _SCHEMA_KEYS)
    if unknown:
        raise LLMConfigurationError(
            f"{path} uses unsupported schema keyword(s): {', '.join(unknown)}"
        )
    types = _type_list(schema, path)

    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, list) or not enum:
            raise LLMConfigurationError(f"{path}.enum must be a non-empty array")
        _json_bytes(enum)

    if "object" in types:
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise LLMConfigurationError(f"{path}.properties must be an object")
        for name, subschema in properties.items():
            if not isinstance(name, str):
                raise LLMConfigurationError(f"{path}.properties keys must be strings")
            validate_schema_definition(subschema, f"{path}.properties[{name!r}]")
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(x, str) for x in required):
            raise LLMConfigurationError(f"{path}.required must be an array of strings")
        if len(set(required)) != len(required):
            raise LLMConfigurationError(f"{path}.required contains duplicates")
        missing = sorted(set(required) - set(properties))
        if missing:
            raise LLMConfigurationError(
                f"{path}.required names undefined properties: {', '.join(missing)}"
            )
        if schema.get("additionalProperties") is not False:
            raise LLMConfigurationError(
                f"{path}.additionalProperties must be false for portable strict output"
            )
        optional = sorted(set(properties) - set(required))
        if optional:
            raise LLMConfigurationError(
                f"{path} must require every property for portable strict output: "
                + ", ".join(optional)
            )
    elif any(key in schema for key in ("properties", "required", "additionalProperties")):
        raise LLMConfigurationError(f"{path} has object-only keywords without object type")

    if "array" in types:
        if "items" not in schema:
            raise LLMConfigurationError(f"{path}.items is required for array output")
        validate_schema_definition(schema["items"], f"{path}.items")
    elif "items" in schema:
        raise LLMConfigurationError(f"{path}.items is only valid for an array")

    for key in ("minLength", "maxLength", "minItems", "maxItems"):
        if key in schema:
            value = schema[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LLMConfigurationError(f"{path}.{key} must be a non-negative integer")
    if "minLength" in schema and "string" not in types:
        raise LLMConfigurationError(f"{path}.minLength requires string type")
    if "maxLength" in schema and "string" not in types:
        raise LLMConfigurationError(f"{path}.maxLength requires string type")
    if "minItems" in schema and "array" not in types:
        raise LLMConfigurationError(f"{path}.minItems requires array type")
    if "maxItems" in schema and "array" not in types:
        raise LLMConfigurationError(f"{path}.maxItems requires array type")

    for key in ("minimum", "maximum"):
        if key in schema:
            value = schema[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise LLMConfigurationError(f"{path}.{key} must be a finite number")
            if not math.isfinite(float(value)):
                raise LLMConfigurationError(f"{path}.{key} must be a finite number")
            if "number" not in types and "integer" not in types:
                raise LLMConfigurationError(f"{path}.{key} requires numeric type")


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return False


def _same_json_scalar(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if left is None or right is None:
        return left is right
    return type(left) is type(right) and left == right


def validate_instance(value: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    types = _type_list(schema, path)
    if not any(_matches_type(value, expected) for expected in types):
        raise LLMSchemaError(f"{path} does not match declared type {types!r}")

    if "enum" in schema and not any(_same_json_scalar(value, item) for item in schema["enum"]):
        raise LLMSchemaError(f"{path} is not one of the allowed enum values")

    if isinstance(value, dict) and "object" in types:
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [name for name in required if name not in value]
        if missing:
            raise LLMSchemaError(f"{path} is missing required field(s): {', '.join(missing)}")
        extra = sorted(set(value) - set(properties))
        if extra:
            raise LLMSchemaError(f"{path} contains unexpected field(s): {', '.join(extra)}")
        for name, item in value.items():
            validate_instance(item, properties[name], f"{path}.{name}")

    if isinstance(value, list) and "array" in types:
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if minimum is not None and len(value) < minimum:
            raise LLMSchemaError(f"{path} has fewer than {minimum} items")
        if maximum is not None and len(value) > maximum:
            raise LLMSchemaError(f"{path} has more than {maximum} items")
        for index, item in enumerate(value):
            validate_instance(item, schema["items"], f"{path}[{index}]")

    if isinstance(value, str) and "string" in types:
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise LLMSchemaError(f"{path} is shorter than minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise LLMSchemaError(f"{path} is longer than maxLength")

    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and ("number" in types or "integer" in types)
    ):
        if isinstance(value, float) and not math.isfinite(value):
            raise LLMSchemaError(f"{path} contains a non-finite number")
        if "minimum" in schema and value < schema["minimum"]:
            raise LLMSchemaError(f"{path} is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise LLMSchemaError(f"{path} is above maximum")


def _schema_name(value: str) -> str:
    if not isinstance(value, str) or _SCHEMA_NAME.fullmatch(value) is None:
        raise LLMConfigurationError(
            "schema_name must contain 1-64 letters, digits, underscores, or hyphens"
        )
    return value


def _int_or_none(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _usage_three(
    value: Any, input_name: str, output_name: str, total_name: Optional[str]
) -> TokenUsage:
    if not isinstance(value, Mapping):
        return TokenUsage()
    input_tokens = _int_or_none(value.get(input_name))
    output_tokens = _int_or_none(value.get(output_name))
    total_tokens = _int_or_none(value.get(total_name)) if total_name else None
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return TokenUsage(input_tokens, output_tokens, total_tokens)


def _response_id(envelope: Mapping[str, Any]) -> Optional[str]:
    value = envelope.get("id")
    return value if isinstance(value, str) and value else None


def _decode_body(response: HTTPResponse) -> Dict[str, Any]:
    try:
        text = response.body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LLMResponseError("provider response is not UTF-8") from exc
    value = _strict_json_loads(text, label="provider response")
    if not isinstance(value, dict):
        raise LLMResponseError("provider response must be a JSON object")
    return value


def _extract_openai(envelope: Mapping[str, Any]) -> Tuple[str, TokenUsage, Optional[str]]:
    if envelope.get("status") != "completed":
        raise LLMResponseError("OpenAI response did not complete successfully")
    output = envelope.get("output")
    if not isinstance(output, list):
        raise LLMResponseError("OpenAI response is missing output")
    texts: List[str] = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping):
                continue
            if part.get("type") == "refusal":
                raise LLMResponseError("OpenAI refused the structured request")
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
    if not texts:
        raise LLMResponseError("OpenAI response contains no output_text")
    return (
        "".join(texts),
        _usage_three(envelope.get("usage"), "input_tokens", "output_tokens", "total_tokens"),
        _response_id(envelope),
    )


def _extract_gemini(envelope: Mapping[str, Any]) -> Tuple[str, TokenUsage, Optional[str]]:
    if envelope.get("status") != "completed":
        raise LLMResponseError("Gemini interaction did not complete successfully")
    steps = envelope.get("steps")
    if not isinstance(steps, list):
        raise LLMResponseError("Gemini interaction is missing steps")
    texts: List[str] = []
    for step in steps:
        if not isinstance(step, Mapping) or step.get("type") != "model_output":
            continue
        content = step.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if (
                isinstance(part, Mapping)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ):
                texts.append(part["text"])
    if not texts:
        raise LLMResponseError("Gemini interaction contains no model text output")
    return (
        "".join(texts),
        _usage_three(
            envelope.get("usage"),
            "total_input_tokens",
            "total_output_tokens",
            "total_tokens",
        ),
        _response_id(envelope),
    )


def _extract_anthropic(envelope: Mapping[str, Any]) -> Tuple[str, TokenUsage, Optional[str]]:
    if envelope.get("stop_reason") in ("max_tokens", "model_context_window_exceeded"):
        raise LLMResponseError("Anthropic response ended before structured output completed")
    content = envelope.get("content")
    if not isinstance(content, list):
        raise LLMResponseError("Anthropic response is missing content")
    texts = [
        part["text"]
        for part in content
        if isinstance(part, Mapping)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
    ]
    if not texts:
        raise LLMResponseError("Anthropic response contains no text block")
    return (
        "".join(texts),
        _usage_three(envelope.get("usage"), "input_tokens", "output_tokens", None),
        _response_id(envelope),
    )


def _extract_openrouter(envelope: Mapping[str, Any]) -> Tuple[str, TokenUsage, Optional[str]]:
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMResponseError("OpenRouter response is missing choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise LLMResponseError("OpenRouter first choice is malformed")
    if first.get("finish_reason") in ("length", "content_filter"):
        raise LLMResponseError("OpenRouter response ended before structured output completed")
    message = first.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise LLMResponseError("OpenRouter response is missing message content")
    return (
        message["content"],
        _usage_three(envelope.get("usage"), "prompt_tokens", "completion_tokens", "total_tokens"),
        _response_id(envelope),
    )


_EXTRACTORS = {
    "openai": _extract_openai,
    "gemini": _extract_gemini,
    "anthropic": _extract_anthropic,
    "openrouter": _extract_openrouter,
}


def _request_for(
    config: LLMConfig,
    prompt: str,
    schema: Mapping[str, Any],
    schema_name: str,
    system: Optional[str],
) -> HTTPRequest:
    provider = config.provider
    assert provider in LIVE_PROVIDERS
    key = config.api_key
    assert key is not None

    if provider == "openai":
        body: Dict[str, Any] = {
            "model": config.model,
            "input": prompt,
            "max_output_tokens": config.max_tokens,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        if system is not None:
            body["instructions"] = system
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    elif provider == "gemini":
        body = {
            "model": config.model,
            "input": prompt,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": schema,
            },
            "generation_config": {"max_output_tokens": config.max_tokens},
            "store": False,
        }
        if system is not None:
            body["system_instruction"] = system
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    elif provider == "anthropic":
        body = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        if system is not None:
            body["system"] = system
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    else:
        messages: List[Dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": config.model,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
            "provider": {"require_parameters": True},
        }
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    return HTTPRequest(
        method="POST",
        url=_ENDPOINTS[provider],
        headers=headers,
        body=_bounded_json_bytes(body, config.max_request_bytes),
        timeout=float(config.timeout),
        max_response_bytes=config.max_response_bytes,
    )


class LLMClient:
    """One provider client with bounded retries and no fallback provider."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: Optional[Transport] = None,
        sleep: Sleeper = time.sleep,
        simulator: Optional[Simulator] = None,
    ) -> None:
        self.config = config
        self.transport = transport or UrllibTransport()
        self.sleep = sleep
        self.simulator = simulator
        if config.mode == "simulation" and simulator is None:
            raise LLMConfigurationError(
                "simulation mode requires an explicit simulator callback; "
                "there is no fake-success default"
            )
        if config.mode == "live" and simulator is not None:
            raise LLMConfigurationError(
                "a live client may not carry a simulator callback; provider failure never falls back"
            )

    def _send(self, request: HTTPRequest) -> Tuple[HTTPResponse, int]:
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = self.transport(request)
            except (LLMTransportError, TimeoutError, OSError):
                if attempt == self.config.max_attempts:
                    raise LLMProviderError(
                        f"{self.config.provider} transport failed after {attempt} attempt(s)"
                    ) from None
                self.sleep(float(self.config.base_delay) * (2 ** (attempt - 1)))
                continue

            if 200 <= response.status < 300:
                return response, attempt
            transient = response.status == 429 or 500 <= response.status <= 599
            if not transient:
                raise LLMProviderError(
                    f"{self.config.provider} returned HTTP {response.status}; request failed closed",
                    status=response.status,
                )
            if attempt == self.config.max_attempts:
                raise LLMProviderError(
                    f"{self.config.provider} returned HTTP {response.status} after "
                    f"{attempt} attempt(s)",
                    status=response.status,
                )
            self.sleep(float(self.config.base_delay) * (2 ** (attempt - 1)))
        raise AssertionError("unreachable retry loop")

    def complete(
        self,
        prompt: str,
        schema: Mapping[str, Any],
        *,
        schema_name: str = "contextmesh_output",
        system: Optional[str] = None,
    ) -> LLMResult:
        if not isinstance(prompt, str) or not prompt or prompt.isspace():
            raise LLMConfigurationError("prompt must be a non-empty string")
        if system is not None and (not isinstance(system, str) or not system or system.isspace()):
            raise LLMConfigurationError("system must be a non-empty string or null")
        # Before schema validation (which serializes enum values), bound all
        # caller-controlled input. Provider envelope overhead is checked below.
        _bounded_json_bytes([prompt, schema, system, schema_name], self.config.max_request_bytes)
        validate_schema_definition(schema)
        schema_name = _schema_name(schema_name)

        if self.config.mode == "simulation":
            assert self.simulator is not None
            value = self.simulator(prompt, schema, system)
            if not isinstance(value, dict):
                raise LLMSchemaError("simulation output must be a JSON object")
            _json_bytes(value)
            validate_instance(value, schema)
            return LLMResult(
                data=dict(value),
                provenance=LLMProvenance(
                    provider="simulation",
                    model=self.config.model,
                    mode="simulation",
                    attempts=1,
                ),
            )

        request = _request_for(self.config, prompt, schema, schema_name, system)
        response, attempts = self._send(request)
        envelope = _decode_body(response)
        text, usage, response_id = _EXTRACTORS[self.config.provider](envelope)
        value = _strict_json_loads(text, label="structured model output")
        if not isinstance(value, dict):
            raise LLMSchemaError("structured model output must be a JSON object")
        validate_instance(value, schema)
        return LLMResult(
            data=value,
            provenance=LLMProvenance(
                provider=self.config.provider,
                model=self.config.model,
                mode="live",
                attempts=attempts,
                usage=usage,
                response_id=response_id,
            ),
        )


AUDIT_PROPOSAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["success", "fail", "disproved"]},
        "reason": {"type": "string", "minLength": 1},
        "evidence_id": {"type": ["string", "null"]},
    },
    "required": ["status", "reason", "evidence_id"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class AuditProposal:
    """An LLM interpretation, deliberately not an execution ``Verdict``."""

    status: str
    reason: str
    evidence_id: Optional[str]
    provenance: LLMProvenance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "evidence_id": self.evidence_id,
            "provenance": self.provenance.to_dict(),
        }


def validate_audit_proposal(context: AuditContext, proposal: AuditProposal) -> AuditProposal:
    """Validate evidence binding without converting interpretation into authority."""
    if proposal.status not in ("success", "fail", "disproved"):
        raise LLMSchemaError(f"unknown audit proposal status {proposal.status!r}")
    if not isinstance(proposal.reason, str) or not proposal.reason.strip():
        raise LLMSchemaError("audit proposal reason must be a non-empty string")
    if proposal.status != "disproved":
        if proposal.evidence_id is not None:
            raise LLMSchemaError(
                "only a disproved proposal may identify evidence; no hidden verdict channel"
            )
        return proposal

    evidence_id = proposal.evidence_id
    if not isinstance(evidence_id, str) or not evidence_id:
        raise LLMSchemaError("a disproved proposal must identify pre-ingested evidence")
    node = context.graph.nodes.get(evidence_id)
    if node is None:
        raise LLMSchemaError(f"audit proposal evidence {evidence_id!r} is not in the graph")
    if node.type is not NodeType.EVIDENCE:
        raise LLMSchemaError(f"audit proposal evidence {evidence_id!r} is not an evidence node")
    if node.invalidated:
        raise LLMSchemaError(f"audit proposal evidence {evidence_id!r} is invalidated")
    return proposal


def _audit_prompt(context: AuditContext, instruction: str, limit: int) -> str:
    def evidence() -> Iterable[Dict[str, Any]]:
        for node in context.graph.nodes.values():
            if node.type is not NodeType.EVIDENCE or node.invalidated:
                continue
            yield {
                "id": node.id,
                "label": node.label,
                "attrs": node.attrs,
                "source_id": node.provenance.source_id if node.provenance else None,
            }
    payload = {
        "task": context.task.name,
        "assumption": {"id": context.assumption.id, "statement": context.assumption.statement},
        "output": context.output,
        "available_evidence": _JsonArray(evidence()),
    }
    if len(instruction) > limit:
        raise LLMRequestTooLargeError(f"request exceeds the {limit}-byte limit")
    return _bounded_prompt(instruction + "\n\nAudit input:\n", payload, limit)


def propose_audit(client: LLMClient, context: AuditContext, *, instruction: str) -> AuditProposal:
    """Ask an LLM to interpret one audit without granting mutation authority."""
    if not isinstance(instruction, str) or not instruction or instruction.isspace():
        raise LLMConfigurationError("audit instruction must be a non-empty string")
    result = client.complete(
        _audit_prompt(context, instruction, client.config.max_request_bytes),
        AUDIT_PROPOSAL_SCHEMA,
        schema_name="contextmesh_audit_proposal",
        system=(
            "Return only the requested audit proposal. Cite only an evidence_id that appears "
            "in available_evidence. You are proposing an interpretation, not changing graph state."
        ),
    )
    proposal = AuditProposal(
        status=result.data["status"],
        reason=result.data["reason"],
        evidence_id=result.data["evidence_id"],
        provenance=result.provenance,
    )
    return validate_audit_proposal(context, proposal)


def make_audit_proposer(
    client: LLMClient,
    instruction: str,
) -> Callable[[AuditContext], AuditProposal]:
    """Build a proposal callable; it is intentionally not a native Auditor."""

    def proposer(context: AuditContext) -> AuditProposal:
        return propose_audit(client, context, instruction=instruction)

    return proposer


def make_worker(
    client: LLMClient,
    instruction: str,
    output_schema: Mapping[str, Any],
    *,
    schema_name: str = "contextmesh_worker_output",
) -> Callable[[RunContext], Dict[str, Any]]:
    """Build a Runner worker whose durable output carries provider provenance."""
    if not isinstance(instruction, str) or not instruction or instruction.isspace():
        raise LLMConfigurationError("worker instruction must be a non-empty string")
    _bounded_json_bytes(
        [instruction, output_schema, schema_name], client.config.max_request_bytes
    )
    validate_schema_definition(output_schema)
    if "object" not in _type_list(output_schema, "$schema"):
        raise LLMConfigurationError("worker output schema must permit an object")
    properties = output_schema.get("properties", {})
    if _RESERVED_OUTPUT_KEY in properties:
        raise LLMConfigurationError(
            f"worker output schema may not define reserved key {_RESERVED_OUTPUT_KEY!r}"
        )
    schema_name = _schema_name(schema_name)

    def worker(context: RunContext) -> Dict[str, Any]:
        payload = {
            "task": context.task.name,
            "attempt": context.attempt,
            "assumption": {
                "id": context.assumption.id,
                "statement": context.assumption.statement,
            },
            "inputs": context.inputs,
        }
        prompt = _bounded_prompt("Task input:\n", payload, client.config.max_request_bytes)
        result = client.complete(
            prompt,
            output_schema,
            schema_name=schema_name,
            system=instruction,
        )
        output = dict(result.data)
        output[_RESERVED_OUTPUT_KEY] = result.provenance.to_dict()
        return output

    return worker


__all__ = [
    "AUDIT_PROPOSAL_SCHEMA",
    "AuditProposal",
    "HTTPResponse",
    "HTTPRequest",
    "LLMClient",
    "LLMConfig",
    "LLMConfigurationError",
    "LLMError",
    "LLMProviderError",
    "LLMProvenance",
    "LLMRequestTooLargeError",
    "LLMResponseTooLargeError",
    "LLMResponseError",
    "LLMResult",
    "LLMSchemaError",
    "LLMTransportError",
    "LIVE_PROVIDERS",
    "TokenUsage",
    "UrllibTransport",
    "make_audit_proposer",
    "make_worker",
    "propose_audit",
    "validate_audit_proposal",
    "validate_instance",
    "validate_schema_definition",
]
