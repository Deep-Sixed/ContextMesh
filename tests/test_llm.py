"""PR #9: native LLM provider adapters stay below execution authority."""

from __future__ import annotations

import json
import threading
import tracemalloc
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List
from unittest.mock import patch

from contextmesh.evidence import submit_evidence
from contextmesh.execute import AuditContext, RunContext, Runner, Verdict
from contextmesh.llm import (
    AuditProposal,
    HTTPRequest,
    HTTPResponse,
    LLMClient,
    LLMConfig,
    LLMConfigurationError,
    LLMProviderError,
    LLMRequestTooLargeError,
    LLMResponseTooLargeError,
    LLMResponseError,
    LLMSchemaError,
    LLMTransportError,
    UrllibTransport,
    make_audit_proposer,
    make_worker,
    propose_audit,
    validate_instance,
    validate_schema_definition,
)
from contextmesh.model import AssumptionStatus, NodeType
from contextmesh.llm import _bounded_json_bytes, _request_for


OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "value": {"type": "string", "minLength": 1},
        "count": {"type": "integer", "minimum": 0},
    },
    "required": ["value", "count"],
    "additionalProperties": False,
}


class QueueTransport:
    def __init__(self, *items: Any) -> None:
        self.items = list(items)
        self.requests: List[HTTPRequest] = []

    def __call__(self, request: HTTPRequest) -> HTTPResponse:
        self.requests.append(request)
        if not self.items:
            raise AssertionError("fake transport ran out of responses")
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, HTTPResponse)
        return item


def _http(envelope: Any, status: int = 200) -> HTTPResponse:
    return HTTPResponse(
        status=status,
        headers={"content-type": "application/json"},
        body=json.dumps(envelope, separators=(",", ":")).encode("utf-8"),
    )


def _structured(provider: str, text: str, *, response_id: str = "response-1") -> HTTPResponse:
    if provider == "openai":
        return _http(
            {
                "id": response_id,
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": text}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
            }
        )
    if provider == "gemini":
        return _http(
            {
                "id": response_id,
                "status": "completed",
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": text}],
                    }
                ],
                "usage": {
                    "total_input_tokens": 3,
                    "total_output_tokens": 4,
                    "total_tokens": 7,
                },
            }
        )
    if provider == "anthropic":
        return _http(
            {
                "id": response_id,
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": 3, "output_tokens": 4},
            }
        )
    if provider == "openrouter":
        return _http(
            {
                "id": response_id,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": text},
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            }
        )
    raise AssertionError(provider)


def _live(provider: str, transport: QueueTransport, **kwargs: Any) -> LLMClient:
    return LLMClient(
        LLMConfig(
            provider=provider,
            model=f"{provider}-model",
            api_key=f"secret-{provider}",
            base_delay=0.01,
            **kwargs,
        ),
        transport=transport,
        sleep=lambda _seconds: None,
    )


class ConfigurationTest(unittest.TestCase):
    def test_live_configuration_reads_only_the_selected_provider_key(self) -> None:
        with self.assertRaisesRegex(LLMConfigurationError, "OPENAI_API_KEY"):
            LLMConfig.live_from_env(
                "openai",
                "model",
                environ={"GEMINI_API_KEY": "wrong-provider-key"},
            )
        config = LLMConfig.live_from_env(
            "openai",
            "model",
            environ={"OPENAI_API_KEY": "right-key", "GEMINI_API_KEY": "other-key"},
        )
        self.assertEqual(config.api_key, "right-key")

    def test_credentials_are_not_exposed_by_repr(self) -> None:
        config = LLMConfig(provider="openai", model="model", api_key="super-secret")
        self.assertNotIn("super-secret", repr(config))

    def test_live_mode_has_no_simulation_fallback(self) -> None:
        config = LLMConfig(provider="openai", model="model", api_key="key")
        with self.assertRaisesRegex(LLMConfigurationError, "never falls back"):
            LLMClient(config, simulator=lambda _p, _s, _sys: {"value": "x", "count": 1})

    def test_simulation_is_explicit_and_needs_a_callback(self) -> None:
        config = LLMConfig.simulation()
        with self.assertRaisesRegex(LLMConfigurationError, "explicit simulator"):
            LLMClient(config)
        with self.assertRaisesRegex(LLMConfigurationError, "provider='simulation'"):
            LLMConfig(provider="openai", model="model", mode="simulation")

    def test_bad_numeric_configuration_fails_closed(self) -> None:
        for field, value in (
            ("timeout", 0),
            ("timeout", float("nan")),
            ("max_attempts", True),
            ("max_attempts", 0),
            ("base_delay", -1),
            ("max_tokens", 0),
            ("max_response_bytes", 0),
            ("max_response_bytes", True),
            ("max_request_bytes", 0),
            ("max_request_bytes", True),
        ):
            with self.subTest(field=field, value=value):
                args = {field: value}
                with self.assertRaises(LLMConfigurationError):
                    LLMConfig(provider="openai", model="model", api_key="key", **args)


class SchemaTest(unittest.TestCase):
    def test_portable_schema_accepts_the_contract_subset(self) -> None:
        validate_schema_definition(OUTPUT_SCHEMA)
        validate_instance({"value": "ok", "count": 2}, OUTPUT_SCHEMA)

    def test_schema_requires_closed_objects_and_all_properties_required(self) -> None:
        open_schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }
        with self.assertRaisesRegex(LLMConfigurationError, "additionalProperties"):
            validate_schema_definition(open_schema)

        optional_schema = {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "optional": {"type": "string"},
            },
            "required": ["value"],
            "additionalProperties": False,
        }
        with self.assertRaisesRegex(LLMConfigurationError, "require every property"):
            validate_schema_definition(optional_schema)

    def test_unknown_schema_keyword_is_refused(self) -> None:
        schema = {"type": "string", "pattern": "^x$"}
        with self.assertRaisesRegex(LLMConfigurationError, "unsupported schema keyword"):
            validate_schema_definition(schema)

    def test_instance_validation_does_not_coerce(self) -> None:
        with self.assertRaises(LLMSchemaError):
            validate_instance({"value": "ok", "count": True}, OUTPUT_SCHEMA)
        with self.assertRaises(LLMSchemaError):
            validate_instance({"value": "ok", "count": "2"}, OUTPUT_SCHEMA)
        with self.assertRaises(LLMSchemaError):
            validate_instance({"value": "ok", "count": 2, "extra": 1}, OUTPUT_SCHEMA)


class ProviderContractTest(unittest.TestCase):
    def test_each_provider_uses_its_native_structured_output_contract(self) -> None:
        expected_urls = {
            "openai": "https://api.openai.com/v1/responses",
            "gemini": "https://generativelanguage.googleapis.com/v1beta/interactions",
            "anthropic": "https://api.anthropic.com/v1/messages",
            "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        }
        text = json.dumps({"value": "ok", "count": 2})
        for provider in expected_urls:
            with self.subTest(provider=provider):
                transport = QueueTransport(_structured(provider, text))
                client = _live(provider, transport)
                result = client.complete(
                    "do the work",
                    OUTPUT_SCHEMA,
                    schema_name="contract_output",
                    system="system rule",
                )
                self.assertEqual(result.data, {"value": "ok", "count": 2})
                self.assertEqual(result.provenance.provider, provider)
                self.assertEqual(result.provenance.model, f"{provider}-model")
                self.assertEqual(result.provenance.mode, "live")
                self.assertEqual(result.provenance.attempts, 1)
                self.assertEqual(result.provenance.usage.total_tokens, 7)
                self.assertEqual(result.provenance.response_id, "response-1")

                request = transport.requests[0]
                self.assertEqual(request.method, "POST")
                self.assertEqual(request.url, expected_urls[provider])
                body = json.loads(request.body.decode("utf-8"))
                self.assertEqual(body["model"], f"{provider}-model")
                self._assert_native_request(provider, request, body)

    def _assert_native_request(
        self,
        provider: str,
        request: HTTPRequest,
        body: Dict[str, Any],
    ) -> None:
        if provider == "openai":
            self.assertEqual(request.headers["Authorization"], "Bearer secret-openai")
            format_ = body["text"]["format"]
            self.assertEqual(format_["type"], "json_schema")
            self.assertEqual(format_["name"], "contract_output")
            self.assertIs(format_["strict"], True)
            self.assertEqual(format_["schema"], OUTPUT_SCHEMA)
            self.assertEqual(body["instructions"], "system rule")
            self.assertIs(body["store"], False)
            return
        if provider == "gemini":
            self.assertEqual(request.headers["x-goog-api-key"], "secret-gemini")
            format_ = body["response_format"]
            self.assertEqual(format_["type"], "text")
            self.assertEqual(format_["mime_type"], "application/json")
            self.assertEqual(format_["schema"], OUTPUT_SCHEMA)
            self.assertEqual(body["system_instruction"], "system rule")
            self.assertIs(body["store"], False)
            return
        if provider == "anthropic":
            self.assertEqual(request.headers["x-api-key"], "secret-anthropic")
            self.assertEqual(request.headers["anthropic-version"], "2023-06-01")
            format_ = body["output_config"]["format"]
            self.assertEqual(format_["type"], "json_schema")
            self.assertEqual(format_["schema"], OUTPUT_SCHEMA)
            self.assertEqual(body["system"], "system rule")
            return
        self.assertEqual(request.headers["Authorization"], "Bearer secret-openrouter")
        format_ = body["response_format"]
        self.assertEqual(format_["type"], "json_schema")
        self.assertEqual(format_["json_schema"]["name"], "contract_output")
        self.assertIs(format_["json_schema"]["strict"], True)
        self.assertEqual(format_["json_schema"]["schema"], OUTPUT_SCHEMA)
        self.assertIs(body["provider"]["require_parameters"], True)
        self.assertEqual(body["messages"][0], {"role": "system", "content": "system rule"})

    def test_simulation_never_calls_the_http_transport(self) -> None:
        transport = QueueTransport(AssertionError("network path was reached"))
        client = LLMClient(
            LLMConfig.simulation(),
            transport=transport,
            simulator=lambda _prompt, _schema, _system: {"value": "sim", "count": 1},
        )
        result = client.complete("simulate", OUTPUT_SCHEMA)
        self.assertEqual(result.data, {"value": "sim", "count": 1})
        self.assertEqual(result.provenance.provider, "simulation")
        self.assertEqual(result.provenance.mode, "simulation")
        self.assertEqual(transport.requests, [])


class RetryAndResponseTest(unittest.TestCase):
    def test_429_and_5xx_retry_with_a_bound(self) -> None:
        sleeps: List[float] = []
        transport = QueueTransport(
            _http({"error": "rate"}, status=429),
            _http({"error": "down"}, status=503),
            _structured("openai", json.dumps({"value": "ok", "count": 1})),
        )
        client = LLMClient(
            LLMConfig(
                provider="openai",
                model="model",
                api_key="key",
                max_attempts=3,
                base_delay=0.25,
            ),
            transport=transport,
            sleep=sleeps.append,
        )
        result = client.complete("work", OUTPUT_SCHEMA)
        self.assertEqual(result.provenance.attempts, 3)
        self.assertEqual(sleeps, [0.25, 0.5])
        self.assertEqual(len(transport.requests), 3)

    def test_non_transient_http_failure_is_not_retried(self) -> None:
        transport = QueueTransport(_http({"error": "bad request"}, status=400))
        client = _live("openai", transport, max_attempts=3)
        with self.assertRaises(LLMProviderError) as caught:
            client.complete("work", OUTPUT_SCHEMA)
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(len(transport.requests), 1)

    def test_transport_failure_retries_but_never_switches_provider(self) -> None:
        transport = QueueTransport(
            LLMTransportError("offline"),
            _structured("openai", json.dumps({"value": "ok", "count": 1})),
        )
        client = _live("openai", transport, max_attempts=2)
        result = client.complete("work", OUTPUT_SCHEMA)
        self.assertEqual(result.provenance.provider, "openai")
        self.assertEqual(result.provenance.attempts, 2)
        self.assertEqual({request.url for request in transport.requests}, {
            "https://api.openai.com/v1/responses"
        })

    def test_duplicate_keys_and_non_finite_output_are_refused(self) -> None:
        for text in (
            '{"value":"first","value":"second","count":1}',
            '{"value":"x","count":NaN}',
        ):
            with self.subTest(text=text):
                client = _live("openai", QueueTransport(_structured("openai", text)))
                with self.assertRaises(LLMResponseError):
                    client.complete("work", OUTPUT_SCHEMA)

    def test_schema_mismatch_is_refused_even_after_provider_success(self) -> None:
        client = _live(
            "openai",
            QueueTransport(_structured("openai", '{"value":"ok","count":"one"}')),
        )
        with self.assertRaises(LLMSchemaError):
            client.complete("work", OUTPUT_SCHEMA)

    def test_provider_envelope_is_parsed_strictly_too(self) -> None:
        response = HTTPResponse(
            status=200,
            headers={},
            body=b'{"status":"completed","status":"failed"}',
        )
        client = _live("openai", QueueTransport(response))
        with self.assertRaisesRegex(LLMResponseError, "duplicate key"):
            client.complete("work", OUTPUT_SCHEMA)


class RequestSizeTest(unittest.TestCase):
    """A worker's inputs or an audit's available evidence come from the
    graph, not from a bounded intake boundary like evidence submission --
    nothing capped how large the assembled request could grow before it was
    handed to a provider. The fix refuses to send an oversized request
    rather than silently trimming it, mirroring how ``_bounded_read``
    refuses an oversized response instead of truncating it."""

    def test_an_oversized_prompt_is_refused_before_anything_is_sent(self) -> None:
        transport = QueueTransport()
        client = _live("openai", transport, max_request_bytes=1024)
        with self.assertRaisesRegex(LLMRequestTooLargeError, "1024-byte limit"):
            client.complete("x" * 4096, OUTPUT_SCHEMA)
        self.assertEqual(transport.requests, [])

    def test_a_prompt_within_budget_is_unaffected(self) -> None:
        transport = QueueTransport(_structured("openai", json.dumps({"value": "ok", "count": 1})))
        client = _live("openai", transport, max_request_bytes=1024)
        result = client.complete("work", OUTPUT_SCHEMA)
        self.assertEqual(result.data["value"], "ok")

    def _audit_context(self, output: Dict[str, Any]) -> tuple:
        runner = Runner("llm-audit-request-size")
        task = runner.task(
            "inspect",
            run=lambda _ctx: {"checked": True},
            assumes="The observed condition still holds",
        )
        assert task.assumption_id is not None
        assumption = runner.graph.assumptions[task.assumption_id]
        source = runner.graph.add_node(
            NodeType.SOURCE,
            "External observation source",
            attrs={"origin": "fixture", "retrieved_at": "fixture"},
        )
        context = AuditContext(
            task=task, output=output, assumption=assumption, graph=runner.graph
        )
        return runner, source, assumption, context

    def test_many_small_items_summing_over_budget_are_refused_not_truncated(self) -> None:
        """One side of the gap a naive post-hoc body-size check leaves open:
        available_evidence grows with everything ever submitted to the
        graph, so an audit built from a graph with enough evidence in it
        could keep growing past any provider's own request-size tolerance
        even though no single piece of it is large. Refusing it here,
        before a request is sent, is the fail-closed answer -- silently
        dropping some evidence so the request fits would change what the
        model is asked to judge without telling anyone."""
        runner, source, assumption, context = self._audit_context({"checked": True})
        for i in range(50):
            submit_evidence(
                runner.graph,
                text=f"Observation number {i}, padded: {'x' * 200}",
                source_id=source.id,
                external_id=f"llm-test-evidence-{i}",
            )
        transport = QueueTransport()
        client = _live("openai", transport, max_request_bytes=2048)
        before = (len(runner.graph.nodes), len(runner.graph.edges), assumption.status)

        with self.assertRaises(LLMRequestTooLargeError):
            propose_audit(client, context, instruction="Check the assumption.")

        self.assertEqual(transport.requests, [])
        after = (len(runner.graph.nodes), len(runner.graph.edges), assumption.status)
        self.assertEqual(before, after)

    def test_a_single_oversized_value_is_refused_without_materializing_it_first(self) -> None:
        """The other side of the gap: a worker's output is not subject to
        evidence intake's own per-string size limit, so one huge value can
        reach an audit's payload directly. A JSON encoder does not chunk a
        single large scalar -- encoding one enormous string yields it back
        as one contiguous piece -- so a check that only inspects the
        *finished* body would have to hold that whole piece in memory
        first. The bound has to reject a value this size before an encoder
        ever runs on it, not just refuse to transmit the result afterward.
        """
        _runner, _source, _assumption, context = self._audit_context({"detail": "x" * 20_000_000})
        transport = QueueTransport()
        client = _live("openai", transport, max_request_bytes=4096)

        with self.assertRaisesRegex(LLMRequestTooLargeError, "4096-byte limit"):
            propose_audit(client, context, instruction="Check the assumption.")

        self.assertEqual(transport.requests, [])

    def test_a_single_oversized_worker_input_is_refused_without_materializing_it_first(
        self,
    ) -> None:
        runner = Runner("llm-worker-request-size")
        task = runner.task("draft", run=lambda _ctx: {}, assumes="Inputs are valid")
        assert task.assumption_id is not None
        assumption = runner.graph.assumptions[task.assumption_id]
        run_context = RunContext(
            task=task,
            attempt=1,
            graph=runner.graph,
            assumption=assumption,
            inputs={"upstream": {"blob": "x" * 20_000_000}},
        )
        transport = QueueTransport()
        client = _live("openai", transport, max_request_bytes=4096)
        worker = make_worker(client, "Produce the requested object.", OUTPUT_SCHEMA)

        with self.assertRaisesRegex(LLMRequestTooLargeError, "4096-byte limit"):
            worker(run_context)

    def test_the_per_value_check_rejects_a_single_oversized_string_directly(self) -> None:
        with self.assertRaises(LLMRequestTooLargeError):
            _bounded_json_bytes({"detail": "x" * 20_000_000}, 4096)

    def test_bounded_dumps_never_hands_an_oversized_value_to_the_encoder(self) -> None:
        """Even one enormous scalar must never reach a whole-string encoder."""
        encode_string = json.encoder.encode_basestring
        with patch.object(json.encoder, "encode_basestring", wraps=encode_string) as encode:
            with self.assertRaises(LLMRequestTooLargeError):
                _bounded_json_bytes({"detail": "x" * 20_000_000}, 4096)
            self.assertTrue(encode.called)
            self.assertTrue(all(len(call.args[0]) <= 1024 for call in encode.call_args_list))

    def test_bounded_dumps_itself_refuses_many_small_values_summing_over_budget(self) -> None:
        """Isolates the running-total check from the per-value check: no
        single string here is anywhere near the limit, so only summing the
        real encoder chunks as they arrive can catch this -- and it has to
        be the serializer doing the catching, not some caller one
        layer up whose own embedding of the result happens to be one big
        string by the time anyone looks at it."""
        value = {"items": [f"item-{i}" for i in range(500)]}
        with self.assertRaises(LLMRequestTooLargeError):
            _bounded_json_bytes(value, 2048)
        # Sanity: no single item is anywhere close to the limit on its own.
        self.assertTrue(all(len(item) < 2048 for item in value["items"]))


class RequestConstructionTest(unittest.TestCase):
    def assert_bounded_refusal(self, operation):
        # Inputs already exist before tracing. The limit is 1 KiB; permit
        # generous interpreter overhead, but not a copy of the 8 MiB input.
        tracemalloc.start()
        try:
            with self.assertRaises(LLMRequestTooLargeError):
                operation()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 256 * 1024)

    def test_large_prompt_system_and_schema_are_refused_before_allocation(self):
        huge = "x" * (8 * 1024 * 1024)
        for provider in ("openai", "gemini", "anthropic", "openrouter"):
            for field in ("prompt", "system", "schema"):
                with self.subTest(provider=provider, field=field):
                    transport = QueueTransport()
                    client = _live(provider, transport, max_request_bytes=1024)
                    prompt = huge if field == "prompt" else "work"
                    system = huge if field == "system" else None
                    schema = (
                        {"type": "string", "enum": [huge]} if field == "schema" else OUTPUT_SCHEMA
                    )
                    self.assert_bounded_refusal(
                        lambda: client.complete(prompt, schema, system=system)
                    )
                    # Exercise the final serializer independently of the
                    # input preflight, so removing either bound fails a test.
                    self.assert_bounded_refusal(
                        lambda: _request_for(client.config, prompt, schema, "output", system)
                    )
                    self.assertEqual(transport.requests, [])

    def test_wire_body_exact_byte_limit_and_one_byte_over_for_every_provider(self):
        for provider in ("openai", "gemini", "anthropic", "openrouter"):
            with self.subTest(provider=provider):
                prompt = 'quotes " newline\n unicode \u00e9\U0001f600' * 8
                first = QueueTransport(_structured(provider, '{"value":"ok","count":1}'))
                _live(provider, first).complete(prompt, OUTPUT_SCHEMA)
                expected = first.requests[0].body
                transport = QueueTransport(_structured(provider, '{"value":"ok","count":1}'))
                _live(provider, transport, max_request_bytes=len(expected)).complete(
                    prompt, OUTPUT_SCHEMA
                )
                self.assertEqual(transport.requests[0].body, expected)
                refused = QueueTransport()
                with self.assertRaises(LLMRequestTooLargeError):
                    _live(provider, refused, max_request_bytes=len(expected) - 1).complete(
                        prompt, OUTPUT_SCHEMA
                    )
                self.assertEqual(refused.requests, [])

    def test_json_escaping_and_types_keep_the_existing_wire_representation(self):
        payloads = [
            {"unicode": "\u00e9\U0001f600", "escape": '\\"\n\r\t\b\f\x00'},
            {"array": [None, True, False, -100, 1.25, {}, (), []]},
            {3: "integer key", 4.5: "float key", None: "null key", False: "bool key"},
            {"chunk boundary": "x" * 1023 + '\\"\n\u00e9' * 1000},
        ]
        for payload in payloads:
            expected = json.dumps(
                payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
            self.assertEqual(_bounded_json_bytes(payload, len(expected)), expected)
            with self.assertRaises(LLMRequestTooLargeError):
                _bounded_json_bytes(payload, len(expected) - 1)

    def test_large_dictionary_keys_and_integers_do_not_allocate_unbounded_chunks(self):
        huge_key = "x" * (8 * 1024 * 1024)
        huge_integer = 1 << 1_000_000
        for value in ({huge_key: "value"}, {"number": huge_integer}):
            self.assert_bounded_refusal(lambda: _bounded_json_bytes(value, 1024))

    def test_unrepresentable_and_circular_values_fail_closed(self):
        cycle = []
        cycle.append(cycle)
        for value in (cycle, {"x": float("nan")}, {"x": object()}, {object(): 1}):
            with self.assertRaises(LLMConfigurationError):
                _bounded_json_bytes(value, 1024)

    def test_worker_input_is_bounded_before_prompt_materialization(self):
        huge = "x" * (8 * 1024 * 1024)
        transport = QueueTransport()
        client = _live("openai", transport, max_request_bytes=1024)
        worker = make_worker(client, "inspect", OUTPUT_SCHEMA)
        runner = Runner("bounded-worker")
        task = runner.task("inspect", run=lambda ctx: {}, assumes="ground")
        context = RunContext(
            task=task, attempt=1, graph=runner.graph,
            assumption=runner.graph.assumptions[task.assumption_id],
            inputs={"upstream": {"data": huge}},
        )
        self.assert_bounded_refusal(lambda: worker(context))
        self.assertEqual(transport.requests, [])
        self.assertEqual(task.attempt, 0)

    def test_audit_evidence_is_streamed_and_bounded_before_prompt_materialization(self):
        runner = Runner("bounded-audit")
        task = runner.task("inspect", run=lambda ctx: {}, assumes="ground")
        huge = "x" * (8 * 1024 * 1024)
        node = runner.graph.add_node(
            NodeType.EVIDENCE, "observation", attrs={"kind": "test", "data": huge}
        )

        class EvidenceStream(dict):
            def values(self):
                yield node
                raise AssertionError("audit enumerated evidence past an oversized record")

        runner.graph.nodes = EvidenceStream(runner.graph.nodes)
        context = AuditContext(
            task=task, output={}, graph=runner.graph,
            assumption=runner.graph.assumptions[task.assumption_id],
        )
        transport = QueueTransport()
        client = _live("openai", transport, max_request_bytes=1024)
        self.assert_bounded_refusal(
            lambda: propose_audit(client, context, instruction="inspect")
        )
        self.assertEqual(transport.requests, [])
        self.assertFalse(node.invalidated)


    def test_many_small_evidence_records_stop_at_budget_without_enumerating_all(self):
        runner = Runner("many-evidence")
        task = runner.task("inspect", run=lambda ctx: {}, assumes="ground")
        node = runner.graph.add_node(
            NodeType.EVIDENCE, "observation", attrs={"kind": "test"}
        )

        class EvidenceStream(dict):
            def values(self):
                for _ in range(100):
                    yield node
                raise AssertionError("audit enumerated every record despite exhausting budget")

        runner.graph.nodes = EvidenceStream(runner.graph.nodes)
        context = AuditContext(
            task=task, output={}, graph=runner.graph,
            assumption=runner.graph.assumptions[task.assumption_id],
        )
        transport = QueueTransport()
        client = _live("openai", transport, max_request_bytes=1024)
        self.assert_bounded_refusal(
            lambda: propose_audit(client, context, instruction="inspect")
        )
        self.assertEqual(transport.requests, [])


class AuditAuthorityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = Runner("llm-audit-boundary")
        self.task = self.runner.task(
            "inspect",
            run=lambda _ctx: {"checked": True},
            assumes="The observed condition still holds",
        )
        assert self.task.assumption_id is not None
        self.assumption = self.runner.graph.assumptions[self.task.assumption_id]
        self.source = self.runner.graph.add_node(
            NodeType.SOURCE,
            "External observation source",
            attrs={"origin": "fixture", "retrieved_at": "fixture"},
        )
        self.evidence = submit_evidence(
            self.runner.graph,
            text="The observed condition no longer holds",
            source_id=self.source.id,
            external_id="llm-test-evidence-1",
        ).node
        self.context = AuditContext(
            task=self.task,
            output={"checked": True},
            assumption=self.assumption,
            graph=self.runner.graph,
        )

    def _client(self, value: Dict[str, Any]) -> LLMClient:
        return LLMClient(
            LLMConfig.simulation(),
            simulator=lambda _prompt, _schema, _system: dict(value),
        )

    def test_llm_disproof_is_only_a_proposal_and_mutates_nothing(self) -> None:
        before = (
            len(self.runner.graph.nodes),
            len(self.runner.graph.edges),
            self.assumption.status,
            sum(node.invalidated for node in self.runner.graph.nodes.values()),
        )
        proposal = propose_audit(
            self._client(
                {
                    "status": "disproved",
                    "reason": "the evidence contradicts the ground",
                    "evidence_id": self.evidence.id,
                }
            ),
            self.context,
            instruction="Check the standing assumption against available evidence.",
        )
        after = (
            len(self.runner.graph.nodes),
            len(self.runner.graph.edges),
            self.assumption.status,
            sum(node.invalidated for node in self.runner.graph.nodes.values()),
        )
        self.assertIsInstance(proposal, AuditProposal)
        self.assertNotIsInstance(proposal, Verdict)
        self.assertEqual(proposal.evidence_id, self.evidence.id)
        self.assertEqual(before, after)
        self.assertIs(self.assumption.status, AssumptionStatus.ACTIVE)

    def test_disproof_must_name_pre_ingested_live_evidence(self) -> None:
        client = self._client(
            {
                "status": "disproved",
                "reason": "unsupported assertion",
                "evidence_id": "evidence:not-present",
            }
        )
        with self.assertRaisesRegex(LLMSchemaError, "not in the graph"):
            propose_audit(client, self.context, instruction="Check the assumption.")

    def test_non_disproof_cannot_smuggle_an_evidence_binding(self) -> None:
        client = self._client(
            {
                "status": "success",
                "reason": "still holds",
                "evidence_id": self.evidence.id,
            }
        )
        with self.assertRaisesRegex(LLMSchemaError, "hidden verdict channel"):
            propose_audit(client, self.context, instruction="Check the assumption.")

    def test_invalidated_evidence_cannot_support_a_proposal(self) -> None:
        self.evidence.invalidated = True
        client = self._client(
            {
                "status": "disproved",
                "reason": "stale evidence should not count",
                "evidence_id": self.evidence.id,
            }
        )
        with self.assertRaisesRegex(LLMSchemaError, "invalidated"):
            propose_audit(client, self.context, instruction="Check the assumption.")

    def test_audit_proposer_factory_still_returns_a_proposal_not_a_verdict(self) -> None:
        proposer = make_audit_proposer(
            self._client({"status": "success", "reason": "holds", "evidence_id": None}),
            "Check the assumption.",
        )
        proposal = proposer(self.context)
        self.assertIsInstance(proposal, AuditProposal)
        self.assertNotIsInstance(proposal, Verdict)


class WorkerTest(unittest.TestCase):
    def test_worker_carries_provider_provenance_in_durable_output(self) -> None:
        runner = Runner("llm-worker")
        task = runner.task("draft", run=lambda _ctx: {}, assumes="Inputs are valid")
        assert task.assumption_id is not None
        assumption = runner.graph.assumptions[task.assumption_id]
        context = RunContext(
            task=task,
            attempt=1,
            graph=runner.graph,
            assumption=assumption,
            inputs={"source": {"value": "input"}},
        )
        client = LLMClient(
            LLMConfig.simulation(model="sim-v1"),
            simulator=lambda _prompt, _schema, _system: {"value": "done", "count": 1},
        )
        worker = make_worker(client, "Produce the requested object.", OUTPUT_SCHEMA)
        output = worker(context)
        self.assertEqual(output["value"], "done")
        self.assertEqual(output["count"], 1)
        provenance = output["_contextmesh_llm"]
        self.assertEqual(provenance["provider"], "simulation")
        self.assertEqual(provenance["model"], "sim-v1")
        self.assertEqual(provenance["mode"], "simulation")

    def test_worker_schema_cannot_claim_the_reserved_provenance_key(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "_contextmesh_llm": {"type": "string"},
            },
            "required": ["_contextmesh_llm"],
            "additionalProperties": False,
        }
        client = LLMClient(
            LLMConfig.simulation(),
            simulator=lambda _prompt, _schema, _system: {"_contextmesh_llm": "fake"},
        )
        with self.assertRaisesRegex(LLMConfigurationError, "reserved key"):
            make_worker(client, "Do work.", schema)


class RedirectRefusalTest(unittest.TestCase):
    """PR #12: a provider request never follows a redirect.

    ``urllib`` copies request headers onto a redirected request and strips only
    the content headers, so following a 30x hands ``Authorization`` and
    ``x-api-key`` to the host the *response* names and returns 200 as though
    the exchange were ordinary. Endpoints are a fixed HTTPS allowlist, so there
    is no redirect worth following and nothing to sanitise: refuse them all and
    let the status surface as an ordinary non-2xx failure.

    These run against two real loopback servers because the behaviour under
    test belongs to urllib, not to this module's own code. A fake transport
    would assert nothing.
    """

    CREDENTIALS = {
        "Authorization": "Bearer sk-not-a-real-key",
        "x-api-key": "not-a-real-key",
    }

    def serve(self, handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_port

    def setUp(self):
        self.seen: List[Dict[str, str]] = []
        received = self.seen

        class Sink(BaseHTTPRequestHandler):
            def respond(self):
                received.append({k.lower(): v for k, v in self.headers.items()})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')

            do_GET = do_POST = respond

            def log_message(self, *args):
                pass

        self.sink_port = self.serve(Sink)

    def redirector(self, code, location):
        class Redirect(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(code)
                self.send_header("Location", location)
                self.end_headers()

            def log_message(self, *args):
                pass

        return self.serve(Redirect)

    def request_to(self, port, *, max_response_bytes=1_048_576):
        return HTTPRequest(
            method="POST",
            url=f"http://127.0.0.1:{port}/v1/responses",
            headers=dict(self.CREDENTIALS, **{"Content-Type": "application/json"}),
            body=b"{}",
            timeout=5.0,
            max_response_bytes=max_response_bytes,
        )

    def credentials_that_escaped(self):
        return [
            {k: v for k, v in headers.items() if k in ("authorization", "x-api-key")}
            for headers in self.seen
        ]

    def test_no_redirect_is_followed_and_no_credential_moves(self):
        for code in (301, 302, 303, 307, 308):
            with self.subTest(code=code):
                port = self.redirector(code, f"http://127.0.0.1:{self.sink_port}/steal")
                response = UrllibTransport()(self.request_to(port))
                self.assertEqual(response.status, code)
                self.assertEqual(self.seen, [])
                self.assertEqual(self.credentials_that_escaped(), [])

    def test_a_same_host_redirect_is_refused_too(self):
        port = self.redirector(302, "/v1/responses-moved")
        response = UrllibTransport()(self.request_to(port))
        self.assertEqual(response.status, 302)

    def test_a_refused_redirect_fails_the_request_closed(self):
        port = self.redirector(302, f"http://127.0.0.1:{self.sink_port}/steal")
        client = LLMClient(
            LLMConfig(provider="openai", model="gpt-test", api_key="sk-not-a-real-key"),
            transport=UrllibTransport(),
            sleep=lambda _seconds: None,
        )
        with self.assertRaises(LLMProviderError) as caught:
            client._send(self.request_to(port))
        self.assertEqual(caught.exception.status, 302)
        self.assertIn("failed closed", str(caught.exception))
        self.assertEqual(self.seen, [])

    def test_a_plain_response_still_arrives(self):
        """The control: refusing redirects does not refuse ordinary replies."""
        response = UrllibTransport()(self.request_to(self.sink_port))
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.body), {"ok": True})

    def oversized_server(self, *, status=200, hits=None):
        payload = b"x" * 4096

        class Oversized(BaseHTTPRequestHandler):
            def do_POST(self):
                if hits is not None:
                    hits.append(status)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        return self.serve(Oversized)

    def test_an_oversized_success_body_is_refused_without_buffering_it(self):
        port = self.oversized_server(status=200)
        request = self.request_to(port, max_response_bytes=1024)
        pattern = r"HTTP 200.*1024-byte limit"
        with self.assertRaisesRegex(LLMResponseTooLargeError, pattern) as caught:
            UrllibTransport()(request)
        self.assertEqual(caught.exception.status, 200)
        # Not a transport failure: a response was obtained, so it must not be
        # retried as if the network had dropped.
        self.assertNotIsInstance(caught.exception, LLMTransportError)

    def test_an_oversized_error_body_is_dropped_and_the_status_kept(self):
        """A huge *error* page is the same memory risk, but nothing reads an
        error body, so it is discarded after the bounded read and the status
        is what the client acts on."""
        port = self.oversized_server(status=500)
        request = self.request_to(port, max_response_bytes=1024)
        response = UrllibTransport()(request)
        self.assertEqual(response.status, 500)
        self.assertEqual(response.body, b"")

    def client(self):
        return LLMClient(
            LLMConfig(provider="openai", model="gpt-test", api_key="sk-not-a-real-key"),
            transport=UrllibTransport(),
            sleep=lambda _seconds: None,
        )

    def test_an_oversized_success_is_not_retried(self):
        hits = []
        port = self.oversized_server(status=200, hits=hits)
        with self.assertRaisesRegex(LLMResponseTooLargeError, "1024-byte limit"):
            self.client()._send(self.request_to(port, max_response_bytes=1024))
        self.assertEqual(hits, [200])

    def test_an_oversized_client_error_fails_closed_on_the_first_attempt(self):
        hits = []
        port = self.oversized_server(status=400, hits=hits)
        with self.assertRaises(LLMProviderError) as caught:
            self.client()._send(self.request_to(port, max_response_bytes=1024))
        self.assertEqual(caught.exception.status, 400)
        self.assertIn("failed closed", str(caught.exception))
        self.assertEqual(hits, [400])

    def test_an_oversized_server_error_keeps_its_retry_semantics(self):
        hits = []
        port = self.oversized_server(status=503, hits=hits)
        client = self.client()
        with self.assertRaises(LLMProviderError) as caught:
            client._send(self.request_to(port, max_response_bytes=1024))
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(len(hits), client.config.max_attempts)

    def test_a_body_at_exactly_the_limit_is_accepted(self):
        port = self.oversized_server(status=200)
        request = self.request_to(port, max_response_bytes=4096)
        response = UrllibTransport()(request)
        self.assertEqual(response.status, 200)
        self.assertEqual(len(response.body), 4096)


if __name__ == "__main__":
    unittest.main()
