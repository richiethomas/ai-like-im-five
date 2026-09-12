"""M2 unit tests: lenient JSON parsing, schema adaptation, retry policy. Offline."""

import json

import pytest

from reviewer.costs import CostLedger
from reviewer.providers import (
    GeminiProvider,
    Provider,
    RawResponse,
    parse_json_lenient,
    strip_fences,
)


# --- parse_json_lenient ------------------------------------------------------

def test_valid_json_passthrough():
    assert parse_json_lenient('{"findings": [{"a": 1}]}') == {"findings": [{"a": 1}]}


def test_fenced_json():
    assert parse_json_lenient('```json\n{"a": 1}\n```') == {"a": 1}


def test_truncated_mid_object_drops_partial_element():
    text = '{"findings": [{"issue": "one", "severity": 3}, {"issue": "two", "sev'
    assert parse_json_lenient(text) == {"findings": [{"issue": "one", "severity": 3}]}


def test_truncated_mid_string():
    text = '{"findings": [{"issue": "complete"}, {"issue": "cut off in the middl'
    assert parse_json_lenient(text) == {"findings": [{"issue": "complete"}]}


def test_truncated_with_nested_structures():
    text = '{"findings": [{"issue": "a", "tags": ["x", "y"]}, {"issue": "b", "tags": ["p"'
    assert parse_json_lenient(text) == {"findings": [{"issue": "a", "tags": ["x", "y"]}]}


def test_escaped_quotes_inside_strings():
    text = '{"findings": [{"issue": "he said \\"hi\\" loudly"}, {"issue": "trunc'
    assert parse_json_lenient(text) == {"findings": [{"issue": 'he said "hi" loudly'}]}


def test_hopeless_input_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_json_lenient("not json at all")


def test_strip_fences_plain_text_untouched():
    assert strip_fences('{"a": 1}') == '{"a": 1}'


# --- Gemini schema adaptation ------------------------------------------------

def test_gemini_strips_unsupported_keys_recursively():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                },
            }
        },
    }
    adapted = GeminiProvider.adapt_schema(schema)
    assert "additionalProperties" not in adapted
    sev = adapted["properties"]["items"]["items"]["properties"]["severity"]
    assert sev == {"type": "integer"}
    # Original untouched
    assert "minimum" in schema["properties"]["items"]["items"]["properties"]["severity"]


# --- base Provider policy: cost recording + truncation retry -----------------

class FakeProvider(Provider):
    """Scripted responses to test the base-class policy."""

    def __init__(self, responses, ledger=None):
        super().__init__("gpt-4o-mini", ledger)  # priced model name
        self.responses = list(responses)
        self.calls = []  # (max_tokens, schema is not None)

    def _call(self, prompt, max_tokens, schema):
        self.calls.append((max_tokens, schema is not None))
        return self.responses.pop(0)


def ok(data_text, tokens=(100, 50), truncated=False):
    return RawResponse(text=data_text, data=None,
                       input_tokens=tokens[0], output_tokens=tokens[1],
                       truncated=truncated)


def test_structured_records_cost():
    ledger = CostLedger()
    p = FakeProvider([ok('{"x": 1}')], ledger)
    assert p.structured("prompt", {"type": "object"}, 1000) == {"x": 1}
    assert ledger.total_usd > 0
    assert ledger.entries[0]["model"] == "gpt-4o-mini"


def test_truncation_retries_once_at_double_tokens():
    ledger = CostLedger()
    p = FakeProvider([
        ok('{"x": 1', truncated=True),   # first call truncated
        ok('{"x": 2}'),                  # retry completes
    ], ledger)
    assert p.structured("prompt", {"type": "object"}, 1000) == {"x": 2}
    assert p.calls == [(1000, True), (2000, True)]
    assert len(ledger.entries) == 2  # both calls cost money


def test_truncated_retry_still_truncated_salvages():
    p = FakeProvider([
        ok('{"findings": [{"a": 1}', truncated=True),
        ok('{"findings": [{"a": 1}, {"b": 2', truncated=True),
    ])
    assert p.structured("prompt", {"type": "object"}, 500) == {"findings": [{"a": 1}]}


def test_pre_parsed_data_skips_json_parsing():
    p = FakeProvider([RawResponse(text=None, data={"direct": True},
                                  input_tokens=10, output_tokens=5, truncated=False)])
    assert p.structured("prompt", {"type": "object"}, 100) == {"direct": True}


# --- Gemini schema-less fallback --------------------------------------------

class ScriptedGemini(GeminiProvider):
    """GeminiProvider with _call scripted instead of hitting the network."""

    def __init__(self, responses, ledger=None):
        super().__init__("gemini-3.5-flash", "gemini-3.5-flash",
                         "GEMINI_API_KEY", ledger)
        self.responses = list(responses)
        self.calls = []  # (max_tokens, schema is not None)

    def _call(self, prompt, max_tokens, schema):
        self.calls.append((max_tokens, schema is not None))
        return self.responses.pop(0)


def test_gemini_empty_structured_response_falls_back_to_text_mode():
    """Live failure mode: response_schema calls return EMPTY text (tokens
    billed, nothing delivered) while text mode works. The provider must
    retry schema-less and parse leniently."""
    from reviewer.costs import CostLedger
    ledger = CostLedger()
    p = ScriptedGemini([
        ok("", tokens=(1960, 146)),                      # structured: empty
        ok('```json\n{"findings": [{"issue": "x"}]}\n```',
           tokens=(1960, 200)),                          # text-mode fallback
    ], ledger)
    assert p.structured("prompt with JSON shape", {"type": "object"}, 4000,
                        label="scan") == {"findings": [{"issue": "x"}]}
    assert p.calls == [(4000, True), (4000, False)]      # second call schema-less
    labels = [e["label"] for e in ledger.entries]
    assert labels == ["scan", "scan:schemaless"]         # both calls billed


def test_gemini_fallback_still_raises_when_text_mode_also_garbage():
    import json as _json
    import pytest as _pytest
    p = ScriptedGemini([ok(""), ok("still not json")])
    with _pytest.raises(_json.JSONDecodeError):
        p.structured("prompt", {"type": "object"}, 4000)


def test_gemini_healthy_structured_response_skips_fallback():
    p = ScriptedGemini([ok('{"findings": []}')])
    assert p.structured("prompt", {"type": "object"}, 4000) == {"findings": []}
    assert p.calls == [(4000, True)]                     # no second call


# --- Gemini truncation detection (the int-2 finish_reason bug) ---------------

def test_gemini_is_truncated_handles_int_and_name():
    # SDK returns MAX_TOKENS as the bare int 2 in some builds, an enum repr in
    # others. Both must read as truncated; STOP (1) and None must not.
    assert GeminiProvider._is_truncated(2) is True
    assert GeminiProvider._is_truncated("FinishReason.MAX_TOKENS") is True
    assert GeminiProvider._is_truncated(1) is False
    assert GeminiProvider._is_truncated("FinishReason.STOP") is False
    assert GeminiProvider._is_truncated(None) is False
