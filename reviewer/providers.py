"""LLM providers: transport + schema-dialect translation, with cost tracking.

Every provider implements one private `_call`; the shared base class owns the
policy: record cost on every call, retry once at double max_tokens when the
response was truncated, and salvage-parse truncated JSON rather than losing
the whole response.

Providers:
  OpenAICompatibleProvider — OpenAI, DeepSeek, Together (/chat/completions).
    schema_mode "json_schema": schema enforced server-side (OpenAI, Together).
    schema_mode "json_object": JSON enforced, structure carried by the prompt
    (DeepSeek — its API 400s on json_schema).
  GeminiProvider — deprecated google-generativeai SDK (spike 2026-09-11: still
    serves gemini-3.5-flash structured output; the new google-genai SDK is not
    installed). Isolated here so migration later is a one-class change.
    Strips schema fields Gemini's validator rejects.
  AnthropicProvider — structured output via forced tool use: the model must
    call a tool whose input_schema IS the response schema, so the SDK hands
    back a parsed dict with no JSON string parsing at all.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests

from .costs import CostLedger


# ---------------------------------------------------------------------------
# Lenient JSON parsing (truncation salvage)
# ---------------------------------------------------------------------------

def decode_maybe_string(value):
    """Some models (notably Claude tool-use) JSON-string-encode nested arrays.
    Returns the decoded value, or [] if the string isn't valid JSON."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return value


def extract_list(data: dict, key: str) -> list:
    """Get data[key] as a list, tolerating the shapes Claude tool-use actually
    emits: a plain list, a JSON-string-encoded list, or a string containing
    the ENTIRE wrapper object again ({key: "{\"key\": [...]}"})."""
    value = decode_maybe_string(data.get(key, []))
    if isinstance(value, dict):
        value = decode_maybe_string(value.get(key, []))
    return value if isinstance(value, list) else []


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def parse_json_lenient(text: str):
    """json.loads with balanced-prefix repair for truncated output.

    If parsing fails, cut at the last position where a value completed
    (`}` or `]`), drop the partial trailing element, close every bracket
    still open at that point, and parse that. Loses only the element the
    truncation landed in — not the whole response.
    """
    text = strip_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    in_str = False
    esc = False
    stack: list[str] = []
    last_good: tuple[int, list[str]] | None = None
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            last_good = (i, list(stack))

    if last_good is None:
        raise json.JSONDecodeError("no complete JSON value found", text, 0)
    i, open_stack = last_good
    closers = "".join("}" if c == "{" else "]" for c in reversed(open_stack))
    return json.loads(text[: i + 1] + closers)


# ---------------------------------------------------------------------------
# Base provider
# ---------------------------------------------------------------------------

@dataclass
class RawResponse:
    text: str | None          # raw text (None when the API returned a dict directly)
    data: dict | None         # pre-parsed dict (AnthropicProvider tool input)
    input_tokens: int
    output_tokens: int
    truncated: bool


class Provider(ABC):
    """name: model id used for the cost ledger and metrics."""

    def __init__(self, name: str, ledger: CostLedger | None = None):
        self.name = name
        self.ledger = ledger

    @abstractmethod
    def _call(self, prompt: str, max_tokens: int, schema: dict | None) -> RawResponse:
        ...

    def _record(self, raw: RawResponse, label: str) -> None:
        if self.ledger is not None:
            self.ledger.record(self.name, raw.input_tokens, raw.output_tokens, label)

    def structured(self, prompt: str, schema: dict, max_tokens: int,
                   label: str = "") -> dict:
        """One structured call; on truncation retry once at 2x max_tokens,
        then salvage-parse whatever came back."""
        raw = self._call(prompt, max_tokens, schema)
        self._record(raw, label)
        if raw.truncated:
            raw = self._call(prompt, max_tokens * 2, schema)
            self._record(raw, f"{label}:retry")
        if raw.data is not None:
            return raw.data
        return parse_json_lenient(raw.text or "")

    def text(self, prompt: str, max_tokens: int, label: str = "") -> str:
        raw = self._call(prompt, max_tokens, None)
        self._record(raw, label)
        if raw.text is not None:
            return raw.text
        return json.dumps(raw.data)


# ---------------------------------------------------------------------------
# OpenAI-compatible (OpenAI, DeepSeek, Together)
# ---------------------------------------------------------------------------

class OpenAICompatibleProvider(Provider):
    def __init__(self, name: str, base_url: str, model: str, api_key_env: str,
                 schema_mode: str, ledger: CostLedger | None = None):
        super().__init__(name, ledger)
        assert schema_mode in ("json_schema", "json_object")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.schema_mode = schema_mode

    def _response_format(self, schema: dict) -> dict:
        if self.schema_mode == "json_schema":
            return {"type": "json_schema",
                    "json_schema": {"name": "response", "schema": schema}}
        return {"type": "json_object"}

    def _call(self, prompt: str, max_tokens: int, schema: dict | None) -> RawResponse:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set")
        payload: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if schema is not None:
            payload["response_format"] = self._response_format(schema)
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        body = resp.json()
        choice = body["choices"][0]
        usage = body.get("usage", {})
        return RawResponse(
            text=choice["message"]["content"],
            data=None,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            truncated=choice.get("finish_reason") == "length",
        )


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

class GeminiProvider(Provider):
    # Schema keys Gemini's validator rejects (found empirically: "Unknown field
    # for Schema: minimum").
    _UNSUPPORTED_KEYS = {"minimum", "maximum", "additionalProperties", "$schema", "strict"}

    def __init__(self, name: str, model: str, api_key_env: str,
                 ledger: CostLedger | None = None):
        super().__init__(name, ledger)
        self.model_name = model
        self.api_key_env = api_key_env
        self._model = None  # lazy: don't import/configure until first call

    def _get_model(self):
        if self._model is None:
            import google.generativeai as genai
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                raise RuntimeError(f"{self.api_key_env} is not set")
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(self.model_name)
        return self._model

    @classmethod
    def adapt_schema(cls, node):
        if isinstance(node, dict):
            return {k: cls.adapt_schema(v) for k, v in node.items()
                    if k not in cls._UNSUPPORTED_KEYS}
        if isinstance(node, list):
            return [cls.adapt_schema(v) for v in node]
        return node

    def structured(self, prompt: str, schema: dict, max_tokens: int,
                   label: str = "") -> dict:
        """Gemini's response_schema mode intermittently returns EMPTY text
        while still billing output tokens (observed live: two scan attempts,
        ~145 output tokens each, zero characters delivered). Its plain-text
        mode has never failed, and every structured prompt already spells out
        the JSON shape — so on a parse failure, fall back to text mode and
        parse leniently."""
        try:
            return super().structured(prompt, schema, max_tokens, label)
        except json.JSONDecodeError:
            raw = self._call(prompt, max_tokens, None)
            self._record(raw, f"{label}:schemaless")
            return parse_json_lenient(raw.text or "")

    def _call(self, prompt: str, max_tokens: int, schema: dict | None) -> RawResponse:
        model = self._get_model()
        config: dict = {"max_output_tokens": max_tokens}
        if schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = self.adapt_schema(schema)
        response = model.generate_content(prompt, generation_config=config)
        usage = getattr(response, "usage_metadata", None)
        finish = None
        if response.candidates:
            finish = getattr(response.candidates[0], "finish_reason", None)
        # finish_reason MAX_TOKENS is enum value 2 in this SDK; compare by name
        truncated = str(finish).endswith("MAX_TOKENS")
        return RawResponse(
            text=response.text,
            data=None,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            truncated=truncated,
        )


# ---------------------------------------------------------------------------
# Anthropic (author)
# ---------------------------------------------------------------------------

class AnthropicProvider(Provider):
    def __init__(self, name: str, model: str, api_key_env: str,
                 ledger: CostLedger | None = None):
        super().__init__(name, ledger)
        self.model = model
        self.api_key_env = api_key_env
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                raise RuntimeError(f"{self.api_key_env} is not set")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _call(self, prompt: str, max_tokens: int, schema: dict | None) -> RawResponse:
        client = self._get_client()
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if schema is not None:
            # Forced tool use: the tool's input_schema IS the response schema,
            # so the API returns a parsed dict — no JSON string parsing.
            kwargs["tools"] = [{
                "name": "emit_response",
                "description": "Emit the structured response.",
                "input_schema": schema,
            }]
            kwargs["tool_choice"] = {"type": "tool", "name": "emit_response"}
        response = client.messages.create(**kwargs)

        data = None
        text = None
        if schema is not None:
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    data = block.input
                    break
            if data is None:  # model failed to call the tool despite forcing
                text = "".join(getattr(b, "text", "") for b in response.content)
        else:
            text = "".join(getattr(b, "text", "") for b in response.content)

        return RawResponse(
            text=text,
            data=data,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            truncated=response.stop_reason == "max_tokens",
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REVIEWER_NAMES = ["gpt-4o-mini", "deepseek-chat", "gemini-3.5-flash", "llama-3.3-70b"]
AUTHOR_NAME = "claude-sonnet-5"


def build_provider(name: str, ledger: CostLedger | None = None) -> Provider:
    if name == "gpt-4o-mini":
        return OpenAICompatibleProvider(
            name, "https://api.openai.com/v1", "gpt-4o-mini",
            "OPENAI_API_KEY", "json_schema", ledger)
    if name == "deepseek-chat":
        return OpenAICompatibleProvider(
            name, "https://api.deepseek.com/v1", "deepseek-chat",
            "DEEPSEEK_API_KEY", "json_object", ledger)
    if name == "llama-3.3-70b":
        return OpenAICompatibleProvider(
            name, "https://api.together.xyz/v1",
            "meta-llama/Llama-3.3-70b-instruct-turbo",
            "TOGETHER_AI_API_KEY", "json_schema", ledger)
    if name == "gemini-3.5-flash":
        return GeminiProvider(name, "gemini-3.5-flash", "GEMINI_API_KEY", ledger)
    if name == "claude-sonnet-5":
        return AnthropicProvider(name, "claude-sonnet-5", "ANTHROPIC_API_KEY", ledger)
    raise ValueError(f"Unknown provider: {name}")
