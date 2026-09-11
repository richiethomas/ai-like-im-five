#!/usr/bin/env python3
"""
MCP Reviewer Server: one shared Reviewer over pluggable Providers.

A Provider handles two things that differ per LLM API:
  1. Transport — how the HTTP/SDK call is made.
  2. Schema dialect — how a plain JSON schema becomes that API's structured-output request.

Everything else (prompts, finding parsing, the server loop) lives once in Reviewer.

Providers:
  OpenAICompatibleProvider — OpenAI, DeepSeek, Together (all /chat/completions).
    schema_mode="json_schema": full schema enforced (OpenAI, Together).
    schema_mode="json_object": JSON output enforced, schema described in prompt (DeepSeek).
  GeminiProvider — google.generativeai, native response_schema (restricted field set).
"""

import json
import sys
import os
import re
from abc import ABC, abstractmethod

import google.generativeai as genai
import requests


# ---------------------------------------------------------------------------
# Canonical schema + prompts (defined once, shared across every model)
# ---------------------------------------------------------------------------

DIMENSIONS = ["CORRECTNESS", "CLARITY", "COMPLETENESS", "CONSISTENCY", "PEDAGOGY", "CLICHÉS"]

FINDINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "dimension": {"type": "string", "enum": DIMENSIONS},
                    "issue": {"type": "string"},
                    "severity": {"type": "integer", "minimum": 1, "maximum": 10},
                    "fix": {"type": "string"},
                },
                "required": ["claim", "dimension", "issue", "severity"],
            },
        }
    },
    "required": ["findings"],
}


def scan_prompt(title: str, content: str) -> str:
    return f"""You are a rigorous technical fact-checker reviewing a blog article about deep learning and computer vision, written for non-technical business stakeholders.

Article: {title}

Content:
{content}

MANDATORY: Check ALL six dimensions. Review dimensions:
1. CORRECTNESS: Factual accuracy. Are claims true? Definitions correct?
2. CLARITY: Clear explanation? Could it mislead? Analogies apt?
3. COMPLETENESS: Missing caveats? Overlooked edge cases?
4. CONSISTENCY: Self-contradiction? Terms used consistently?
5. PEDAGOGY: Appropriate for non-technical readers? Too dense? Too simplified?
6. CLICHÉS: LLM clichés like "honest", "genuine", "load-bearing", "rides on", "shines for", "belt and suspenders", "it's a X worth Y-ing", "land" (as verb)?

Find ALL significant issues (severity >= 3). This is a 500-1000 word article, so expect multiple issues per dimension.

Return findings as JSON with this structure:
{{
  "findings": [
    {{"claim": "...", "dimension": "CORRECTNESS", "issue": "...", "severity": 5, "fix": "..."}},
    ...
  ]
}}

Be comprehensive and specific. If no issues found, return {{"findings": []}}."""


def debate_stance_prompt(dimension: str, your_concerns: list, author_response: str) -> str:
    concerns_text = "\n".join([f"- {c.get('issue', 'N/A')}" for c in your_concerns])
    return f"""You reviewed an article and raised {dimension} concerns:

{concerns_text}

The author has now responded:

{author_response}

---

Do you accept the author's response, or push back further? Consider:
- Did they address your core concern?
- Is their rationale convincing?
- Should you concede, rebut, or negotiate?

Respond with:
ACCEPT | [brief explanation]
REBUT | [your counterargument]
NEGOTIATE | [proposed middle ground]

Be concise and substantive."""


def debate_prompt(claim: str, my_position: str, other_positions: dict) -> str:
    return f"""Debate claim: "{claim}"

Your position: {my_position}

Others: {json.dumps(other_positions, indent=2)}

Maintain or change your position? Be specific about accepting/rejecting points. Focus on accuracy."""


def _extract_json(text: str) -> str:
    """Strip markdown code fences some models wrap JSON in."""
    text = text.strip()
    if text.startswith("```"):
        # remove opening fence (```json or ```) and closing fence
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def _loads_findings(text: str) -> dict:
    """Parse a findings payload, salvaging complete objects from truncated JSON.

    Some models (DeepSeek in json_object mode) emit very long finding lists and
    get cut off at the token limit, leaving the outer array unclosed. Rather than
    lose the whole response, recover every complete finding object that did land.
    """
    text = _extract_json(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        findings = []
        stack = []
        for i, ch in enumerate(text):
            if ch == "{":
                stack.append(i)
            elif ch == "}" and stack:
                frag = text[stack.pop():i + 1]
                try:
                    obj = json.loads(frag)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and ("claim" in obj or "issue" in obj):
                    findings.append(obj)
        return {"findings": findings}


# ---------------------------------------------------------------------------
# Providers: transport + schema-dialect translation
# ---------------------------------------------------------------------------

class Provider(ABC):
    """Turns 'give me structured output matching this schema' into one API's request."""

    @abstractmethod
    def structured(self, prompt: str, schema: dict, max_tokens: int) -> dict:
        """Return a parsed JSON dict conforming (best-effort) to schema."""

    @abstractmethod
    def text(self, prompt: str, max_tokens: int) -> str:
        """Return a plain-text completion."""


class OpenAICompatibleProvider(Provider):
    """Any provider exposing the OpenAI /chat/completions contract."""

    # Keys the OpenAI json_schema validator rejects when strict; harmless to keep,
    # but json_object mode ignores the schema entirely.
    def __init__(self, base_url: str, model: str, api_key_env: str, schema_mode: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = os.getenv(api_key_env)
        self.schema_mode = schema_mode  # "json_schema" | "json_object"

    def _post(self, messages: list, max_tokens: int, response_format=None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
        if response_format:
            data["response_format"] = response_format
        resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _response_format(self, schema: dict):
        if self.schema_mode == "json_schema":
            return {"type": "json_schema",
                    "json_schema": {"name": "findings", "schema": schema}}
        # json_object: JSON output enforced, structure carried by the prompt only
        return {"type": "json_object"}

    def structured(self, prompt: str, schema: dict, max_tokens: int) -> dict:
        text = self._post(
            [{"role": "user", "content": prompt}],
            max_tokens,
            response_format=self._response_format(schema),
        )
        return _loads_findings(text)

    def text(self, prompt: str, max_tokens: int) -> str:
        return self._post([{"role": "user", "content": prompt}], max_tokens)


class GeminiProvider(Provider):
    """google.generativeai native structured output via response_schema."""

    # Fields Gemini's Schema type does not accept — stripped before sending.
    _UNSUPPORTED = {"minimum", "maximum", "additionalProperties", "$schema", "strict"}

    def __init__(self, model: str, api_key_env: str):
        genai.configure(api_key=os.getenv(api_key_env))
        self.model = genai.GenerativeModel(model)

    def _adapt_schema(self, node):
        """Recursively drop keys Gemini's schema validator rejects."""
        if isinstance(node, dict):
            return {k: self._adapt_schema(v) for k, v in node.items()
                    if k not in self._UNSUPPORTED}
        if isinstance(node, list):
            return [self._adapt_schema(v) for v in node]
        return node

    def structured(self, prompt: str, schema: dict, max_tokens: int) -> dict:
        response = self.model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": self._adapt_schema(schema),
            },
        )
        return _loads_findings(response.text)

    def text(self, prompt: str, max_tokens: int) -> str:
        response = self.model.generate_content(prompt, stream=False)
        return response.text


# ---------------------------------------------------------------------------
# Reviewer: one implementation, parameterized by a Provider
# ---------------------------------------------------------------------------

class Reviewer:
    def __init__(self, model_id: str, provider: Provider):
        self.model_id = model_id
        self.provider = provider

    # --- MCP server loop -----------------------------------------------------

    def run(self):
        for line in sys.stdin:
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                print(json.dumps(response))
                sys.stdout.flush()
            except json.JSONDecodeError:
                print(json.dumps({"error": "Invalid JSON"}))
                sys.stdout.flush()
            except Exception as e:
                print(json.dumps({"error": str(e)}))
                sys.stdout.flush()

    def handle_request(self, request: dict) -> dict:
        method = request.get("method")
        params = request.get("params", {})
        if method == "scan_article":
            return self.scan_article(**params)
        elif method == "debate_stance":
            return self.debate_stance(**params)
        elif method == "debate":
            return self.debate(**params)
        return {"error": f"Unknown method: {method}"}

    # --- methods -------------------------------------------------------------

    def scan_article(self, title: str, content: str) -> dict:
        try:
            data = self.provider.structured(scan_prompt(title, content), FINDINGS_SCHEMA, 4000)
            return {"findings": self._parse_findings(data)}
        except Exception as e:
            return {"error": str(e), "findings": []}

    def debate_stance(self, title: str, content: str, dimension: str,
                      your_concerns: list, author_response: str) -> dict:
        try:
            prompt = debate_stance_prompt(dimension, your_concerns, author_response)
            return {"response": self.provider.text(prompt, 500)}
        except Exception as e:
            return {"error": str(e), "response": ""}

    def debate(self, claim: str, my_position: str, other_positions: dict) -> dict:
        try:
            prompt = debate_prompt(claim, my_position, other_positions)
            return {"response": self.provider.text(prompt, 500)}
        except Exception as e:
            return {"error": str(e), "response": ""}

    # --- parsing -------------------------------------------------------------

    def _parse_findings(self, response_data) -> list:
        """Normalize findings from a structured response (dict) or fall back to text."""
        if isinstance(response_data, dict):
            findings = response_data.get("findings", [])
            for f in findings:
                if "suggested_fix" not in f and "fix" in f:
                    f["suggested_fix"] = f.pop("fix")
                if "severity" in f and isinstance(f["severity"], str):
                    try:
                        f["severity"] = int(re.search(r"\d+", f["severity"]).group())
                    except Exception:
                        f["severity"] = 5
            return findings

        # Fallback: parse CLAIM/DIMENSION/ISSUE/SEVERITY/FIX text blocks
        text = response_data if isinstance(response_data, str) else str(response_data)
        findings = []
        for part in text.split("CLAIM:")[1:]:
            try:
                lines = part.strip().split("\n")
                claim = lines[0].strip() if lines else ""
                issue, dimension, severity, fix = "", "CORRECTNESS", 5, ""
                for line in lines[1:]:
                    if line.startswith("DIMENSION:"):
                        dimension = line.replace("DIMENSION:", "").strip()
                    elif line.startswith("ISSUE:"):
                        issue = line.replace("ISSUE:", "").strip()
                    elif line.startswith("SEVERITY:"):
                        try:
                            severity = max(1, min(10, int(re.search(r"\d+", line).group())))
                        except Exception:
                            severity = 5
                    elif line.startswith("FIX:"):
                        fix = line.replace("FIX:", "").strip()
                if claim and issue:
                    findings.append({
                        "claim": claim, "issue": issue, "severity": severity,
                        "dimension": dimension, "suggested_fix": fix or None,
                    })
            except Exception:
                continue
        return findings


# ---------------------------------------------------------------------------
# Registry: model name -> (model_id, provider). This is the only place that
# knows which API each reviewer speaks.
# ---------------------------------------------------------------------------

def build_reviewer(name: str) -> Reviewer:
    if name == "openai":
        return Reviewer("gpt-4o-mini", OpenAICompatibleProvider(
            "https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY", "json_schema"))
    if name == "deepseek":
        return Reviewer("deepseek-chat", OpenAICompatibleProvider(
            "https://api.deepseek.com/v1", "deepseek-chat", "DEEPSEEK_API_KEY", "json_object"))
    if name == "together":
        return Reviewer("llama-3-70b", OpenAICompatibleProvider(
            "https://api.together.xyz/v1", "meta-llama/Llama-3.3-70b-instruct-turbo",
            "TOGETHER_AI_API_KEY", "json_schema"))
    if name == "gemini":
        return Reviewer("gemini-3.5-flash", GeminiProvider(
            "gemini-3.5-flash", "GEMINI_API_KEY"))
    raise ValueError(f"Unknown model: {name}")


def main():
    server = build_reviewer(os.getenv("REVIEWER_MODEL", "openai"))
    server.run()


if __name__ == "__main__":
    main()
