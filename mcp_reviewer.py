#!/usr/bin/env python3
"""
MCP Reviewer Server: Generic reviewer implementation for any LLM model.
Used for OpenAI, DeepSeek, Gemini, Together AI Llama.
"""

import json
import sys
import os
import re
from abc import ABC, abstractmethod
import openai
import google.generativeai as genai
import requests

class ReviewerMCPServer(ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id

    def run(self):
        """Main server loop: read JSON requests, dispatch, return responses."""
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
        """Route request to handler based on method."""
        method = request.get("method")
        params = request.get("params", {})

        if method == "scan_article":
            return self.scan_article(**params)
        elif method == "debate_stance":
            return self.debate_stance(**params)
        elif method == "debate":
            return self.debate(**params)
        else:
            return {"error": f"Unknown method: {method}"}

    @abstractmethod
    def scan_article(self, title: str, content: str) -> dict:
        """Find all significant issues across all 6 dimensions."""
        pass

    @abstractmethod
    def debate_stance(self, title: str, content: str, dimension: str, your_concerns: list, author_response: str) -> dict:
        """Respond to author's defense of your concerns."""
        pass

    @abstractmethod
    def debate(self, claim: str, my_position: str, other_positions: dict) -> dict:
        """Debate a claim (kept for API consistency)."""
        pass

    def _parse_findings(self, response_data) -> list:
        """Parse findings from LLM response. Handles both JSON (preferred) and text formats."""
        # If response is already a dict (from JSON mode), extract findings
        if isinstance(response_data, dict):
            findings = response_data.get("findings", [])
            # Normalize fields
            for f in findings:
                if "suggested_fix" not in f and "fix" in f:
                    f["suggested_fix"] = f.pop("fix")
                if "severity" in f and isinstance(f["severity"], str):
                    try:
                        f["severity"] = int(re.search(r'\d+', f["severity"]).group())
                    except:
                        f["severity"] = 5
            return findings

        # Fallback: parse text format (for backwards compatibility)
        if isinstance(response_data, str):
            text = response_data
        else:
            text = str(response_data)

        findings = []
        parts = text.split("CLAIM:")

        for part in parts[1:]:
            try:
                lines = part.strip().split("\n")
                claim = lines[0].strip() if lines else ""

                issue = ""
                dimension = "CORRECTNESS"
                severity = 5
                fix = ""

                for line in lines[1:]:
                    if line.startswith("DIMENSION:"):
                        dimension = line.replace("DIMENSION:", "").strip()
                    elif line.startswith("ISSUE:"):
                        issue = line.replace("ISSUE:", "").strip()
                    elif line.startswith("SEVERITY:"):
                        try:
                            severity = int(re.search(r'\d+', line).group())
                            severity = max(1, min(10, severity))
                        except:
                            severity = 5
                    elif line.startswith("FIX:"):
                        fix = line.replace("FIX:", "").strip()

                if claim and issue:
                    findings.append({
                        "claim": claim,
                        "issue": issue,
                        "severity": severity,
                        "dimension": dimension,
                        "suggested_fix": fix or None
                    })
            except:
                continue

        return findings


class OpenAIReviewer(ReviewerMCPServer):
    def __init__(self):
        super().__init__("gpt-4o-mini")
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    FINDINGS_SCHEMA = {
        "type": "json_schema",
        "json_schema": {
            "name": "findings",
            "schema": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim": {"type": "string"},
                                "dimension": {"type": "string", "enum": ["CORRECTNESS", "CLARITY", "COMPLETENESS", "CONSISTENCY", "PEDAGOGY", "CLICHÉS"]},
                                "issue": {"type": "string"},
                                "severity": {"type": "integer", "minimum": 1, "maximum": 10},
                                "fix": {"type": "string"}
                            },
                            "required": ["claim", "dimension", "issue", "severity"]
                        }
                    }
                },
                "required": ["findings"]
            }
        }
    }

    def scan_article(self, title: str, content: str) -> dict:
        prompt = f"""You are a rigorous technical fact-checker reviewing a blog article about deep learning and computer vision, written for non-technical business stakeholders.

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

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=2000,
                response_format=self.FINDINGS_SCHEMA,
                messages=[{"role": "user", "content": prompt}]
            )
            data = json.loads(response.choices[0].message.content)
            findings = self._parse_findings(data)
            return {"findings": findings}
        except Exception as e:
            return {"error": str(e), "findings": []}

    def debate_stance(self, title: str, content: str, dimension: str, your_concerns: list, author_response: str) -> dict:
        concerns_text = "\n".join([f"- {c.get('issue', 'N/A')}" for c in your_concerns])

        prompt = f"""You reviewed an article and raised {dimension} concerns:

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

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            return {"response": response.choices[0].message.content}
        except Exception as e:
            return {"error": str(e), "response": ""}

    def debate(self, claim: str, my_position: str, other_positions: dict) -> dict:
        prompt = f"""Debate claim: "{claim}"

Your position: {my_position}

Others: {json.dumps(other_positions, indent=2)}

Maintain or change your position? Be specific about accepting/rejecting points. Focus on accuracy."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            return {"response": response.choices[0].message.content}
        except Exception as e:
            return {"error": str(e), "response": ""}


class DeepSeekReviewer(ReviewerMCPServer):
    def __init__(self):
        super().__init__("deepseek-chat")
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/v1"
        self.FINDINGS_SCHEMA = {
            "type": "json_schema",
            "json_schema": {
                "name": "findings",
                "schema": {
                    "type": "object",
                    "properties": {
                        "findings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "claim": {"type": "string"},
                                    "dimension": {"type": "string", "enum": ["CORRECTNESS", "CLARITY", "COMPLETENESS", "CONSISTENCY", "PEDAGOGY", "CLICHÉS"]},
                                    "issue": {"type": "string"},
                                    "severity": {"type": "integer", "minimum": 1, "maximum": 10},
                                    "fix": {"type": "string"}
                                },
                                "required": ["claim", "dimension", "issue", "severity"]
                            }
                        }
                    },
                    "required": ["findings"]
                }
            }
        }

    def _call_api(self, prompt: str, max_tokens: int = 2000, response_format=None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "model": "deepseek-chat",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        if response_format:
            data["response_format"] = response_format
        response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def scan_article(self, title: str, content: str) -> dict:
        prompt = f"""You are a rigorous technical fact-checker reviewing a blog article about deep learning and computer vision, written for non-technical business stakeholders.

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

Return findings as JSON:
{{
  "findings": [
    {{"claim": "...", "dimension": "CORRECTNESS", "issue": "...", "severity": 5, "fix": "..."}},
    ...
  ]
}}

Be comprehensive and specific. If no issues found, return {{"findings": []}}."""

        try:
            text = self._call_api(prompt, 2000, self.FINDINGS_SCHEMA)
            data = json.loads(text)
            findings = self._parse_findings(data)
            return {"findings": findings}
        except Exception as e:
            return {"error": str(e), "findings": []}

    def debate_stance(self, title: str, content: str, dimension: str, your_concerns: list, author_response: str) -> dict:
        concerns_text = "\n".join([f"- {c.get('issue', 'N/A')}" for c in your_concerns])

        prompt = f"""You reviewed an article and raised {dimension} concerns:

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

        try:
            return {"response": self._call_api(prompt, 500)}
        except Exception as e:
            return {"error": str(e), "response": ""}

    def debate(self, claim: str, my_position: str, other_positions: dict) -> dict:
        prompt = f"""Debate claim: "{claim}"

Your position: {my_position}

Others: {json.dumps(other_positions, indent=2)}

Maintain or change your position? Be specific about accepting/rejecting points. Focus on accuracy."""

        try:
            return {"response": self._call_api(prompt, 500)}
        except Exception as e:
            return {"error": str(e), "response": ""}


class GeminiReviewer(ReviewerMCPServer):
    def __init__(self):
        super().__init__("gemini-3.5-flash")
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-3.5-flash")
        self.schema = {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "dimension": {"type": "string", "enum": ["CORRECTNESS", "CLARITY", "COMPLETENESS", "CONSISTENCY", "PEDAGOGY", "CLICHÉS"]},
                            "issue": {"type": "string"},
                            "severity": {"type": "integer", "minimum": 1, "maximum": 10},
                            "fix": {"type": "string"}
                        },
                        "required": ["claim", "dimension", "issue", "severity"]
                    }
                }
            },
            "required": ["findings"]
        }

    def scan_article(self, title: str, content: str) -> dict:
        prompt = f"""You are a rigorous technical fact-checker reviewing a blog article about deep learning and computer vision, written for non-technical business stakeholders.

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

Return findings as JSON:
{{
  "findings": [
    {{"claim": "...", "dimension": "CORRECTNESS", "issue": "...", "severity": 5, "fix": "..."}},
    ...
  ]
}}

Be comprehensive and specific. If no issues found, return {{"findings": []}}."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": self.schema
                }
            )
            data = json.loads(response.text)
            findings = self._parse_findings(data)
            return {"findings": findings}
        except Exception as e:
            return {"error": str(e), "findings": []}

    def debate_stance(self, title: str, content: str, dimension: str, your_concerns: list, author_response: str) -> dict:
        concerns_text = "\n".join([f"- {c.get('issue', 'N/A')}" for c in your_concerns])

        prompt = f"""You reviewed an article and raised {dimension} concerns:

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

        try:
            response = self.model.generate_content(prompt, stream=False)
            return {"response": response.text}
        except Exception as e:
            return {"error": str(e), "response": ""}

    def debate(self, claim: str, my_position: str, other_positions: dict) -> dict:
        prompt = f"""Debate claim: "{claim}"

Your position: {my_position}

Others: {json.dumps(other_positions, indent=2)}

Maintain or change your position? Be specific about accepting/rejecting points. Focus on accuracy."""

        try:
            response = self.model.generate_content(prompt, stream=False)
            return {"response": response.text}
        except Exception as e:
            return {"error": str(e), "response": ""}


class TogetherReviewer(ReviewerMCPServer):
    def __init__(self):
        super().__init__("llama-3-70b")
        self.api_key = os.getenv("TOGETHER_AI_API_KEY")
        self.base_url = "https://api.together.xyz/v1"
        self.FINDINGS_SCHEMA = {
            "type": "json_schema",
            "json_schema": {
                "name": "findings",
                "schema": {
                    "type": "object",
                    "properties": {
                        "findings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "claim": {"type": "string"},
                                    "dimension": {"type": "string", "enum": ["CORRECTNESS", "CLARITY", "COMPLETENESS", "CONSISTENCY", "PEDAGOGY", "CLICHÉS"]},
                                    "issue": {"type": "string"},
                                    "severity": {"type": "integer", "minimum": 1, "maximum": 10},
                                    "fix": {"type": "string"}
                                },
                                "required": ["claim", "dimension", "issue", "severity"]
                            }
                        }
                    },
                    "required": ["findings"]
                }
            }
        }

    def _call_api(self, prompt: str, max_tokens: int = 2000, response_format=None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "model": "meta-llama/Llama-3.3-70b-instruct-turbo",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        if response_format:
            data["response_format"] = response_format
        response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def scan_article(self, title: str, content: str) -> dict:
        prompt = f"""You are a rigorous technical fact-checker reviewing a blog article about deep learning and computer vision, written for non-technical business stakeholders.

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

Return findings as JSON:
{{
  "findings": [
    {{"claim": "...", "dimension": "CORRECTNESS", "issue": "...", "severity": 5, "fix": "..."}},
    ...
  ]
}}

Be comprehensive and specific. If no issues found, return {{"findings": []}}."""

        try:
            text = self._call_api(prompt, 2000, self.FINDINGS_SCHEMA)
            data = json.loads(text)
            findings = self._parse_findings(data)
            return {"findings": findings}
        except Exception as e:
            return {"error": str(e), "findings": []}

    def debate_stance(self, title: str, content: str, dimension: str, your_concerns: list, author_response: str) -> dict:
        concerns_text = "\n".join([f"- {c.get('issue', 'N/A')}" for c in your_concerns])

        prompt = f"""You reviewed an article and raised {dimension} concerns:

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

        try:
            return {"response": self._call_api(prompt, 500)}
        except Exception as e:
            return {"error": str(e), "response": ""}

    def debate(self, claim: str, my_position: str, other_positions: dict) -> dict:
        prompt = f"""Debate claim: "{claim}"

Your position: {my_position}

Others: {json.dumps(other_positions, indent=2)}

Maintain or change your position? Be specific about accepting/rejecting points. Focus on accuracy."""

        try:
            return {"response": self._call_api(prompt, 500)}
        except Exception as e:
            return {"error": str(e), "response": ""}


def main():
    """Determine which reviewer to instantiate based on environment or argument."""
    model = os.getenv("REVIEWER_MODEL", "openai")

    if model == "openai":
        server = OpenAIReviewer()
    elif model == "deepseek":
        server = DeepSeekReviewer()
    elif model == "gemini":
        server = GeminiReviewer()
    elif model == "together":
        server = TogetherReviewer()
    else:
        raise ValueError(f"Unknown model: {model}")

    server.run()


if __name__ == "__main__":
    main()
