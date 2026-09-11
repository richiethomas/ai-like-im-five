#!/usr/bin/env python3
from base_server import BaseMCPServer
import google.generativeai as genai
import os, json

class GeminiMCPServer(BaseMCPServer):
    def __init__(self):
        super().__init__("gemini-3.5-flash")
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-3.5-flash")

    def scan_article(self, title: str, content: str) -> dict:
        prompt = f"""You are a rigorous technical fact-checker reviewing a blog article about deep learning and computer vision, written for non-technical business stakeholders.

Article: {title}

Content:
{content}

MANDATORY: Check ALL six dimensions before responding. Review dimensions:
1. CORRECTNESS: Factual accuracy. Are claims true? Are definitions correct?
2. CLARITY: Is the explanation clear? Could it mislead readers? Are analogies apt?
3. COMPLETENESS: Are important caveats missing? Are edge cases overlooked?
4. CONSISTENCY: Does the article contradict itself? Are terms used consistently?
5. PEDAGOGY: Is the explanation appropriate for non-technical readers? Too dense? Too simplified?
6. CLICHÉS: Flag LLM clichés like "honest", "genuine", "load-bearing", "rides on", "shines for", "belt and suspenders", "it's a X worth Y-ing", "land" (as verb), generic praise.

Your task: Find ALL significant issues. For each dimension, identify every problem worth fixing (severity >= 3). This is a 500-1000 word article, so expect multiple issues per dimension.

For each issue found, respond with:
CLAIM: [the specific claim or phrase being criticized]
DIMENSION: [CORRECTNESS | CLARITY | COMPLETENESS | CONSISTENCY | PEDAGOGY | CLICHÉS]
ISSUE: [what's wrong with it]
SEVERITY: [1-10, where 10 is most severe]
FIX: [suggested correction]

Be comprehensive, not selective. Be direct and specific. Avoid clichés in your own response. Do not include trivial issues (severity must be >= 3). If you find no significant issues, respond with: "NO ISSUES FOUND"."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"max_output_tokens": 2000}
            )
            text = response.text
            findings = self._parse_findings(text)
            return {"findings": findings}
        except Exception as e:
            return {"error": str(e), "findings": []}

    def debate(self, claim: str, my_position: str, other_positions: dict) -> dict:
        prompt = f"""You are fact-checking this claim from a blog article: "{claim}"

Your previous position on this:
{my_position}

Other models' positions:
{json.dumps(other_positions, indent=2)}

Do you maintain your position, or do you agree/disagree with any of the other models? Be specific about which points you accept or reject and why. Focus on technical accuracy."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"max_output_tokens": 2000}
            )
            return {"response": response.text}
        except Exception as e:
            return {"error": str(e), "response": ""}

def main():
    server = GeminiMCPServer()
    server.run()

if __name__ == "__main__":
    main()
