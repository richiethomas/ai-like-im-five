#!/usr/bin/env python3
from base_server import BaseMCPServer
import requests, os, json

class TogetherMCPServer(BaseMCPServer):
    def __init__(self):
        super().__init__("llama-3-70b")
        self.api_key = os.getenv("TOGETHER_AI_API_KEY")
        self.base_url = "https://api.together.xyz/v1"

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

Your task: Identify 3-5 most important issues across ALL dimensions (not just one). Prioritize by severity.

For each issue found, respond with:
CLAIM: [the specific claim or phrase being criticized]
DIMENSION: [CORRECTNESS | CLARITY | COMPLETENESS | CONSISTENCY | PEDAGOGY | CLICHÉS]
ISSUE: [what's wrong with it]
SEVERITY: [1-10, where 10 is most severe]
FIX: [suggested correction]

Be direct and specific. Avoid clichés in your own response. Do not include minor issues (severity must be >= 3). If you find no significant issues, respond with: "NO ISSUES FOUND"."""

        try:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            data = {"model": "meta-llama/Llama-3.3-70b-instruct-turbo", "max_tokens": 1000, "messages": [{"role": "user", "content": prompt}]}
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
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
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            data = {"model": "meta-llama/Llama-3.3-70b-instruct-turbo", "max_tokens": 500, "messages": [{"role": "user", "content": prompt}]}
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            return {"response": response.json()["choices"][0]["message"]["content"]}
        except Exception as e:
            return {"error": str(e), "response": ""}

def main():
    server = TogetherMCPServer()
    server.run()

if __name__ == "__main__":
    main()
