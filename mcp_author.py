#!/usr/bin/env python3
"""
MCP Author Server: Claude Sonnet 5 defending article against reviewer feedback.
Standalone server with clean message dispatching.
"""

import json
import sys
import os
import anthropic

class AuthorMCPServer:
    def __init__(self):
        self.model_id = "claude-sonnet-5-author"
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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
        elif method == "respond_to_roundtable":
            return self.respond_to_roundtable(**params)
        elif method == "respond_to_rebuttals":
            return self.respond_to_rebuttals(**params)
        else:
            return {"error": f"Unknown method: {method}"}

    def scan_article(self, title: str, content: str) -> dict:
        """Scan article (not used in roundtable, kept for API consistency)."""
        return {"findings": []}

    def respond_to_roundtable(self, title: str, content: str, dimension: str, reviewer_concerns_by_model: dict) -> dict:
        """Author responds to all reviewers' concerns in roundtable.

        Args:
            title: Article title
            content: Article content
            dimension: Review dimension being debated
            reviewer_concerns_by_model: {model_name: [concerns]} where each concern is {claim, issue, suggested_fix}

        Returns:
            response: Author's response with CONCEDE/DEFEND/NEGOTIATE stances
        """
        concerns_text = ""
        for model_name, concerns in reviewer_concerns_by_model.items():
            concerns_text += f"\n**{model_name}**:\n"
            for c in concerns:
                concerns_text += f"  - CLAIM: {c.get('claim', 'N/A')}\n"
                concerns_text += f"    ISSUE: {c.get('issue', 'N/A')}\n"
                concerns_text += f"    SUGGESTED FIX: {c.get('suggested_fix', 'N/A')}\n"

        prompt = f"""You are the author of this article defending it against reviewer feedback.

Article: {title}

Content:
{content}

---

Reviewers raised these {dimension} concerns:
{concerns_text}

---

Respond to the collective feedback. For each concern:
- CONCEDE if they're right: propose concrete fix
- DEFEND if your choice is intentional: explain why
- NEGOTIATE if there's middle ground: propose revised phrasing

Format each response as:

CONCERN: [brief restatement]
REVIEWERS: [which models raised this]
STANCE: [CONCEDE | DEFEND | NEGOTIATE]
RATIONALE: [1-2 sentences]
PROPOSED_FIX: [if applicable]

Be direct and substantive. This is genuine debate, not performative agreement."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-5",
                max_tokens=2500,
                messages=[{"role": "user", "content": prompt}]
            )
            return {"response": response.content[0].text}
        except Exception as e:
            return {"error": str(e), "response": ""}

    def respond_to_rebuttals(self, title: str, content: str, dimension: str, reviewer_rebuttals: str) -> dict:
        """Author responds to reviewer rebuttals pushing back on author's previous response.

        Args:
            title: Article title
            content: Article content
            dimension: Review dimension being debated
            reviewer_rebuttals: Text of all reviewer rebuttals

        Returns:
            response: Author's response to rebuttals
        """
        prompt = f"""You are the author of this article. Reviewers are pushing back on your previous response.

Article: {title}

Content:
{content}

---

Reviewers' rebuttals:
{reviewer_rebuttals}

---

Respond to their pushback. You can:
1. Concede on specific points if they make good arguments
2. Push back further and explain why your position is correct
3. Propose a compromise

Format each response as:

REBUTTAL_TO: [brief restatement of what reviewers are pushing back on]
STANCE: [CONCEDE | DEFEND | NEGOTIATE]
RATIONALE: [1-2 sentences]
PROPOSED_FIX: [if applicable]

Be direct and substantive."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-5",
                max_tokens=2500,
                messages=[{"role": "user", "content": prompt}]
            )
            return {"response": response.content[0].text}
        except Exception as e:
            return {"error": str(e), "response": ""}

def main():
    server = AuthorMCPServer()
    server.run()

if __name__ == "__main__":
    main()
