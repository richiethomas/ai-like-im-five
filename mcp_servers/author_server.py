#!/usr/bin/env python3
"""
MCP Server for Claude acting as Article Author defending against reviewer concerns.
Explains writing choices, defends claims, or proposes fixes.
"""

import json
import os
import sys
import anthropic

from base_server import BaseMCPServer

class AuthorMCPServer(BaseMCPServer):
    def __init__(self):
        super().__init__("claude-sonnet-5-author")
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def scan_article(self, title: str, content: str) -> dict:
        """Prepare author context for debate (not used in author/reviewer mode)."""
        return {"context": "Author ready to defend article"}

    def respond_to_roundtable(self, title: str, content: str, dimension: str, reviewer_concerns_by_model: dict) -> dict:
        """Author responds to ALL reviewer concerns in a roundtable setting.

        reviewer_concerns_by_model: {model_name: [concerns]} where each concern is {claim, issue, suggested_fix}.
        Author sees all reviewers' feedback at once and responds collectively.
        """
        concerns_text = ""
        for model_name, concerns in reviewer_concerns_by_model.items():
            concerns_text += f"\n**{model_name}**:\n"
            for c in concerns:
                concerns_text += f"  - CLAIM: {c['claim']}\n    ISSUE: {c['issue']}\n    SUGGESTED FIX: {c.get('suggested_fix', 'N/A')}\n"

        prompt = f"""You are the author of this article and are defending it against feedback from multiple reviewers in a roundtable discussion.

Article: {title}

Content:
{content}

---

Multiple reviewers raised the following {dimension} concerns:

{concerns_text}

---

Respond to the collective feedback. For concerns you agree with, propose a concrete fix. For concerns you push back on, explain why your choice is intentional/correct. For areas of compromise, negotiate a revised phrasing.

Format your response as:

CONCERN: [brief restatement of the issue]
REVIEWER(S): [which model(s) raised this]
STANCE: [CONCEDE | DEFEND | NEGOTIATE]
RATIONALE: [1-2 sentences explaining your position]
PROPOSED_FIX: [if applicable, concrete text or code snippet]

---

Be direct. Address each unique concern once, grouping by topic if multiple reviewers raised the same issue."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-5",
                max_tokens=2500,
                messages=[{"role": "user", "content": prompt}]
            )
            return {"response": response.content[0].text}
        except Exception as e:
            return {"error": str(e), "response": ""}

    def debate(self, claim: str, my_position: str, other_positions: dict) -> dict:
        """Not used in author/reviewer mode."""
        return {"response": ""}

def main():
    server = AuthorMCPServer()
    server.run()

if __name__ == "__main__":
    main()
