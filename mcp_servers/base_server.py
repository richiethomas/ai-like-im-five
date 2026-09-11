#!/usr/bin/env python3
"""
Base MCP server for fact-checking.
Subclasses implement model-specific logic.
"""

import json
import sys
import os
from abc import ABC, abstractmethod
from typing import Optional

class BaseMCPServer(ABC):
    """Base class for model MCP servers."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.messages = []
        self.state = {}

    def run(self):
        """Main server loop - read requests from stdin, send responses to stdout."""
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
        """Route request to appropriate handler."""
        method = request.get("method")
        params = request.get("params", {})

        if method == "scan_article":
            return self.scan_article(**params)
        elif method == "debate":
            return self.debate(**params)
        else:
            return {"error": f"Unknown method: {method}"}

    @abstractmethod
    def scan_article(self, title: str, content: str) -> dict:
        """Scan article and return findings."""
        pass

    @abstractmethod
    def debate(self, claim: str, my_position: str, other_positions: dict) -> dict:
        """Respond to debate on a claim."""
        pass

    def _parse_findings(self, text: str) -> list:
        """Parse findings from LLM response."""
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
                        "suggested_fix": fix or None,
                        "model_id": self.model_id
                    })
            except:
                continue

        return findings
