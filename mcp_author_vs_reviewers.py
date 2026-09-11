#!/usr/bin/env python3
"""
Author vs. Reviewers roundtable debate using MCP.
Claude defends article against feedback from 4 reviewer models in priority-ordered dimensions.
"""

import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import re

@dataclass
class ReviewIssue:
    claim: str
    issue: str
    severity: int
    dimension: str
    suggested_fix: Optional[str] = None
    reviewer: Optional[str] = None

@dataclass
class AuthorResponse:
    concern: str
    stance: str  # CONCEDE | DEFEND | NEGOTIATE
    rationale: str
    proposed_fix: Optional[str] = None

DIMENSION_PRIORITY = [
    "CORRECTNESS",
    "CLARITY",
    "COMPLETENESS",
    "CONSISTENCY",
    "PEDAGOGY",
    "CLICHÉS"
]

class AuthorVsReviewersOrchestrator:
    def __init__(self, article_path: str, max_rounds_per_dimension: int = 3):
        self.article_path = article_path
        self.max_rounds_per_dimension = max_rounds_per_dimension

        # Load article
        self.title, self.excerpt, self.content = self._extract_article()

        # MCP server processes
        self.servers = {}
        self.server_processes = {}

        # Roundtable state per dimension
        self.dimension_debates = {}  # dimension -> {reviewers: {issues}, author_response, round_count}
        self.transcript = []
        self.agreed_changes = []  # Issues author conceded
        self.open_disagreements = []  # Issues author pushed back on
        self.negotiated = []  # Issues with partial consensus

    def _extract_article(self):
        """Extract title, excerpt, and content from MDX file."""
        with open(self.article_path, 'r') as f:
            content = f.read()

        parts = content.split('---')
        frontmatter = parts[1]
        body = parts[2].strip()

        title_match = re.search(r'title:\s*"([^"]+)"', frontmatter)
        excerpt_match = re.search(r'excerpt:\s*"([^"]+)"', frontmatter)

        title = title_match.group(1) if title_match else "Unknown"
        excerpt = excerpt_match.group(1) if excerpt_match else ""

        return title, excerpt, body

    def start_mcp_servers(self):
        """Start MCP servers: 1 Author (Claude) + 4 Reviewers."""
        print(f"\n{'='*60}")
        print(f"Starting Author + 4 Reviewers (MCP)...")
        print(f"{'='*60}\n")

        author_model = "claude-sonnet-5"
        reviewer_models = [
            "gpt-4o-mini",
            "deepseek-chat",
            "gemini-3.5-flash",
            "llama-3-70b"
        ]

        # Start author
        server_script = "mcp_servers/author_server.py"
        try:
            process = subprocess.Popen(
                ["python3", server_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            self.server_processes[author_model] = process
            print(f"✓ Author (Claude) ready (PID: {process.pid})")
        except Exception as e:
            print(f"✗ Failed to start Author: {e}")

        # Start reviewers
        for model in reviewer_models:
            server_script = self._get_server_script_for_model(model)
            try:
                process = subprocess.Popen(
                    ["python3", server_script],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                self.server_processes[model] = process
                print(f"✓ Reviewer ({model}) ready (PID: {process.pid})")
            except Exception as e:
                print(f"✗ Failed to start {model}: {e}")

        self.servers = list(self.server_processes.keys())
        print(f"\nLoaded {len(self.servers)} participants\n")

    def _get_server_script_for_model(self, model: str) -> str:
        """Get the MCP server script path for a model."""
        server_map = {
            "claude-sonnet-5": "mcp_servers/author_server.py",
            "gpt-4o-mini": "mcp_servers/openai_server.py",
            "deepseek-chat": "mcp_servers/deepseek_server.py",
            "gemini-3.5-flash": "mcp_servers/gemini_server.py",
            "llama-3-70b": "mcp_servers/together_server.py",
        }
        return server_map.get(model, f"mcp_servers/{model.replace('-', '_')}_server.py")

    def run_roundtable(self):
        """Run roundtable debate through dimensions in priority order."""
        print(f"{'='*60}")
        print(f"Roundtable: {self.title}")
        print(f"{'='*60}\n")

        for dimension_idx, dimension in enumerate(DIMENSION_PRIORITY, 1):
            print(f"\n{'─'*60}")
            print(f"Dimension {dimension_idx}/6: {dimension}")
            print(f"{'─'*60}\n")

            self._debate_dimension(dimension)

        # Generate report
        report = self._generate_report()
        return report

    def _debate_dimension(self, dimension: str):
        """Debate one dimension: reviewers → author → optional rebuttals."""
        reviewer_models = [m for m in self.servers if m != "claude-sonnet-5"]
        author_model = "claude-sonnet-5"

        # Phase 1: Reviewers scan for this dimension
        print(f"Phase 1: Reviewers identify {dimension} issues...")
        reviewer_concerns = {}

        for reviewer in reviewer_models:
            try:
                request = {
                    "method": "scan_article",
                    "params": {
                        "title": self.title,
                        "content": self.content[:3000]
                    }
                }
                response = self._call_mcp_server(reviewer, request)
                findings = response.get("findings", [])

                # Filter to this dimension
                dimension_findings = [f for f in findings if f.get("dimension") == dimension]
                if dimension_findings:
                    reviewer_concerns[reviewer] = dimension_findings
                    print(f"  {reviewer}: {len(dimension_findings)} {dimension} issues")
            except Exception as e:
                print(f"  {reviewer}: error — {e}")

        if not reviewer_concerns:
            print(f"  ✓ No {dimension} issues found")
            return

        # Phase 2: Author responds to all concerns
        print(f"\nPhase 2: Author responds to {len(reviewer_concerns)} reviewers...")
        try:
            request = {
                "method": "respond_to_roundtable",
                "params": {
                    "title": self.title,
                    "content": self.content[:3000],
                    "dimension": dimension,
                    "reviewer_concerns_by_model": reviewer_concerns
                }
            }
            response = self._call_mcp_server(author_model, request)
            author_response = response.get("response", "")

            # Parse author stances
            self._parse_author_response(author_response, dimension, reviewer_concerns)

            print(f"  ✓ Author responded")

            # Log to transcript
            self.transcript.append({
                "dimension": dimension,
                "reviewers_raised": list(reviewer_concerns.keys()),
                "reviewer_concerns_count": sum(len(c) for c in reviewer_concerns.values()),
                "author_response": author_response
            })

        except Exception as e:
            print(f"  Error: {e}")

    def _parse_author_response(self, response_text: str, dimension: str, reviewer_concerns: dict):
        """Parse author's response to extract stances."""
        # Simple parsing: look for STANCE: lines
        stances = re.findall(r'STANCE:\s*(CONCEDE|DEFEND|NEGOTIATE)', response_text)
        proposed_fixes = re.findall(r'PROPOSED_FIX:\s*(.+?)(?=\n\n|\nCONCERN:|$)', response_text, re.DOTALL)

        # Track outcomes
        for stance, fix in zip(stances, proposed_fixes + [None] * len(stances)):
            if stance == "CONCEDE":
                self.agreed_changes.append({
                    "dimension": dimension,
                    "stance": stance,
                    "proposed_fix": fix.strip() if fix else None
                })
            elif stance == "DEFEND":
                self.open_disagreements.append({
                    "dimension": dimension,
                    "stance": stance
                })
            elif stance == "NEGOTIATE":
                self.negotiated.append({
                    "dimension": dimension,
                    "stance": stance,
                    "proposed_fix": fix.strip() if fix else None
                })

    def _call_mcp_server(self, model: str, request: dict) -> dict:
        """Call a model's MCP server and get response."""
        process = self.server_processes.get(model)
        if not process:
            raise Exception(f"No process for {model}")

        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()

        response_line = process.stdout.readline()
        if not response_line:
            raise Exception(f"No response from {model}")

        return json.loads(response_line)

    def _generate_report(self):
        """Generate final roundtable report."""
        print(f"\n{'='*60}")
        print("Roundtable Summary")
        print(f"{'='*60}\n")

        print(f"Agreed Changes: {len(self.agreed_changes)}")
        print(f"Negotiated: {len(self.negotiated)}")
        print(f"Open Disagreements: {len(self.open_disagreements)}")
        print()

        # Breakdown by dimension
        print("By Dimension:")
        for dim in DIMENSION_PRIORITY:
            agreed = len([c for c in self.agreed_changes if c['dimension'] == dim])
            neg = len([c for c in self.negotiated if c['dimension'] == dim])
            open_d = len([c for c in self.open_disagreements if c['dimension'] == dim])
            if agreed + neg + open_d > 0:
                print(f"  {dim}: {agreed} agreed + {neg} negotiated + {open_d} open")

        report = {
            "article": self.title,
            "mode": "author_vs_reviewers",
            "agreed_changes": self.agreed_changes,
            "negotiated_changes": self.negotiated,
            "open_disagreements": self.open_disagreements,
            "transcript": self.transcript,
            "action_items": len(self.agreed_changes) + len(self.negotiated)
        }
        return report

    def stop_mcp_servers(self):
        """Shutdown all MCP servers."""
        for model, process in self.server_processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()

def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp_author_vs_reviewers.py <article_path>")
        sys.exit(1)

    article_path = sys.argv[1]
    orchestrator = AuthorVsReviewersOrchestrator(article_path)

    try:
        orchestrator.start_mcp_servers()
        report = orchestrator.run_roundtable()

        # Save report
        report_path = Path(article_path).parent / f"{Path(article_path).stem}_roundtable_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nReport saved to: {report_path}")
        print(f"Action Items: {report['action_items']}")

    finally:
        orchestrator.stop_mcp_servers()

if __name__ == "__main__":
    main()
