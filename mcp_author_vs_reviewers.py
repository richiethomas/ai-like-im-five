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
    def __init__(self, article_path: str, max_rounds: int = 100):
        self.article_path = article_path
        self.max_rounds = max_rounds

        # Load article
        self.title, self.excerpt, self.content = self._extract_article()

        # MCP server processes
        self.servers = {}
        self.server_processes = {}

        # Roundtable state: continuous debate, not phase-based
        self.current_dimension_idx = 0  # Track which dimension we're actively debating
        self.dimension_coverage = {dim: {"rounds": 0, "resolved": False} for dim in DIMENSION_PRIORITY}
        self.transcript = []
        self.agreed_changes = []  # Issues author conceded
        self.open_disagreements = []  # Issues author pushed back on
        self.negotiated = []  # Issues with partial consensus
        self.reviewer_positions = {}  # round -> {reviewer: position_text}
        self.author_positions = {}  # round -> position_text

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
        """Run continuous roundtable debate across all dimensions for up to 100 rounds."""
        print(f"{'='*60}")
        print(f"Roundtable: {self.title}")
        print(f"Max rounds: {self.max_rounds}")
        print(f"{'='*60}\n")

        reviewer_models = [m for m in self.servers if m != "claude-sonnet-5"]
        author_model = "claude-sonnet-5"

        round_num = 1
        while round_num <= self.max_rounds:
            current_dimension = DIMENSION_PRIORITY[self.current_dimension_idx]
            print(f"\nRound {round_num}: {current_dimension}")

            if round_num == 1:
                # Round 1: Reviewers raise initial issues on current dimension
                reviewer_concerns = self._reviewers_raise_issues(reviewer_models, current_dimension)
                if not reviewer_concerns:
                    print(f"  ✓ No issues found, moving to next dimension")
                    self.current_dimension_idx += 1
                    if self.current_dimension_idx >= len(DIMENSION_PRIORITY):
                        print(f"\n✓ All dimensions reviewed")
                        break
                    continue

                self.reviewer_positions[round_num] = reviewer_concerns
                self._parse_reviewer_concerns(reviewer_concerns, current_dimension)

            # Author responds to current reviewer positions
            author_response = self._author_responds(author_model, current_dimension, self.reviewer_positions.get(round_num))
            self.author_positions[round_num] = author_response
            self._parse_author_response(author_response, current_dimension)

            # Check if reviewers want to rebut
            if round_num < self.max_rounds:
                reviewer_rebuttals = self._reviewers_rebut(reviewer_models, current_dimension, author_response)
                if reviewer_rebuttals:
                    self.reviewer_positions[round_num + 1] = reviewer_rebuttals
                else:
                    # No rebuttals: move to next dimension
                    self.dimension_coverage[current_dimension]["resolved"] = True
                    self.current_dimension_idx += 1
                    if self.current_dimension_idx >= len(DIMENSION_PRIORITY):
                        print(f"\n✓ All dimensions reviewed in {round_num} rounds")
                        break

            round_num += 1

        # Generate report
        report = self._generate_report()
        return report

    def _reviewers_raise_issues(self, reviewers: list, dimension: str) -> dict:
        """Reviewers scan for issues in current dimension."""
        concerns = {}
        for reviewer in reviewers:
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
                dimension_findings = [f for f in findings if f.get("dimension") == dimension]
                if dimension_findings:
                    concerns[reviewer] = dimension_findings
                    print(f"  {reviewer}: {len(dimension_findings)} issues")
            except Exception as e:
                print(f"  {reviewer}: error — {e}")
        return concerns

    def _parse_reviewer_concerns(self, concerns_by_model: dict, dimension: str):
        """Track reviewer concerns for later analysis."""
        pass

    def _author_responds(self, author_model: str, dimension: str, reviewer_concerns: dict) -> str:
        """Author responds to current reviewer concerns."""
        try:
            request = {
                "method": "respond_to_roundtable",
                "params": {
                    "title": self.title,
                    "content": self.content[:3000],
                    "dimension": dimension,
                    "reviewer_concerns_by_model": reviewer_concerns or {}
                }
            }
            response = self._call_mcp_server(author_model, request)
            print(f"  Author responded")
            return response.get("response", "")
        except Exception as e:
            print(f"  Author error: {e}")
            return ""

    def _reviewers_rebut(self, reviewers: list, dimension: str, author_response: str) -> dict:
        """Reviewers respond to author's defense."""
        # For now, simplified: if author gave DEFEND responses, reviewers can push back
        # Real implementation would parse author stances and rebut accordingly
        if "DEFEND" in author_response:
            # Could trigger rebuttal debate
            return {}
        return {}

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
        """Generate final roundtable report with coverage analysis."""
        print(f"\n{'='*60}")
        print("Roundtable Summary")
        print(f"{'='*60}\n")

        print(f"Agreed Changes: {len(self.agreed_changes)}")
        print(f"Negotiated: {len(self.negotiated)}")
        print(f"Open Disagreements: {len(self.open_disagreements)}")
        print()

        # Dimension coverage
        print("Dimension Coverage:")
        covered = []
        deferred = []
        for dim in DIMENSION_PRIORITY:
            agreed = len([c for c in self.agreed_changes if c['dimension'] == dim])
            neg = len([c for c in self.negotiated if c['dimension'] == dim])
            open_d = len([c for c in self.open_disagreements if c['dimension'] == dim])
            total = agreed + neg + open_d

            status = "✓ Covered" if self.dimension_coverage[dim]["resolved"] else "○ Deferred"
            print(f"  {dim}: {total} issues — {status}")

            if self.dimension_coverage[dim]["resolved"]:
                covered.append(dim)
            else:
                deferred.append(dim)

        print()
        if deferred:
            print(f"Deferred dimensions (need more rounds): {', '.join(deferred)}")
            print("Recommendation: Further discussion warranted on deferred topics")
        else:
            print("All dimensions covered")

        report = {
            "article": self.title,
            "mode": "author_vs_reviewers",
            "rounds_used": len(self.author_positions),
            "max_rounds": self.max_rounds,
            "dimensions_covered": covered,
            "dimensions_deferred": deferred,
            "agreed_changes": self.agreed_changes,
            "negotiated_changes": self.negotiated,
            "open_disagreements": self.open_disagreements,
            "transcript": self.transcript,
            "action_items": len(self.agreed_changes) + len(self.negotiated),
            "needs_further_discussion": len(deferred) > 0
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
