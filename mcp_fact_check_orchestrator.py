#!/usr/bin/env python3
"""
MCP-based fact-checking orchestrator.
Each model runs as an MCP server. Orchestrator manages debate rounds.
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
class Finding:
    claim: str
    issue: str
    severity: int
    dimension: str
    suggested_fix: Optional[str] = None
    model_id: Optional[str] = None
    round_number: int = 1

@dataclass
class DebateState:
    """Tracks debate state for a single claim across all models."""
    claim: str
    positions: dict  # model_id -> list of positions
    round_count: dict  # model_id -> count of times they've spoken

class MCPFactCheckOrchestrator:
    def __init__(self, article_path: str, max_rounds: int = 100, consensus_threshold: int = 5):
        self.article_path = article_path
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold

        # Load article
        self.title, self.excerpt, self.content = self._extract_article()

        # MCP server processes
        self.servers = {}
        self.server_processes = {}

        # Debate state
        self.findings_by_claim = defaultdict(lambda: defaultdict(list))
        self.debate_states = {}  # claim -> DebateState
        self.transcript = []
        self.cost_tracker = defaultdict(float)

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

    def start_mcp_servers(self, models: list[str]):
        """Start MCP servers for each model."""
        print(f"\n{'='*60}")
        print(f"Starting {len(models)} MCP servers...")
        print(f"{'='*60}\n")

        for model in models:
            # Start server process
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
                print(f"✓ Started {model} MCP server (PID: {process.pid})")
            except Exception as e:
                print(f"✗ Failed to start {model}: {e}")
                continue

        self.servers = {m: m for m in self.server_processes.keys()}
        print(f"\nLoaded {len(self.servers)} models: {', '.join(self.servers.keys())}\n")

    def _get_server_script_for_model(self, model: str) -> str:
        """Get the MCP server script path for a model."""
        # Map model names to server scripts
        server_map = {
            "claude-sonnet-5": "mcp_servers/claude_server.py",
            "gpt-4o-mini": "mcp_servers/openai_server.py",
            "deepseek-chat": "mcp_servers/deepseek_server.py",
            "gemini-3.5-flash": "mcp_servers/gemini_server.py",
            "llama-3-70b": "mcp_servers/together_server.py",
        }
        return server_map.get(model, f"mcp_servers/{model.replace('-', '_')}_server.py")

    def run_debate(self):
        """Main debate loop."""
        # Round 1: Initial fact-checks
        print(f"{'='*60}")
        print(f"Fact-checking: {self.title}")
        print(f"{'='*60}\n")

        print("Round 1: Initial fact-checks...")
        findings = self._initial_scan()

        # Debate rounds
        round_num = 2
        while round_num <= self.max_rounds:
            consensus, blockers = self._analyze_consensus()

            if len(blockers) == 0:
                print(f"\n✓ Consensus reached in round {round_num - 1} — terminating early to save tokens")
                break

            print(f"\nRound {round_num}: Debate round...")
            print(f"  Active disagreements: {len(blockers)}")

            self._debate_round(blockers, round_num)
            round_num += 1

        # Generate report
        consensus, blockers = self._analyze_consensus()
        report = self._generate_report(consensus, blockers)

        return report

    def _initial_scan(self) -> dict:
        """Round 1: Each model scans independently."""
        findings = {}

        for model in self.servers.keys():
            try:
                # Send scan request to MCP server
                request = {
                    "method": "scan_article",
                    "params": {
                        "title": self.title,
                        "content": self.content[:3000]
                    }
                }

                response = self._call_mcp_server(model, request)
                model_findings = response.get("findings", [])

                findings[model] = model_findings
                print(f"  {model}: {len(model_findings)} findings")

                # Track findings
                for finding_dict in model_findings:
                    finding = Finding(**finding_dict)
                    self.findings_by_claim[finding.claim][model].append(finding)

                    if finding.claim not in self.debate_states:
                        self.debate_states[finding.claim] = DebateState(
                            claim=finding.claim,
                            positions={m: [] for m in self.servers.keys()},
                            round_count={m: 0 for m in self.servers.keys()}
                        )

                    self.debate_states[finding.claim].positions[model].append(finding.issue)
                    self.debate_states[finding.claim].round_count[model] += 1

                    # Record in transcript
                    self.transcript.append({
                        "round": 1,
                        "model": model,
                        "type": "initial_scan",
                        "claim": finding.claim,
                        "issue": finding.issue,
                        "severity": finding.severity
                    })
            except Exception as e:
                print(f"  Error from {model}: {e}")
                findings[model] = []

        return findings

    def _debate_round(self, blocker_items: list, round_num: int):
        """Run a debate round on blocker items."""
        for blocker_claim in blocker_items[:3]:  # Top 3 blockers per round
            state = self.debate_states[blocker_claim]

            for model in self.servers.keys():
                other_positions = {m: pos for m, pos in state.positions.items() if m != model}

                if not other_positions:
                    continue

                try:
                    request = {
                        "method": "debate",
                        "params": {
                            "claim": blocker_claim,
                            "my_position": state.positions[model][-1] if state.positions[model] else "Not yet reviewed",
                            "other_positions": other_positions
                        }
                    }

                    response = self._call_mcp_server(model, request)
                    response_text = response.get("response", "")

                    state.positions[model].append(response_text)
                    state.round_count[model] += 1

                    # Record in transcript
                    self.transcript.append({
                        "round": round_num,
                        "model": model,
                        "type": "debate_response",
                        "claim": blocker_claim,
                        "response": response_text
                    })
                except Exception as e:
                    print(f"  Debate error from {model}: {e}")

    def _call_mcp_server(self, model: str, request: dict) -> dict:
        """Call a model's MCP server and get response."""
        process = self.server_processes.get(model)
        if not process:
            raise Exception(f"No process for {model}")

        # Send request
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()

        # Read response
        response_line = process.stdout.readline()
        if not response_line:
            raise Exception(f"No response from {model}")

        return json.loads(response_line)

    def _analyze_consensus(self):
        """Analyze findings to detect consensus vs blockers."""
        consensus = []
        blockers = []

        for claim, state in self.debate_states.items():
            # Check if enough models have weighed in
            models_with_positions = {m: pos for m, pos in state.positions.items() if pos}

            if len(models_with_positions) >= 3:
                # Consensus = 3+ models agreeing
                consensus.append(claim)
            elif max(state.round_count.values()) >= self.consensus_threshold:
                # Blocker = 5+ rounds per model with no agreement
                blockers.append(claim)

        return consensus, blockers

    def _generate_report(self, consensus, blockers):
        """Generate final report."""
        report = {
            "article": self.title,
            "excerpt": self.excerpt,
            "status": "PASS" if not consensus else "REVIEW",
            "findings": {
                "consensus": consensus,
                "blockers": blockers
            },
            "transcript": self.transcript,
            "cost_estimate": sum(self.cost_tracker.values())
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
        print("Usage: python mcp_fact_check_orchestrator.py <article_path>")
        sys.exit(1)

    article_path = sys.argv[1]
    models = [
        "claude-sonnet-5",
        "gpt-4o-mini",
        "deepseek-chat",
        "gemini-3.5-flash",
        "llama-3-70b"
    ]

    orchestrator = MCPFactCheckOrchestrator(article_path)

    try:
        orchestrator.start_mcp_servers(models)
        report = orchestrator.run_debate()

        # Save report
        report_path = Path(article_path).parent / f"{Path(article_path).stem}_mcp_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nReport saved to: {report_path}")
        print(f"Status: {report['status']}")
        print(f"Findings: {report['findings']}")

    finally:
        orchestrator.stop_mcp_servers()

if __name__ == "__main__":
    main()
