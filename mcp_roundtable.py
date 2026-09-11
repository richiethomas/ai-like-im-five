#!/usr/bin/env python3
"""
Clean MCP Roundtable Orchestrator: Author vs 4 Reviewers debate.
Back-and-forth rounds until consensus or 100 rounds.
"""

import json
import subprocess
import sys
import os
import re
from pathlib import Path
from typing import Optional

DIMENSION_PRIORITY = [
    "CORRECTNESS",
    "CLARITY",
    "COMPLETENESS",
    "CONSISTENCY",
    "PEDAGOGY",
    "CLICHÉS"
]

REVIEWER_MODELS = [
    ("openai", "gpt-4o-mini"),
    ("deepseek", "deepseek-chat"),
    ("gemini", "gemini-3.5-flash"),
    ("together", "llama-3-70b")
]

class RoundtableOrchestrator:
    def __init__(self, article_path: str, max_rounds: int = 100):
        self.article_path = article_path
        self.max_rounds = max_rounds
        self.title, self.excerpt, self.content = self._extract_article()

        self.author_process = None
        self.reviewer_processes = {}
        self.transcript = []
        self.agreed_changes = []
        self.open_disagreements = []
        self.negotiated = []

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

    def start_servers(self):
        """Start author + 4 reviewer MCP servers."""
        print(f"\n{'='*60}")
        print("Starting MCP Roundtable Servers")
        print(f"{'='*60}\n")

        # Start author
        try:
            self.author_process = subprocess.Popen(
                ["python3", "mcp_author.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            print(f"✓ Author ready (PID: {self.author_process.pid})")
        except Exception as e:
            print(f"✗ Failed to start Author: {e}")
            return False

        # Start reviewers
        for env_name, model_name in REVIEWER_MODELS:
            try:
                env = os.environ.copy()
                env["REVIEWER_MODEL"] = env_name
                process = subprocess.Popen(
                    ["python3", "mcp_reviewer.py"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=env
                )
                self.reviewer_processes[model_name] = process
                print(f"✓ Reviewer ({model_name}) ready (PID: {process.pid})")
            except Exception as e:
                print(f"✗ Failed to start {model_name}: {e}")
                return False

        print()
        return True

    def call_server(self, process, request: dict) -> dict:
        """Send request to MCP server, get response."""
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()

        response_line = process.stdout.readline()
        if not response_line:
            raise Exception(f"No response from server")

        return json.loads(response_line)

    def run_roundtable(self):
        """Run continuous back-and-forth roundtable debate."""
        print(f"{'='*60}")
        print(f"Roundtable: {self.title}")
        print(f"Max rounds: {self.max_rounds}")
        print(f"{'='*60}\n")

        round_num = 1
        dimension_idx = 0
        dimension_state = {}

        while round_num <= self.max_rounds and dimension_idx < len(DIMENSION_PRIORITY):
            dimension = DIMENSION_PRIORITY[dimension_idx]

            # Start new dimension
            if dimension not in dimension_state:
                print(f"Round {round_num}: {dimension} (Initial scan)")
                dimension_state[dimension] = {
                    "reviewer_findings": {},
                    "author_responses": [],
                    "debate_rounds": 0
                }

                # All reviewers scan
                for model_name, process in self.reviewer_processes.items():
                    try:
                        request = {
                            "method": "scan_article",
                            "params": {"title": self.title, "content": self.content[:3000]}
                        }
                        response = self.call_server(process, request)
                        findings = response.get("findings", [])
                        if findings:
                            dimension_state[dimension]["reviewer_findings"][model_name] = findings
                            print(f"  {model_name}: {len(findings)} issues")
                    except Exception as e:
                        print(f"  {model_name}: error — {e}")

                if not dimension_state[dimension]["reviewer_findings"]:
                    print(f"  ✓ No issues — next dimension")
                    dimension_idx += 1
                    round_num += 1
                    continue

                # Author responds to initial findings
                round_num += 1
                if round_num > self.max_rounds:
                    break

                print(f"Round {round_num}: {dimension} (Author response)")
                try:
                    request = {
                        "method": "respond_to_roundtable",
                        "params": {
                            "title": self.title,
                            "content": self.content[:3000],
                            "dimension": dimension,
                            "reviewer_concerns_by_model": dimension_state[dimension]["reviewer_findings"]
                        }
                    }
                    response = self.call_server(self.author_process, request)
                    author_response = response.get("response", "")
                    dimension_state[dimension]["author_responses"].append(author_response)
                    self._parse_author_response(author_response, dimension)
                    print(f"  Author responded")
                except Exception as e:
                    print(f"  Author error: {e}")

                # Check if further debate needed
                if "DEFEND" not in author_response:
                    print(f"  → Author conceded — next dimension")
                    dimension_idx += 1
                    round_num += 1
                    continue

            else:
                # Continued debate on this dimension
                print(f"Round {round_num}: {dimension} (Reviewer rebuttal)")
                dimension_state[dimension]["debate_rounds"] += 1

                # Reviewers respond to author's last response
                author_last = dimension_state[dimension]["author_responses"][-1]
                rebuttals = {}

                for model_name, process in self.reviewer_processes.items():
                    try:
                        concerns = dimension_state[dimension]["reviewer_findings"].get(model_name, [])
                        if not concerns:
                            continue

                        request = {
                            "method": "debate_stance",
                            "params": {
                                "title": self.title,
                                "content": self.content[:3000],
                                "dimension": dimension,
                                "your_concerns": concerns,
                                "author_response": author_last
                            }
                        }
                        response = self.call_server(process, request)
                        rebuttal = response.get("response", "")
                        if rebuttal:
                            rebuttals[model_name] = rebuttal
                            print(f"  {model_name}: submitted response")
                    except Exception as e:
                        print(f"  {model_name}: error — {e}")

                if not rebuttals:
                    print(f"  → Reviewers accepted — next dimension")
                    dimension_idx += 1
                    round_num += 1
                    continue

                # Author responds to rebuttals
                round_num += 1
                if round_num > self.max_rounds:
                    break

                print(f"Round {round_num}: {dimension} (Author responds to rebuttals)")
                rebuttals_text = "\n\n".join([f"**{m}**: {r}" for m, r in rebuttals.items()])

                try:
                    request = {
                        "method": "respond_to_rebuttals",
                        "params": {
                            "title": self.title,
                            "content": self.content[:3000],
                            "dimension": dimension,
                            "reviewer_rebuttals": rebuttals_text
                        }
                    }
                    response = self.call_server(self.author_process, request)
                    author_response = response.get("response", "")
                    dimension_state[dimension]["author_responses"].append(author_response)
                    self._parse_author_response(author_response, dimension)
                    print(f"  Author responded")
                except Exception as e:
                    print(f"  Author error: {e}")

                # Move to next dimension if too many rounds or all conceded
                if dimension_state[dimension]["debate_rounds"] >= 5 or "DEFEND" not in author_response:
                    print(f"  → Moving to next dimension")
                    dimension_idx += 1

            round_num += 1

        # Generate report
        status = "COMPLETE" if dimension_idx >= len(DIMENSION_PRIORITY) else "PARTIAL"
        print(f"\n{'='*60}")
        print(f"Roundtable Summary ({status})")
        print(f"{'='*60}\n")

        print(f"Rounds used: {round_num - 1}/{self.max_rounds}")
        print(f"Dimensions covered: {dimension_idx}/{len(DIMENSION_PRIORITY)}")
        print(f"Agreed changes: {len(self.agreed_changes)}")
        print(f"Negotiated: {len(self.negotiated)}")
        print(f"Open disagreements: {len(self.open_disagreements)}")
        print()

        report = {
            "article": self.title,
            "rounds_used": round_num - 1,
            "dimensions_covered": dimension_idx,
            "status": status,
            "agreed_changes": self.agreed_changes,
            "negotiated": self.negotiated,
            "open_disagreements": self.open_disagreements,
            "action_items": len(self.agreed_changes) + len(self.negotiated)
        }

        return report

    def _parse_author_response(self, response: str, dimension: str):
        """Extract CONCEDE/DEFEND/NEGOTIATE stances from author response."""
        stances = re.findall(r'STANCE:\s*(CONCEDE|DEFEND|NEGOTIATE)', response)
        fixes = re.findall(r'PROPOSED_FIX:\s*(.+?)(?=\n\n|$)', response, re.DOTALL)

        for i, stance in enumerate(stances):
            fix = fixes[i].strip() if i < len(fixes) else None

            if stance == "CONCEDE":
                self.agreed_changes.append({
                    "dimension": dimension,
                    "stance": stance,
                    "fix": fix
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
                    "fix": fix
                })

    def stop_servers(self):
        """Terminate all MCP servers."""
        if self.author_process:
            self.author_process.terminate()
            self.author_process.wait(timeout=5)

        for process in self.reviewer_processes.values():
            process.terminate()
            process.wait(timeout=5)

    def save_report(self, report: dict):
        """Save report to JSON file."""
        report_path = Path(self.article_path).parent / f"{Path(self.article_path).stem}_roundtable_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved: {report_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp_roundtable.py <article_path>")
        sys.exit(1)

    article_path = sys.argv[1]
    orchestrator = RoundtableOrchestrator(article_path)

    try:
        if not orchestrator.start_servers():
            sys.exit(1)

        report = orchestrator.run_roundtable()
        orchestrator.save_report(report)

    finally:
        orchestrator.stop_servers()


if __name__ == "__main__":
    main()
