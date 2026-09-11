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
import time
from pathlib import Path
from typing import Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler('/tmp/roundtable_debug.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
log = logging.getLogger(__name__)

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
        log.info("Starting MCP servers...")
        print(f"\n{'='*60}")
        print("Starting MCP Roundtable Servers")
        print(f"{'='*60}\n")

        # Start author
        try:
            log.debug("Starting author server...")
            self.author_process = subprocess.Popen(
                ["python3", "mcp_author.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            log.info(f"Author server started (PID: {self.author_process.pid})")
            print(f"✓ Author ready (PID: {self.author_process.pid})")
        except Exception as e:
            log.error(f"Failed to start Author: {e}")
            print(f"✗ Failed to start Author: {e}")
            return False

        # Start reviewers
        for env_name, model_name in REVIEWER_MODELS:
            try:
                log.debug(f"Starting reviewer server for {model_name}...")
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
                log.info(f"Reviewer {model_name} started (PID: {process.pid})")
                print(f"✓ Reviewer ({model_name}) ready (PID: {process.pid})")
            except Exception as e:
                log.error(f"Failed to start {model_name}: {e}")
                print(f"✗ Failed to start {model_name}: {e}")
                return False

        log.info(f"All {len(self.reviewer_processes) + 1} servers started successfully")
        print()
        return True

    def call_server(self, process, request: dict, timeout_sec: float = 30) -> dict:
        """Send request to MCP server, get response."""
        method = request.get("method", "unknown")
        log.debug(f"  Calling server method={method}")

        start = time.time()
        try:
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            log.debug(f"  Request sent ({time.time() - start:.2f}s)")

            response_line = process.stdout.readline()
            elapsed = time.time() - start
            log.debug(f"  Response received ({elapsed:.2f}s total)")

            if not response_line:
                log.error(f"  No response from server after {elapsed:.2f}s")
                raise Exception(f"No response from server (timeout after {elapsed:.1f}s)")

            response = json.loads(response_line)
            log.debug(f"  Response parsed successfully")
            return response
        except Exception as e:
            elapsed = time.time() - start
            log.error(f"  Server call failed after {elapsed:.2f}s: {e}")
            raise

    def run_roundtable(self):
        """Run continuous back-and-forth roundtable debate."""
        log.info(f"{'='*60}")
        log.info(f"Starting roundtable: {self.title}")
        log.info(f"Max rounds: {self.max_rounds}")
        log.info(f"{'='*60}")

        print(f"{'='*60}")
        print(f"Roundtable: {self.title}")
        print(f"Max rounds: {self.max_rounds}")
        print(f"{'='*60}\n")

        round_num = 1
        dimension_idx = 0
        dimension_state = {}

        log.debug(f"Starting main loop with {len(DIMENSION_PRIORITY)} dimensions")

        while round_num <= self.max_rounds and dimension_idx < len(DIMENSION_PRIORITY):
            dimension = DIMENSION_PRIORITY[dimension_idx]
            log.info(f"Round {round_num}: Dimension {dimension_idx + 1}/6 ({dimension})")

            # Start new dimension
            if dimension not in dimension_state:
                log.debug(f"  Starting new dimension: {dimension}")
                print(f"Round {round_num}: {dimension} (Initial scan)")
                dimension_state[dimension] = {
                    "reviewer_findings": {},
                    "author_responses": [],
                    "debate_rounds": 0
                }

                # All reviewers scan in parallel
                log.debug(f"  Calling {len(self.reviewer_processes)} reviewers for scan_article (parallel)")
                request = {
                    "method": "scan_article",
                    "params": {"title": self.title, "content": self.content[:3000]}
                }

                def scan_reviewer(model_name, process):
                    """Scan with one reviewer, return (model_name, findings, elapsed)."""
                    log.debug(f"    Scanning with {model_name}...")
                    start = time.time()
                    try:
                        response = self.call_server(process, request)
                        findings = response.get("findings", [])
                        elapsed = time.time() - start
                        return (model_name, findings, elapsed)
                    except Exception as e:
                        elapsed = time.time() - start
                        log.error(f"    {model_name}: error after {elapsed:.1f}s — {e}")
                        return (model_name, [], elapsed)

                # Run all scans concurrently
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {
                        executor.submit(scan_reviewer, model_name, process): model_name
                        for model_name, process in self.reviewer_processes.items()
                    }

                    for future in as_completed(futures):
                        model_name, findings, elapsed = future.result()
                        if findings:
                            dimension_state[dimension]["reviewer_findings"][model_name] = findings
                            log.info(f"    {model_name}: {len(findings)} issues ({elapsed:.1f}s)")
                            print(f"  {model_name}: {len(findings)} issues")
                        else:
                            log.debug(f"    {model_name}: no findings ({elapsed:.1f}s)")

                log.info(f"  All reviewers scanned in parallel")

                if not dimension_state[dimension]["reviewer_findings"]:
                    log.info(f"  No issues found — moving to next dimension")
                    print(f"  ✓ No issues — next dimension")
                    dimension_idx += 1
                    round_num += 1
                    continue

                # Author responds to initial findings
                round_num += 1
                if round_num > self.max_rounds:
                    log.warning(f"Round limit reached, breaking")
                    break

                log.info(f"Round {round_num}: {dimension} (Author response)")
                print(f"Round {round_num}: {dimension} (Author response)")
                try:
                    start = time.time()
                    request = {
                        "method": "respond_to_roundtable",
                        "params": {
                            "title": self.title,
                            "content": self.content[:3000],
                            "dimension": dimension,
                            "reviewer_concerns_by_model": dimension_state[dimension]["reviewer_findings"]
                        }
                    }
                    log.debug(f"  Calling author.respond_to_roundtable...")
                    response = self.call_server(self.author_process, request)
                    author_response = response.get("response", "")
                    elapsed = time.time() - start

                    dimension_state[dimension]["author_responses"].append(author_response)
                    log.debug(f"  Parsing author response...")
                    self._parse_author_response(author_response, dimension)
                    log.info(f"  Author responded ({elapsed:.1f}s)")
                    print(f"  Author responded")
                except Exception as e:
                    log.error(f"  Author error: {e}")
                    print(f"  Author error: {e}")

                # Check if further debate needed
                if "DEFEND" not in author_response:
                    print(f"  → Author conceded — next dimension")
                    dimension_idx += 1
                    round_num += 1
                    continue

            else:
                # Continued debate on this dimension
                log.info(f"Round {round_num}: {dimension} (Reviewer rebuttal)")
                print(f"Round {round_num}: {dimension} (Reviewer rebuttal)")
                dimension_state[dimension]["debate_rounds"] += 1

                # Reviewers respond to author's last response (in parallel)
                author_last = dimension_state[dimension]["author_responses"][-1]
                rebuttals = {}

                def get_rebuttal(model_name, process):
                    """Get rebuttal from one reviewer."""
                    concerns = dimension_state[dimension]["reviewer_findings"].get(model_name, [])
                    if not concerns:
                        return (model_name, None)

                    try:
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
                        return (model_name, rebuttal)
                    except Exception as e:
                        log.error(f"  {model_name}: rebuttal error — {e}")
                        return (model_name, None)

                # Get rebuttals in parallel
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {
                        executor.submit(get_rebuttal, model_name, process): model_name
                        for model_name, process in self.reviewer_processes.items()
                    }

                    for future in as_completed(futures):
                        model_name, rebuttal = future.result()
                        if rebuttal:
                            rebuttals[model_name] = rebuttal
                            log.debug(f"  {model_name}: submitted rebuttal")
                            print(f"  {model_name}: submitted response")

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
    # 10 minute timeout for full roundtable
    orchestrator = RoundtableOrchestrator(article_path, max_rounds=100)

    try:
        if not orchestrator.start_servers():
            sys.exit(1)

        report = orchestrator.run_roundtable()
        orchestrator.save_report(report)

    finally:
        orchestrator.stop_servers()


if __name__ == "__main__":
    main()
