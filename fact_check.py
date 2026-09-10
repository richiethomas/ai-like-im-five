#!/usr/bin/env python3
"""
Multi-LLM fact-checking orchestrator for blog articles.
Runs parallel fact-checks with debate rounds until consensus or 100 rounds max.
"""

import json
import os
import sys
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import anthropic
import openai
from collections import defaultdict

# Configure API clients
try:
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
except Exception as e:
    print(f"Warning: Claude initialization failed: {e}")
    anthropic_client = None

try:
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    print(f"Warning: OpenAI initialization failed: {e}")
    openai_client = None

# DeepSeek: use OpenAI-compatible endpoint
# Set DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL in env, or leave blank for placeholder
# Gemini: requires separate google.generativeai setup
# Llama: use Together AI (TOGETHER_API_KEY) or Groq (GROQ_API_KEY)

@dataclass
class FactCheckFinding:
    """A single finding from a fact-check."""
    claim: str
    issue: str
    severity: int  # 1-10
    suggested_fix: Optional[str] = None
    model_id: Optional[str] = None
    round_number: int = 0

@dataclass
class ConsensusItem:
    """An item where all models agree."""
    claim: str
    issue: str
    severity: int
    suggested_fix: Optional[str]
    models_agreed: list[str]

@dataclass
class BlockerItem:
    """An item with no consensus after 5+ rounds per model."""
    claim: str
    positions: dict  # model_id -> list of positions taken
    rounds_debated: int

class FactCheckOrchestrator:
    """Orchestrates fact-checking across multiple LLMs."""

    def __init__(self, budget_per_article: float = 10.0):
        self.budget_per_article = budget_per_article
        self.cost_tracker = defaultdict(float)
        self.models = ["claude-3-5-sonnet-20241022", "gpt-4o-mini", "deepseek-chat", "gemini-2.0-flash"]
        self.max_rounds = 100
        self.consensus_threshold_rounds = 5
        self.debate_history = []  # Track all debate messages
        self.findings_by_claim = defaultdict(lambda: defaultdict(list))  # claim -> model_id -> findings
        self.claim_round_counts = defaultdict(lambda: defaultdict(int))  # claim -> model_id -> count

    def extract_article_content(self, article_path: str) -> tuple[str, str, str]:
        """Extract title, excerpt, and content from MDX file."""
        with open(article_path, 'r') as f:
            content = f.read()

        # Parse frontmatter
        parts = content.split('---')
        frontmatter = parts[1]
        body = parts[2].strip()

        # Extract title and excerpt from frontmatter
        title_match = re.search(r'title:\s*"([^"]+)"', frontmatter)
        excerpt_match = re.search(r'excerpt:\s*"([^"]+)"', frontmatter)

        title = title_match.group(1) if title_match else "Unknown"
        excerpt = excerpt_match.group(1) if excerpt_match else ""

        return title, excerpt, body

    def run_fact_check(self, article_path: str) -> dict:
        """Run complete fact-check pipeline for an article."""
        title, excerpt, content = self.extract_article_content(article_path)

        print(f"\n{'='*60}")
        print(f"Fact-checking: {title}")
        print(f"{'='*60}\n")

        # Round 1: Initial fact-checks from all models
        print("Round 1: Initial fact-checks...")
        initial_findings = {}
        for model in self.models:
            findings = self._initial_fact_check(model, title, content)
            initial_findings[model] = findings
            print(f"  {model}: {len(findings)} findings")

        # Main debate loop
        round_num = 2
        while round_num <= self.max_rounds:
            # Check if consensus reached
            consensus_items, blocker_items = self._analyze_consensus(initial_findings)
            if len(blocker_items) == 0:
                print(f"\n✓ Consensus reached in round {round_num - 1}")
                break

            print(f"\nRound {round_num}: Debate round...")
            print(f"  Active disagreements: {len(blocker_items)}")

            # Run debate round
            debate_occurred = self._debate_round(blocker_items, initial_findings)
            if not debate_occurred:
                break

            round_num += 1

        # Generate final report
        consensus_items, blocker_items = self._analyze_consensus(initial_findings)
        report = self._generate_report(title, excerpt, consensus_items, blocker_items)

        return report

    def _initial_fact_check(self, model: str, title: str, content: str) -> list[FactCheckFinding]:
        """Run initial fact-check from a single model."""
        prompt = f"""You are a rigorous technical fact-checker reviewing a blog article about deep learning and computer vision.

Article: {title}

Content:
{content[:3000]}...

Your task: Identify the 3-5 most important factual errors, oversimplifications, or misleading claims in this article. Focus ONLY on significant issues that would mislead readers. Ignore trivial nitpicks.

For each issue found, respond with:
CLAIM: [the specific claim being fact-checked]
ISSUE: [what's wrong with it]
SEVERITY: [1-10, where 10 is most severe]
FIX: [suggested correction]

Be direct and rigorous. Do not include minor issues."""

        findings = []
        try:
            if model == "claude-3-5-sonnet-20241022":
                response = anthropic_client.messages.create(
                    model=model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.content[0].text
                self.cost_tracker[model] += 0.5  # Rough estimate
            elif model == "gpt-4o-mini":
                response = openai_client.chat.completions.create(
                    model=model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.choices[0].message.content
                self.cost_tracker[model] += 0.3  # Rough estimate
            else:
                # Placeholder for DeepSeek, Gemini, Llama
                text = f"[{model} placeholder response]"
                self.cost_tracker[model] += 0.2

            # Parse findings from response
            findings = self._parse_findings(text, model)
        except Exception as e:
            print(f"  Error from {model}: {e}")

        return findings

    def _parse_findings(self, text: str, model: str) -> list[FactCheckFinding]:
        """Parse fact-check findings from model response."""
        findings = []

        # Split by "CLAIM:" to get individual findings
        parts = text.split("CLAIM:")
        for part in parts[1:]:  # Skip first part before first CLAIM
            try:
                lines = part.strip().split("\n")
                claim = lines[0].strip() if lines else ""

                issue = ""
                severity = 5
                fix = ""

                for line in lines[1:]:
                    if line.startswith("ISSUE:"):
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
                    findings.append(FactCheckFinding(
                        claim=claim,
                        issue=issue,
                        severity=severity,
                        suggested_fix=fix or None,
                        model_id=model,
                        round_number=1
                    ))
                    self.findings_by_claim[claim][model].append(
                        FactCheckFinding(claim=claim, issue=issue, severity=severity,
                                       suggested_fix=fix, model_id=model, round_number=1)
                    )
                    self.claim_round_counts[claim][model] += 1
            except:
                continue

        return findings

    def _debate_round(self, blocker_items: list[BlockerItem], all_findings: dict) -> bool:
        """Run a debate round on blocker items."""
        if not blocker_items:
            return False

        for blocker in blocker_items[:3]:  # Limit to top 3 blockers per round
            # Ask each model to respond to others' positions
            for model in self.models:
                positions = blocker.positions.get(model, [])
                other_positions = {m: pos for m, pos in blocker.positions.items() if m != model}

                if not other_positions:
                    continue

                prompt = f"""You are fact-checking this claim from a blog article: "{blocker.claim}"

Your previous position on this:
{positions[-1] if positions else "Not yet reviewed"}

Other models' positions:
{json.dumps(other_positions, indent=2)}

Do you maintain your position, or do you agree/disagree with any of the other models? Be specific about which points you accept or reject and why. Focus on technical accuracy."""

                try:
                    if model == "claude-3-5-sonnet-20241022":
                        response = anthropic_client.messages.create(
                            model=model,
                            max_tokens=500,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        response_text = response.content[0].text
                    elif model == "gpt-4o-mini":
                        response = openai_client.chat.completions.create(
                            model=model,
                            max_tokens=500,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        response_text = response.choices[0].message.content
                    else:
                        response_text = f"[{model} response]"

                    blocker.positions[model].append(response_text)
                    self.claim_round_counts[blocker.claim][model] += 1
                except Exception as e:
                    print(f"  Debate error from {model}: {e}")

        return True

    def _analyze_consensus(self, all_findings: dict) -> tuple[list[ConsensusItem], list[BlockerItem]]:
        """Analyze findings across models to identify consensus and blockers."""
        consensus_items = []
        blocker_items = []

        # Group by claim
        claims_seen = set()
        for model_findings in all_findings.values():
            for finding in model_findings:
                claims_seen.add(finding.claim)

        for claim in claims_seen:
            # Get all models' findings for this claim
            models_addressing = {}
            for model in self.models:
                if claim in self.findings_by_claim and model in self.findings_by_claim[claim]:
                    models_addressing[model] = self.findings_by_claim[claim][model]

            if len(models_addressing) >= 3:  # Consensus = 3+ models agree
                # Check if severity is consistent
                severities = [f.severity for findings in models_addressing.values() for f in findings]
                avg_severity = sum(severities) / len(severities) if severities else 5

                if len(set(int(s) for s in severities)) == 1:  # All same severity
                    consensus_items.append(ConsensusItem(
                        claim=claim,
                        issue=models_addressing[list(models_addressing.keys())[0]][0].issue,
                        severity=int(avg_severity),
                        suggested_fix=models_addressing[list(models_addressing.keys())[0]][0].suggested_fix,
                        models_agreed=list(models_addressing.keys())
                    ))
                else:
                    # Check if rounds >= threshold
                    rounds = max(self.claim_round_counts[claim].values()) if self.claim_round_counts[claim] else 0
                    if rounds >= self.consensus_threshold_rounds:
                        blocker_items.append(BlockerItem(
                            claim=claim,
                            positions={m: [f.issue for f in findings] for m, findings in models_addressing.items()},
                            rounds_debated=rounds
                        ))

        return consensus_items, blocker_items

    def _generate_report(self, title: str, excerpt: str, consensus: list[ConsensusItem],
                        blockers: list[BlockerItem]) -> dict:
        """Generate final fact-check report."""
        # Filter consensus items by severity > 2 (our threshold)
        action_items = [item for item in consensus if item.severity > 2]

        report = {
            "article": title,
            "excerpt": excerpt,
            "status": "PASS" if not action_items else "NEEDS_REVISION",
            "agreed_upon_changes": [
                {
                    "claim": item.claim,
                    "issue": item.issue,
                    "severity": item.severity,
                    "suggested_fix": item.suggested_fix,
                    "consensus_models": item.models_agreed
                }
                for item in action_items
            ],
            "consensus_blockers": [
                {
                    "claim": item.claim,
                    "conflicting_positions": item.positions,
                    "rounds_debated": item.rounds_debated
                }
                for item in blockers
            ],
            "cost_estimate": sum(self.cost_tracker.values())
        }

        return report

def main():
    if len(sys.argv) < 2:
        print("Usage: python fact_check.py <article_path>")
        sys.exit(1)

    article_path = sys.argv[1]
    orchestrator = FactCheckOrchestrator(budget_per_article=10.0)

    report = orchestrator.run_fact_check(article_path)

    # Save report
    report_path = Path(article_path).parent / f"{Path(article_path).stem}_fact_check.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Report saved to: {report_path}")
    print(f"Status: {report['status']}")
    print(f"Action items (severity > 2): {len(report['agreed_upon_changes'])}")
    print(f"Consensus blockers: {len(report['consensus_blockers'])}")
    print(f"Estimated cost: ${report['cost_estimate']:.2f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
