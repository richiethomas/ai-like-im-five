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

# Clichés to flag in articles and avoid in responses
CLICHES = [
    "honest", "honestly", "genuine", "genuinely",
    "load-bearing", "rides on", "riding on", "seams",
    "it's a.*worth", "shines for", "shines when",
    "belt and suspenders", "keep the data honest",
    "here's what I'd watch", "the point is",
    "really", "really important", "real risk", "real cost",
    "land", "changes that land", "decisions that land",
    "great question", "excellent point", "sharp observation",
    "good catch", "nice catch", "that's a great instinct",
]

@dataclass
class FactCheckFinding:
    """A single finding from a fact-check."""
    claim: str
    issue: str
    severity: int  # 1-10
    dimension: str = "CORRECTNESS"  # which review dimension
    suggested_fix: Optional[str] = None
    model_id: Optional[str] = None
    round_number: int = 0

@dataclass
class ConsensusItem:
    """An item where all models agree."""
    claim: str
    issue: str
    severity: int
    dimension: str = "CORRECTNESS"
    suggested_fix: Optional[str] = None
    models_agreed: list[str] = None

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

        # Build model list based on available API keys
        self.models = []
        if os.getenv("ANTHROPIC_API_KEY"):
            self.models.append("claude-sonnet-5")  # Latest Claude model
        if os.getenv("OPENAI_API_KEY"):
            self.models.append("gpt-4o-mini")
        if os.getenv("DEEPSEEK_API_KEY"):
            self.models.append("deepseek-chat")
        if os.getenv("GEMINI_API_KEY"):
            self.models.append("gemini-3.5-flash")  # 2.0-flash deprecated
        if os.getenv("GROQ_API_KEY"):
            self.models.append("llama-3.1-70b")  # GROQ model: current Llama 3.1

        if not self.models:
            raise ValueError("No API keys configured. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY at minimum.")

        print(f"Loaded {len(self.models)} models: {', '.join(self.models)}\n")

        self.max_rounds = 100
        self.consensus_threshold_rounds = 5
        self.transcript = []  # Full record of all model interactions (initial scans + debate)
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

            # If all issues have consensus, we're done early
            if len(blocker_items) == 0:
                print(f"\n✓ Consensus reached in round {round_num - 1} — terminating early to save tokens")
                break

            print(f"\nRound {round_num}: Debate round...")
            print(f"  Active disagreements: {len(blocker_items)}")

            # Run debate round
            debate_occurred = self._debate_round(blocker_items, initial_findings, round_num=round_num)
            if not debate_occurred:
                break

            round_num += 1

        # Generate final report
        consensus_items, blocker_items = self._analyze_consensus(initial_findings)
        report = self._generate_report(title, excerpt, consensus_items, blocker_items)

        return report

    def _initial_fact_check(self, model: str, title: str, content: str) -> list[FactCheckFinding]:
        """Run initial fact-check from a single model."""
        prompt = f"""You are a rigorous technical fact-checker reviewing a blog article about deep learning and computer vision, written for non-technical business stakeholders.

Article: {title}

Content:
{content[:3000]}...

MANDATORY: Check ALL six dimensions before responding. You must scan for issues in each category, not just correctness.

Review dimensions:
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

Instructions for your response:
- Check each dimension systematically.
- Be direct and specific. No filler.
- Avoid clichés in your own response.
- Do not include minor issues (severity must be >= 3 to be worth reporting).
- If you find no significant issues across all dimensions, respond with: "NO ISSUES FOUND"."""

        findings = []
        try:
            if model == "claude-sonnet-5":
                if not anthropic_client:
                    raise Exception("Claude client not initialized")
                response = anthropic_client.messages.create(
                    model=model,
                    max_tokens=16000,  # Allow extended thinking
                    messages=[{"role": "user", "content": prompt}]
                )
                # Extract text from response (handle ThinkingBlock + TextBlock)
                text = ""
                for block in response.content:
                    if hasattr(block, 'text'):
                        text += block.text
                if not text:
                    raise Exception("No text content in Claude response")
                self.cost_tracker[model] += 0.5
            elif model == "gpt-4o-mini":
                if not openai_client:
                    raise Exception("OpenAI client not initialized")
                response = openai_client.chat.completions.create(
                    model=model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.choices[0].message.content
                self.cost_tracker[model] += 0.3
            elif model == "deepseek-chat":
                from openai import OpenAI
                ds_client = OpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com/v1"
                )
                response = ds_client.chat.completions.create(
                    model="deepseek-chat",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.choices[0].message.content
                self.cost_tracker[model] += 0.2
            elif model == "gemini-3.5-flash":
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                    gemini_model = genai.GenerativeModel("gemini-3.5-flash")
                    response = gemini_model.generate_content(prompt)
                    text = response.text
                    self.cost_tracker[model] += 0.15
                except ImportError:
                    raise Exception("google.generativeai not installed. Run: pip install google-generativeai")
            elif model == "llama-3.1-70b":
                groq_key = os.getenv("GROQ_API_KEY")
                if groq_key:
                    try:
                        from groq import Groq
                        groq_client = Groq(api_key=groq_key)
                        response = groq_client.chat.completions.create(
                            model="llama-3.1-70b-versatile",
                            max_tokens=1000,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        text = response.choices[0].message.content
                        self.cost_tracker[model] += 0.1
                    except ImportError:
                        raise Exception("groq not installed. Run: pip install groq")
                else:
                    raise Exception("GROQ_API_KEY not set")
            else:
                raise Exception(f"Unknown model: {model}")

            # Parse findings from response
            findings = self._parse_findings(text, model)

            # Record in transcript
            self.transcript.append({
                "round": 1,
                "model": model,
                "type": "initial_scan",
                "prompt": prompt,
                "response": text,
                "findings_count": len(findings)
            })
        except Exception as e:
            print(f"  Error from {model}: {e}")
            self.transcript.append({
                "round": 1,
                "model": model,
                "type": "initial_scan",
                "error": str(e)
            })

        return findings

    def _parse_findings(self, text: str, model: str) -> list[FactCheckFinding]:
        """Parse fact-check findings from model response."""
        findings = []

        # Check for "NO ISSUES FOUND" response
        if "NO ISSUES FOUND" in text.upper():
            return findings

        # Split by "CLAIM:" to get individual findings
        parts = text.split("CLAIM:")
        for part in parts[1:]:  # Skip first part before first CLAIM
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
                    findings.append(FactCheckFinding(
                        claim=claim,
                        issue=issue,
                        severity=severity,
                        dimension=dimension,
                        suggested_fix=fix or None,
                        model_id=model,
                        round_number=1
                    ))
                    self.findings_by_claim[claim][model].append(
                        FactCheckFinding(claim=claim, issue=issue, severity=severity,
                                       dimension=dimension, suggested_fix=fix, model_id=model, round_number=1)
                    )
                    self.claim_round_counts[claim][model] += 1
            except:
                continue

        return findings

    def _debate_round(self, blocker_items: list[BlockerItem], all_findings: dict, round_num: int = 2) -> bool:
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

                    # Record in transcript
                    self.transcript.append({
                        "round": round_num,
                        "model": model,
                        "type": "debate_response",
                        "claim_being_debated": blocker.claim,
                        "prompt": prompt,
                        "response": response_text
                    })
                except Exception as e:
                    print(f"  Debate error from {model}: {e}")
                    self.transcript.append({
                        "round": round_num,
                        "model": model,
                        "type": "debate_response",
                        "error": str(e)
                    })

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
                    first_finding = models_addressing[list(models_addressing.keys())[0]][0]
                    consensus_items.append(ConsensusItem(
                        claim=claim,
                        issue=first_finding.issue,
                        severity=int(avg_severity),
                        dimension=first_finding.dimension,
                        suggested_fix=first_finding.suggested_fix,
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

        # Aggregate findings by dimension for audit trail
        findings_by_dimension = defaultdict(list)
        for claim, models_dict in self.findings_by_claim.items():
            for model, findings in models_dict.items():
                for finding in findings:
                    findings_by_dimension[finding.dimension].append({
                        "claim": finding.claim,
                        "issue": finding.issue,
                        "severity": finding.severity,
                        "model": model
                    })

        report = {
            "article": title,
            "excerpt": excerpt,
            "status": "PASS" if not action_items else "NEEDS_REVISION",
            "agreed_upon_changes": [
                {
                    "claim": item.claim,
                    "issue": item.issue,
                    "dimension": item.dimension,
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
            "audit_trail": {
                "findings_by_dimension": dict(findings_by_dimension),
                "dimensions_checked": sorted(set(findings_by_dimension.keys())),
                "total_findings_reviewed": sum(len(findings) for findings in findings_by_dimension.values())
            },
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

    # Save full transcript (all model interactions)
    transcript_path = Path(article_path).parent / f"{Path(article_path).stem}_fact_check_transcript.json"
    with open(transcript_path, 'w') as f:
        json.dump({
            "article": report["article"],
            "transcript": orchestrator.transcript,
            "total_interactions": len(orchestrator.transcript),
            "cost_estimate": report["cost_estimate"]
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Report saved to: {report_path}")
    print(f"Transcript saved to: {transcript_path}")
    print(f"Status: {report['status']}")
    print(f"Action items (severity > 2): {len(report['agreed_upon_changes'])}")
    print(f"Consensus blockers: {len(report['consensus_blockers'])}")
    print(f"Estimated cost: ${report['cost_estimate']:.2f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
