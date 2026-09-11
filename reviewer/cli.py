"""CLI: run a full multi-model review of one article.

Usage:
    .venv-fact-check/bin/python -m reviewer.cli <article.mdx> [--budget 10] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from .orchestrator import run_review
from .schemas import BUDGET_USD


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-LLM roundtable review of a blog article.")
    parser.add_argument("article", help="path to the .mdx/.md article")
    parser.add_argument("--budget", type=float, default=BUDGET_USD,
                        help=f"USD budget for the run (default {BUDGET_USD})")
    parser.add_argument("--out-dir", default=None,
                        help="artifact directory (default review-runs/<article-slug>)")
    args = parser.parse_args(argv)

    # Keys come from repo-root .env; shell env still wins if already set.
    load_dotenv(Path(__file__).parent.parent / ".env")

    result = run_review(args.article, out_dir=args.out_dir,
                        budget_usd=args.budget)
    engine, ledger = result.engine, result.ledger

    from .report import build_report
    report = build_report(engine, ledger, result.article.title,
                          result.article.path, result.dropped)

    print(f"\n{'=' * 60}")
    print(f"Review complete: {result.article.title}")
    print(f"{'=' * 60}")
    print(f"Outcome:        {report['outcome']}")
    print(f"Rounds:         {report['rounds_used']} "
          f"({report['debate_turns']} debate turns)")
    print(f"Agreed changes: {len(report['agreed_changes'])} "
          f"({sum(1 for c in report['agreed_changes'] if c['severity'] > 2)} above severity 2)")
    print(f"Blockers:       {len(report['blockers'])}")
    print(f"Dismissed:      {len(report['dismissed'])}")
    print(f"Nitpicks:       {len(report['recorded_nitpicks'])} (recorded, not debated)")
    print(f"Consensus:      {len(report['consensus_claims'])} claims with 3+ model support")
    print(f"Cost:           ${report['cost']['total_usd']:.4f} "
          f"of ${report['cost']['budget_usd']:.2f}")
    from .metrics import build_metrics
    metrics = build_metrics(engine, ledger)
    reviewer_rows = {k: v for k, v in metrics.items() if k != "claude-sonnet-5"}
    if reviewer_rows:
        print("\nModel usefulness:")
        print(f"  {'model':20s} {'raised':>6s} {'solo':>4s} {'agreed':>6s} "
              f"{'dism':>4s} {'sev-bias':>8s} {'cost':>8s} {'$/agreed':>9s}")
        for name, v in sorted(reviewer_rows.items(),
                              key=lambda kv: -kv[1]["agreed_raised"]):
            bias = v["severity_bias"] if v["severity_bias"] is not None else "—"
            cpa = (f"${v['cost_per_agreed_raised']:.4f}"
                   if v["cost_per_agreed_raised"] is not None else "—")
            print(f"  {name:20s} {v['claims_raised']:>6d} {v['solo_claims']:>4d} "
                  f"{v['agreed_raised']:>6d} {v['dismissed_raised']:>4d} "
                  f"{str(bias):>8s} ${v['cost_usd']:>7.4f} {cpa:>9s}")

    print(f"\nArtifacts in {result.out_dir}/:")
    print("  report.json, transcript.md, metrics.json, checkpoint.json")

    return 0 if report["outcome"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
