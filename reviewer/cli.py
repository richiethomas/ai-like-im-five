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


def reviewer_failure_warnings(events) -> list[str]:
    """Lines to print when a reviewer's scan failed or its votes failed, so a
    review that silently ran with fewer models is visible (e.g. Gemini ran out
    of credits on 2026-10-03 and every review after that used 3 of 4)."""
    lines: list[str] = []
    for e in events:
        if e.type == "scan_failed":
            lines.append(f"WARNING: {e.data.get('model')} scan failed, so it raised "
                         f"no findings: {str(e.data.get('error', ''))[:160]}")
    failed_votes: dict[str, int] = {}
    for e in events:
        if e.type == "vote" and "call failed" in str(e.data.get("reason") or ""):
            m = e.data.get("model", "?")
            failed_votes[m] = failed_votes.get(m, 0) + 1
    for m, n in failed_votes.items():
        lines.append(f"WARNING: {m} failed to vote on {n} claim(s) (counted as abstain)")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-LLM roundtable review of a blog article.")
    parser.add_argument("article", help="path to the .mdx/.md article")
    parser.add_argument("--budget", type=float, default=BUDGET_USD,
                        help=f"USD budget for the run (default {BUDGET_USD})")
    parser.add_argument("--out-dir", default=None,
                        help="artifact directory (default review-runs/<article-slug>)")
    parser.add_argument("--experiments", action="store_true",
                        help="instead of a review, generate try-it-yourself "
                             "experiment ideas (roundtable propose/merge/score)")
    parser.add_argument("--metaphors", action="store_true",
                        help="instead of a review, propose metaphors for the "
                             "article's abstract concepts (roundtable "
                             "propose/merge/score)")
    parser.add_argument("--min-score", type=float, default=None,
                        help="surface metaphors scoring this or higher, out of "
                             "15 (applies to --metaphors and the automatic pass)")
    parser.add_argument("--no-metaphors", action="store_true",
                        help="skip the metaphor-opportunity pass that otherwise "
                             "runs automatically at the end of a review")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip the figure check that otherwise runs "
                             "automatically at the end of a review")
    parser.add_argument("--figures-only", action="store_true",
                        help="instead of a review, run only the figure check "
                             "(image-capable models look at each rendered figure)")
    args = parser.parse_args(argv)

    # Keys come from repo-root .env; shell env still wins if already set.
    load_dotenv(Path(__file__).parent.parent / ".env")

    if args.experiments:
        from .experiments import main as experiments_main
        cli_args = [args.article]
        if args.out_dir:
            cli_args += ["--out-dir", args.out_dir]
        cli_args += ["--budget", str(args.budget)]
        return experiments_main(cli_args)

    if args.metaphors:
        from .metaphors import main as metaphors_main
        cli_args = [args.article]
        if args.out_dir:
            cli_args += ["--out-dir", args.out_dir]
        cli_args += ["--budget", str(args.budget)]
        if args.min_score is not None:
            cli_args += ["--min-score", str(args.min_score)]
        return metaphors_main(cli_args)

    if args.figures_only:
        from .figures import main as figures_main
        cli_args = [args.article, "--budget", str(args.budget)]
        if args.out_dir:
            cli_args += ["--out-dir", args.out_dir]
        return figures_main(cli_args)

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
    gating = sum(1 for c in report["agreed_changes"] if c["gates_pass"])
    print(f"Agreed changes: {len(report['agreed_changes'])} "
          f"({gating} gate PASS: independently raised by 2+ models above "
          f"severity 2; {len(report['agreed_changes']) - gating} advisory)")
    print(f"Blockers:       {len(report['blockers'])}")
    print(f"Dismissed:      {len(report['dismissed'])}")
    print(f"Nitpicks:       {len(report['recorded_nitpicks'])} (recorded, not debated)")
    print(f"Consensus:      {len(report['consensus_claims'])} claims with 3+ model support")
    print(f"Cost:           ${report['cost']['total_usd']:.4f} "
          f"of ${report['cost']['budget_usd']:.2f}")
    for line in reviewer_failure_warnings(engine.events):
        print(line)
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

    # The metaphor-opportunity pass runs automatically after every review,
    # writing into the same artifact directory. It is a cheap generative add-on
    # (~$0.02), so a failure there must never sink a completed review.
    metaphor_summary = None
    if not args.no_metaphors:
        from .metaphors import run_metaphors, DEFAULT_MIN_SCORE
        min_score = (args.min_score if args.min_score is not None
                     else DEFAULT_MIN_SCORE)
        try:
            metaphor_summary = run_metaphors(
                args.article, out_dir=result.out_dir,
                budget_usd=args.budget, min_score=min_score)
        except Exception as exc:  # noqa: BLE001 — never fail a done review
            print(f"\n(metaphor pass skipped: {exc})")

    # The figure check also runs automatically, only when the article has
    # images. Advisory like the metaphor pass: it never changes the outcome.
    figure_summary = None
    if not args.no_figures and "<img" in result.article.body:
        from .figures import run_figure_check
        try:
            figure_summary = run_figure_check(
                args.article, out_dir=result.out_dir, budget_usd=args.budget)
        except Exception as exc:  # noqa: BLE001 — never fail a done review
            print(f"\n(figure check skipped: {exc})")

    print(f"\nArtifacts in {result.out_dir}/:")
    artifacts = "  report.json, transcript.md, metrics.json, checkpoint.json"
    if metaphor_summary is not None:
        artifacts += ", metaphors.md, metaphors.json"
    if figure_summary is not None:
        artifacts += ", figures.md, figures.json, figures/"
    print(artifacts)

    if figure_summary is not None:
        from .figures import print_summary
        print_summary(figure_summary)

    if metaphor_summary is not None:
        surfaced = metaphor_summary["surfaced"]
        print(f"\nMetaphor opportunities (strength >= "
              f"{metaphor_summary['min_score']:g} of 15): {len(surfaced)}")
        for e in surfaced:
            sc = e.get("score", {})
            print(f"  [{sc.get('strength', 0):>4}/15] {e.get('concept', '')}")

    return 0 if report["outcome"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
