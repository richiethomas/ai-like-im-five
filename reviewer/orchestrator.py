"""Run orchestration: scan -> validate -> align -> debate loop -> artifacts.

Checkpointing: full state JSON (claims, events, cost ledger, raw scans) is
written after the scan and after every debate round, and again in a `finally`
— a crash at round 4 of a paid run never loses the spend. Budget is checked
at round boundaries only, with headroom for one more round, so a wave is
never killed half-recorded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .alignment import align_findings
from .article import Article, load_article
from .author import request_stances
from .costs import CostLedger
from .engine import DebateEngine
from .findings import Dropped, anchor_findings, repair_findings
from .providers import AUTHOR_NAME, REVIEWER_NAMES, Provider, build_provider
from .scan import run_scans
from .schemas import BUDGET_USD, BudgetExceededError, Event
from .votes import request_votes

# Conservative estimate of one more debate round (author stance call + 4
# reviewer votes). Observed round 1 on article 1: ~$0.06; 0.50 leaves margin
# for longer histories late in a debate.
EST_ROUND_COST_USD = 0.50

# Loop guard: rounds where no wave is valid don't consume turns, so a pack of
# dead reviewers could spin forever without this ceiling on round count.
MAX_LOOP_ROUNDS = 30


@dataclass
class RunResult:
    article: Article
    engine: DebateEngine
    ledger: CostLedger
    out_dir: Path
    dropped: list[Dropped]


def checkpoint(out_dir: Path, tag: str, article: Article, engine: DebateEngine,
               ledger: CostLedger, raw_scans: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "tag": tag,
        "article_path": article.path,
        "article_title": article.title,
        "engine": engine.to_dict(),
        "ledger": ledger.to_dict(),
        "raw_scans": {k: v for k, v in raw_scans.items()
                      if not isinstance(v, Exception)},
    }
    tmp = out_dir / "checkpoint.json.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    tmp.replace(out_dir / "checkpoint.json")


def debate_loop(article: Article, engine: DebateEngine, author: Provider,
                reviewers: list[Provider], ledger: CostLedger,
                checkpoint_fn=None) -> None:
    """Drive rounds until the engine finishes or the budget/loop guard stops it."""
    while not engine.finished():
        if engine.round >= MAX_LOOP_ROUNDS:
            engine.force_stop(f"loop guard: {engine.round} rounds")
            break
        try:
            ledger.check(headroom_usd=EST_ROUND_COST_USD)
        except BudgetExceededError:
            engine.stop_for_budget(ledger.total_usd, ledger.budget_usd)
            break

        engine.start_round()
        open_claims = engine.open_claims()
        stances = request_stances(article, open_claims, author)
        engine.apply_stances(stances)

        contested = engine.contested_claims()
        conceded = engine.conceded_this_round()
        if contested or conceded:
            # Source of truth is the engine's pending stances (includes
            # engine-defaulted DEFENDs), never the author's returned list.
            pending = engine.pending_stances()
            pairs = [(c, pending[c.id]) for c in contested]
            votes = request_votes(article, pairs, conceded, reviewers)
            engine.apply_votes(votes)

        if checkpoint_fn:
            checkpoint_fn(f"round{engine.round}")


def run_review(article_path: str | Path,
               out_dir: str | Path | None = None,
               budget_usd: float = BUDGET_USD,
               reviewer_names: list[str] | None = None) -> RunResult:
    article = load_article(article_path)
    if out_dir is None:
        out_dir = Path("review-runs") / Path(article_path).parent.name
    out_dir = Path(out_dir)

    ledger = CostLedger(budget_usd=budget_usd)
    names = reviewer_names or REVIEWER_NAMES
    reviewers = [build_provider(n, ledger) for n in names]
    author = build_provider(AUTHOR_NAME, ledger)
    adjudicator = next((p for p in reviewers if p.name == "gpt-4o-mini"),
                       reviewers[0])

    # --- scan (round 0) + validate ---
    raw_scans = run_scans(article, reviewers)
    pre_events: list[Event] = []
    findings = []
    dropped: list[Dropped] = []
    for name, payload in raw_scans.items():
        if isinstance(payload, Exception):
            pre_events.append(Event(seq=len(pre_events), type="scan_failed",
                                    data={"model": name, "error": str(payload)}))
            continue
        fs, rep_drops = repair_findings(payload.get("findings", []), name)
        anchored, anc_drops = anchor_findings(fs, article.body)
        findings.extend(anchored)
        model_drops = rep_drops + anc_drops
        dropped.extend(model_drops)
        pre_events.append(Event(seq=len(pre_events), type="scan_complete",
                                data={"model": name, "raw": len(payload.get("findings", [])),
                                      "anchored": len(anchored)}))
        for d in model_drops:
            pre_events.append(Event(seq=len(pre_events), type="finding_dropped",
                                    data={"model": d.model, "reason": d.reason,
                                          "raw": d.raw}))

    # --- align ---
    aligned = align_findings(findings, article.body, adjudicator=adjudicator)
    if aligned.adjudications:
        pre_events.append(Event(seq=len(pre_events), type="adjudications",
                                data={"pairs": aligned.adjudications}))

    engine = DebateEngine(aligned.claims, names, initial_events=pre_events)
    ckpt = lambda tag: checkpoint(out_dir, tag, article, engine, ledger, raw_scans)  # noqa: E731
    ckpt("scan")

    result = RunResult(article=article, engine=engine, ledger=ledger,
                       out_dir=out_dir, dropped=dropped)

    # --- debate ---
    try:
        debate_loop(article, engine, author, reviewers, ledger,
                    checkpoint_fn=ckpt)
    finally:
        # Even on a crash: final checkpoint + (partial) report and transcript.
        ckpt("final")
        try:
            write_artifacts(result)
        except Exception:  # noqa: BLE001 — never mask the original error
            pass

    return result


def write_artifacts(result: RunResult) -> tuple[Path, Path]:
    """Render report.json + transcript.md + metrics.json into the run dir."""
    from .metrics import build_metrics
    from .report import build_report
    from .transcript import render_transcript

    result.out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(result.engine, result.ledger, result.article.title,
                          result.article.path, result.dropped)
    report_path = result.out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    metrics = build_metrics(result.engine, result.ledger)
    (result.out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2))

    transcript = render_transcript(result.engine.events, result.engine.claims,
                                   result.article.title, metrics=metrics)
    transcript_path = result.out_dir / "transcript.md"
    transcript_path.write_text(transcript)
    return report_path, transcript_path
