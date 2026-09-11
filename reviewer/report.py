"""Final JSON report: computable outcomes from engine state, not prose parsing."""

from __future__ import annotations

from .costs import CostLedger
from .engine import DebateEngine
from .schemas import (
    CONSENSUS_MODEL_COUNT,
    NITPICK_SEVERITY_CEILING,
    PASS_BLOCK_SUPPORT_COUNT,
    Claim,
    ClaimStatus,
    ReportOutcome,
)


def outcome_for(claims: list[Claim], stopped: bool) -> str:
    """Outcome precedence: INCOMPLETE > BLOCKED > NEEDS_CHANGES > PASS.

    - INCOMPLETE: the run was stopped (budget/round cap) with claims unresolved.
    - BLOCKED: at least one claim debated to deadlock.
    - NEEDS_CHANGES: agreed changes above the nitpick ceiling with 2+ model
      support (raisers plus endorsers) are pending. Single-model agreed items
      are advisory — reported, but they don't gate PASS (user decision
      2026-09-11: a lone model's concern isn't enough to hold an article).
    - PASS: nothing open or blocked; no multi-supported agreed item above
      severity 2.
    """
    statuses = {c.status for c in claims}
    if stopped or ClaimStatus.UNRESOLVED_BUDGET.value in statuses \
            or ClaimStatus.OPEN.value in statuses:
        return ReportOutcome.INCOMPLETE.value
    if ClaimStatus.BLOCKED.value in statuses:
        return ReportOutcome.BLOCKED.value
    if any(c.status == ClaimStatus.AGREED.value
           and c.aggregate_severity > NITPICK_SEVERITY_CEILING
           and c.supporter_count >= PASS_BLOCK_SUPPORT_COUNT
           for c in claims):
        return ReportOutcome.NEEDS_CHANGES.value
    return ReportOutcome.PASS.value


def _claim_entry(c: Claim) -> dict:
    return {
        "id": c.id,
        "status": c.status,
        "dimension": c.dimension,
        "severity": c.aggregate_severity,
        "anchor_kind": c.anchor.kind,
        "quote": c.anchor.quote,
        "section": c.anchor.section,
        "raised_by": c.models,
        "endorsed_by": c.endorsements,
        "supporters": c.supporter_count,
        "consensus": c.supporter_count >= CONSENSUS_MODEL_COUNT,
        "rounds": c.rounds,
        "issues": [{"model": f.model, "dimension": f.dimension,
                    "severity": f.severity, "issue": f.issue} for f in c.findings],
        "resolution_fix": c.resolution_fix,
        "gates_pass": (c.status == ClaimStatus.AGREED.value
                       and c.aggregate_severity > NITPICK_SEVERITY_CEILING
                       and c.supporter_count >= PASS_BLOCK_SUPPORT_COUNT),
    }


def build_report(engine: DebateEngine, ledger: CostLedger,
                 article_title: str, article_path: str,
                 dropped_findings: list | None = None) -> dict:
    claims = list(engine.claims.values())

    def of_status(status: ClaimStatus) -> list[dict]:
        entries = [_claim_entry(c) for c in claims if c.status == status.value]
        return sorted(entries, key=lambda e: -e["severity"])

    dropped_summary: dict[str, dict[str, int]] = {}
    for d in dropped_findings or []:
        by_model = dropped_summary.setdefault(d.model, {})
        by_model[d.reason] = by_model.get(d.reason, 0) + 1

    return {
        "article": article_title,
        "article_path": article_path,
        "outcome": outcome_for(claims, engine.stopped),
        "rounds_used": engine.round,
        "debate_turns": engine.total_turns,
        "agreed_changes": of_status(ClaimStatus.AGREED),
        "blockers": of_status(ClaimStatus.BLOCKED),
        "dismissed": of_status(ClaimStatus.DISMISSED),
        "recorded_nitpicks": of_status(ClaimStatus.RECORDED),
        "unresolved": of_status(ClaimStatus.UNRESOLVED_BUDGET),
        "consensus_claims": sorted(c.id for c in engine.consensus_claims()),
        "dropped_findings": dropped_summary,
        "cost": {
            "total_usd": round(ledger.total_usd, 4),
            "budget_usd": ledger.budget_usd,
            "by_model": {m: {**v, "cost_usd": round(v["cost_usd"], 4)}
                         for m, v in ledger.by_model().items()},
        },
    }
