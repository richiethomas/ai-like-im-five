"""Per-model usefulness metrics: which reviewers are pulling their weight.

Computed from engine state (claims + event stream) and the cost ledger, so
they can be produced from a live run or from any checkpoint.
"""

from __future__ import annotations

from .costs import CostLedger
from .engine import DebateEngine
from .schemas import ClaimStatus, VoteChoice, VoteKind


def build_metrics(engine: DebateEngine, ledger: CostLedger) -> dict:
    models: dict[str, dict] = {}

    def m(name: str) -> dict:
        return models.setdefault(name, {
            "findings_raised": 0,
            "claims_raised": 0,
            "solo_claims": 0,             # claims no other model raised
            "agreed_raised": 0,           # raised claims that ended AGREED (signal)
            "dismissed_raised": 0,        # raised claims that ended DISMISSED (noise)
            "consensus_participation": 0, # consensus claims this model supported
            "endorsements_given": 0,
            "endorsement_opportunities": 0,
            "votes_cast": 0,
            "abstains": 0,
            "severity_bias": None,        # mean(own severity - claim severity) on shared claims
            "scan_raw": 0,
            "scan_anchored": 0,
            "findings_dropped": {},       # reason -> count
            "cost_usd": 0.0,
            "cost_per_agreed_raised": None,
        })

    # --- from claims ---------------------------------------------------------
    bias_samples: dict[str, list[int]] = {}
    for c in engine.claims.values():
        raisers = c.models
        for f in c.findings:
            m(f.model)["findings_raised"] += 1
        for name in raisers:
            entry = m(name)
            entry["claims_raised"] += 1
            if len(raisers) == 1:
                entry["solo_claims"] += 1
            if c.status == ClaimStatus.AGREED.value:
                entry["agreed_raised"] += 1
            elif c.status == ClaimStatus.DISMISSED.value:
                entry["dismissed_raised"] += 1
        if c.supporter_count >= 3:
            for name in set(raisers) | set(c.endorsements):
                m(name)["consensus_participation"] += 1
        # severity calibration only where models can be compared
        if len(raisers) >= 2:
            agg = c.aggregate_severity
            per_model_max: dict[str, int] = {}
            for f in c.findings:
                per_model_max[f.model] = max(per_model_max.get(f.model, 0), f.severity)
            for name, sev in per_model_max.items():
                bias_samples.setdefault(name, []).append(sev - agg)
        for name in c.endorsements:
            m(name)["endorsements_given"] += 1

    for name, samples in bias_samples.items():
        m(name)["severity_bias"] = round(sum(samples) / len(samples), 2)

    # --- from events ---------------------------------------------------------
    for e in engine.events:
        d = e.data
        if e.type == "scan_complete":
            entry = m(d["model"])
            entry["scan_raw"] = d["raw"]
            entry["scan_anchored"] = d["anchored"]
        elif e.type == "finding_dropped":
            drops = m(d["model"])["findings_dropped"]
            drops[d["reason"]] = drops.get(d["reason"], 0) + 1
        elif e.type == "vote":
            entry = m(d["model"])
            if d["kind"] == VoteKind.RESOLUTION.value:
                if d["choice"] == VoteChoice.ABSTAIN.value:
                    entry["abstains"] += 1
                else:
                    entry["votes_cast"] += 1
            elif d["kind"] == VoteKind.ENDORSEMENT.value:
                entry["endorsement_opportunities"] += 1

    # --- costs ---------------------------------------------------------------
    for name, agg in ledger.by_model().items():
        entry = m(name)
        entry["cost_usd"] = round(agg["cost_usd"], 4)
        if entry["agreed_raised"]:
            entry["cost_per_agreed_raised"] = round(
                agg["cost_usd"] / entry["agreed_raised"], 4)

    # anchor drop rate as a derived convenience
    for entry in models.values():
        raw = entry["scan_raw"]
        entry["anchor_drop_rate"] = (
            round(1 - entry["scan_anchored"] / raw, 3) if raw else None)

    return models
