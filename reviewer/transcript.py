"""Markdown transcript: a pure renderer over the engine's event stream.

The discussion script the user reads — who raised what, how the author
responded, how each reviewer voted, and how every claim resolved.
"""

from __future__ import annotations

from .schemas import Claim, Event


def _claim_heading(c: Claim) -> str:
    loc = ""
    if c.anchor.quote:
        q = c.anchor.quote.replace("\n", " ")
        if len(q) > 120:
            q = q[:117] + "..."
        loc = f'\n> "{q}"'
        if c.anchor.quote_b:
            qb = c.anchor.quote_b.replace("\n", " ")[:120]
            loc += f'\n> …contradicts: "{qb}"'
    elif c.anchor.section:
        loc = f"\n> (section: {c.anchor.section})"
    else:
        loc = "\n> (document-wide)"
    return (f"### {c.id} — {c.dimension}, severity {c.aggregate_severity} "
            f"— raised by {', '.join(c.models)}{loc}")


def _metrics_table(metrics: dict) -> list[str]:
    lines = ["", "## Model Usefulness", "",
             "| model | raised | solo | agreed | dismissed | consensus | sev-bias | anchor-drop | cost | $/agreed |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    rows = sorted(metrics.items(), key=lambda kv: -kv[1].get("agreed_raised", 0))
    for name, v in rows:
        if v.get("claims_raised", 0) == 0 and v.get("votes_cast", 0) == 0 \
                and v.get("endorsements_given", 0) == 0 and name == "claude-sonnet-5":
            continue  # the author raises nothing by design; skip its empty row
        bias = v.get("severity_bias")
        drop = v.get("anchor_drop_rate")
        cpa = v.get("cost_per_agreed_raised")
        lines.append(
            f"| {name} | {v.get('claims_raised', 0)} | {v.get('solo_claims', 0)} "
            f"| {v.get('agreed_raised', 0)} | {v.get('dismissed_raised', 0)} "
            f"| {v.get('consensus_participation', 0)} "
            f"| {bias if bias is not None else '—'} "
            f"| {f'{drop:.0%}' if drop is not None else '—'} "
            f"| ${v.get('cost_usd', 0):.4f} "
            f"| {f'${cpa:.4f}' if cpa is not None else '—'} |")
    lines.append("")
    lines.append("_raised = claims this model surfaced; solo = raised by it alone; "
                 "agreed/dismissed = final status of raised claims; consensus = "
                 "consensus claims it supported; sev-bias = mean severity vs claim "
                 "median on shared claims; anchor-drop = findings lost to failed "
                 "quote anchoring._")
    return lines


def render_transcript(events: list[Event], claims: dict[str, Claim],
                      article_title: str, metrics: dict | None = None) -> str:
    lines: list[str] = [
        f"# Roundtable Review: {article_title}",
        "",
        "Multi-model debate: reviewer findings anchored to article text, "
        "author stances, reviewer votes, and computed resolutions.",
        "",
    ]

    for e in events:
        d = e.data
        t = e.type

        if t == "scan_complete":
            lines.append(f"- **Scan** {d['model']}: {d['raw']} findings, "
                         f"{d['anchored']} anchored")
        elif t == "scan_failed":
            lines.append(f"- **Scan** {d['model']}: FAILED — {d['error']}")
        elif t == "finding_dropped":
            raw = d.get("raw") or {}
            detail = raw.get("quote") or raw.get("issue") or ""
            lines.append(f"  - dropped ({d['model']}, {d['reason']}): "
                         f"{str(detail)[:80]}")
        elif t == "adjudications":
            lines.append(f"- **Alignment**: {len(d['pairs'])} ambiguous pairs "
                         f"adjudicated")
        elif t == "claims_formed":
            lines.append("")
            lines.append(f"## Claims ({len(d['claims'])})")
            if d.get("recorded_nitpicks"):
                lines.append(f"_Recorded as nitpicks (severity ≤ 2, not "
                             f"debated): {', '.join(d['recorded_nitpicks'])}_")
            lines.append("")
            for cid in d["claims"]:
                c = claims.get(cid)
                if not c:
                    continue
                lines.append(_claim_heading(c))
                for f in c.findings:
                    lines.append(f"- **{f.model}** ({f.dimension} {f.severity}): "
                                 f"{f.issue}")
                lines.append("")
        elif t == "round_start":
            lines.append("")
            lines.append(f"## Round {d['round']}")
            lines.append("")
        elif t == "stance":
            fix = f"\n  - Proposed fix: {d['proposed_fix']}" if d.get("proposed_fix") else ""
            flag = " _(defaulted — author returned no stance)_" if d.get("defaulted") else ""
            lines.append(f"**Author on {d['claim_id']}: {d['stance']}**{flag} — "
                         f"{d['rationale']}{fix}")
        elif t == "vote":
            reason = f" — {d['reason']}" if d.get("reason") else ""
            if d["kind"] == "endorsement":
                verb = "endorses" if d["choice"] == "ACCEPT" else "does not endorse"
                lines.append(f"- {d['model']} {verb} {d['claim_id']}")
            else:
                lines.append(f"- {d['model']} votes {d['choice']} on "
                             f"{d['claim_id']}{reason}")
        elif t == "resolution":
            lines.append(f"- ✅ **{d['claim_id']} → {d['status']}** ({d['detail']})")
        elif t == "round_continue":
            lines.append(f"- ↩️ {d['claim_id']} continues "
                         f"({d['accepts']}-{d['rejects']}) — next round")
        elif t == "round_not_counted":
            lines.append(f"- ⚠️ {d['claim_id']}: vote wave invalid "
                         f"({d['valid_votes']} valid votes) — round not counted")
        elif t == "budget_stop":
            lines.append("")
            lines.append(f"**⛔ Budget stop** — spent ${d['spent_usd']} of "
                         f"${d['budget_usd']}; unresolved: "
                         f"{', '.join(d['unresolved_claims']) or 'none'}")
        elif t == "round_cap_stop":
            lines.append("")
            lines.append(f"**⛔ Round-cap stop** — unresolved: "
                         f"{', '.join(d['unresolved_claims']) or 'none'}")

    # Final summary
    lines += ["", "## Outcome", ""]
    by_status: dict[str, list[Claim]] = {}
    for c in claims.values():
        by_status.setdefault(c.status, []).append(c)
    for status in ("AGREED", "DISMISSED", "BLOCKED", "RECORDED", "UNRESOLVED_BUDGET"):
        cs = by_status.get(status)
        if not cs:
            continue
        lines.append(f"**{status}** ({len(cs)}):")
        for c in sorted(cs, key=lambda c: -c.aggregate_severity):
            extra = ""
            if status == "AGREED" and c.resolution_fix:
                extra = f" — fix: {c.resolution_fix}"
            elif status == "BLOCKED":
                extra = f" — deadlocked after {c.rounds} rounds"
            consensus = " ⭐consensus" if c.supporter_count >= 3 else ""
            lines.append(f"- {c.id} (sev {c.aggregate_severity}, "
                         f"{c.dimension}){consensus}{extra}")
        lines.append("")

    if metrics:
        lines += _metrics_table(metrics)

    return "\n".join(lines)
