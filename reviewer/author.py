"""Author stance calls: Claude defends the article against aligned claims.

The author sees the FULL article (never truncated) plus every open claim with
its per-model issues and complete debate history (stances, votes, reasons),
and must return exactly one stance per open claim, keyed by claim_id.
Missing stances are re-asked once; anything still missing defaults to DEFEND
inside the engine.
"""

from __future__ import annotations

from .article import Article
from .providers import Provider, extract_list
from .schemas import (
    STANCE_RESPONSE_SCHEMA,
    Claim,
    Stance,
    StanceKind,
)

STANCE_MAX_TOKENS = 4000
_STANCE_KINDS = {s.value for s in StanceKind}


def _claim_block(c: Claim) -> str:
    lines = [f"CLAIM {c.id} [{c.dimension}, severity {c.aggregate_severity}, "
             f"raised by {', '.join(c.models)}]"]
    a = c.anchor
    if a.kind in ("quote", "quote_pair") and a.quote:
        lines.append(f'  Article text: "{a.quote}"')
        if a.quote_b:
            lines.append(f'  Contradicting text: "{a.quote_b}"')
    elif a.kind == "section":
        lines.append(f"  Location: section {a.section!r} (absent content)")
    else:
        lines.append("  Location: document-wide")
    for f in c.findings:
        fix = f" | suggested fix: {f.suggested_fix}" if f.suggested_fix else ""
        lines.append(f"  - {f.model} ({f.dimension} {f.severity}): {f.issue}{fix}")
    for rec in c.history:
        votes = ", ".join(
            f"{v['model']} {v['choice']}" + (f" ({v['reason']})" if v.get("reason") else "")
            for v in rec.get("votes", []))
        lines.append(f"  Round {rec['round']}: you chose {rec['stance']} "
                     f"— {rec.get('rationale', '')}"
                     + (f" | votes: {votes}" if votes else ""))
    return "\n".join(lines)


def stance_prompt(article: Article, claims: list[Claim]) -> str:
    blocks = "\n\n".join(_claim_block(c) for c in claims)
    return f"""You are the author of this article, defending it in a review roundtable with several independent reviewers. Their concerns have been merged into claims below.

Article title: {article.title}

Full article:
---
{article.body}
---

Open claims — respond to EVERY one of them by its claim_id:

{blocks}

For each claim take exactly one stance:
- CONCEDE: the reviewers are right. Give a concrete fix (the exact replacement text or change).
- DEFEND: your original choice is correct for this audience. Explain why in the rationale.
- NEGOTIATE: partial merit. Propose a middle-ground fix.

Take genuine positions: concede when they're right, hold your ground when they're wrong, don't perform agreement. If a claim's history shows reviewers rejected your defense, engage with their reasons; repeat yourself only if they added nothing new.

STYLE RULE for proposed_fix text: never use em-dashes. Use periods, commas, colons, or parentheses instead. Fix text containing an em-dash will itself be flagged as a defect.

Return JSON: {{"stances": [{{"claim_id": "c001", "stance": "CONCEDE", "rationale": "<max 50 words>", "proposed_fix": "<concrete text, required for CONCEDE and NEGOTIATE>"}}, ...]}}
Exactly one stance per claim_id listed above."""


def _parse_stances(data: dict) -> list[Stance]:
    out = []
    for raw in extract_list(data, "stances"):
        if not isinstance(raw, dict):
            continue
        cid = raw.get("claim_id")
        kind = str(raw.get("stance", "")).strip().upper()
        if not isinstance(cid, str) or kind not in _STANCE_KINDS:
            continue
        out.append(Stance(
            claim_id=cid.strip(),
            stance=kind,
            rationale=str(raw.get("rationale", "")).strip(),
            proposed_fix=(str(raw["proposed_fix"]).strip()
                          if raw.get("proposed_fix") else None),
        ))
    return out


def request_stances(article: Article, claims: list[Claim],
                    provider: Provider) -> list[Stance]:
    """One structured stance call; missing claim_ids re-asked once."""
    data = provider.structured(stance_prompt(article, claims),
                               STANCE_RESPONSE_SCHEMA, STANCE_MAX_TOKENS,
                               label="stances")
    stances = _parse_stances(data)
    have = {s.claim_id for s in stances}
    missing = [c for c in claims if c.id not in have]
    if missing:
        data2 = provider.structured(stance_prompt(article, missing),
                                    STANCE_RESPONSE_SCHEMA, STANCE_MAX_TOKENS,
                                    label="stances:reask")
        stances.extend(s for s in _parse_stances(data2)
                       if s.claim_id not in have)
    return stances
