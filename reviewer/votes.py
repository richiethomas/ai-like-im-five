"""Reviewer vote wave: one structured call per reviewer, in parallel.

Each reviewer votes ACCEPT/REJECT on every contested claim (the author's
DEFEND/NEGOTIATE stances) and answers endorsement questions for claims the
author conceded that the reviewer did not itself raise. A reviewer whose call
fails after retries contributes ABSTAIN resolution votes — the engine excludes
those from majorities and doesn't count waves with fewer than 2 valid votes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from .article import Article
from .providers import Provider
from .schemas import (
    VOTE_RESPONSE_SCHEMA,
    Claim,
    Stance,
    Vote,
    VoteChoice,
    VoteKind,
)

VOTE_MAX_TOKENS = 2500
VOTE_RETRIES = 2


def _contested_block(c: Claim, s: Stance) -> str:
    lines = [f"CLAIM {c.id} [{c.dimension}, severity {c.aggregate_severity}]"]
    if c.anchor.quote:
        lines.append(f'  Article text: "{c.anchor.quote}"')
    for f in c.findings:
        lines.append(f"  - {f.model}: {f.issue}")
    lines.append(f"  AUTHOR'S STANCE: {s.stance} — {s.rationale}")
    if s.proposed_fix:
        lines.append(f"  AUTHOR'S PROPOSED FIX: {s.proposed_fix}")
    return "\n".join(lines)


def _conceded_block(c: Claim) -> str:
    lines = [f"CLAIM {c.id} [{c.dimension}, severity {c.aggregate_severity}, "
             f"raised by {', '.join(c.models)}]"]
    if c.anchor.quote:
        lines.append(f'  Article text: "{c.anchor.quote}"')
    lines.append(f"  Issue: {c.findings[0].issue}")
    if c.resolution_fix:
        lines.append(f"  Agreed fix: {c.resolution_fix}")
    return "\n".join(lines)


def vote_prompt(article: Article, reviewer: str,
                contested: list[tuple[Claim, Stance]],
                conceded_to_endorse: list[Claim]) -> str:
    parts = [f"""You are reviewer "{reviewer}" in a roundtable review of this article. The author has responded to the panel's claims; now you vote.

Article title: {article.title}

Full article:
---
{article.body}
---"""]

    if contested:
        blocks = "\n\n".join(_contested_block(c, s) for c, s in contested)
        parts.append(f"""CONTESTED CLAIMS — vote on each:

{blocks}

For each contested claim vote:
- ACCEPT: the author's position (or proposed fix) reasonably settles the concern.
- REJECT: the concern stands; the author's response does not resolve it.
Judge on the merits — accept good defenses, reject hand-waving. Give a reason (max 25 words).""")

    if conceded_to_endorse:
        blocks = "\n\n".join(_conceded_block(c) for c in conceded_to_endorse)
        parts.append(f"""CONCEDED CLAIMS — the author already accepted these (raised by other reviewers). For each, say whether you independently endorse the concern as significant (severity 3+):

{blocks}""")

    schema_hint = '{"votes": [{"claim_id": "c001", "choice": "ACCEPT", "reason": "<max 25 words>"}], "endorsements": [{"claim_id": "c002", "endorse": true}]}'
    parts.append(f"Return JSON: {schema_hint}\n"
                 f"Vote on every contested claim listed"
                 + ("; endorse-or-not every conceded claim listed." if conceded_to_endorse else "."))
    return "\n\n".join(parts)


def _parse_votes(data: dict, reviewer: str, contested_ids: set[str],
                 endorse_ids: set[str]) -> list[Vote]:
    votes: list[Vote] = []
    for raw in data.get("votes", []):
        if not isinstance(raw, dict):
            continue
        cid = str(raw.get("claim_id", "")).strip()
        choice = str(raw.get("choice", "")).strip().upper()
        if cid in contested_ids and choice in (VoteChoice.ACCEPT.value,
                                               VoteChoice.REJECT.value):
            votes.append(Vote(claim_id=cid, model=reviewer,
                              kind=VoteKind.RESOLUTION.value, choice=choice,
                              reason=(str(raw["reason"]).strip()
                                      if raw.get("reason") else None)))
    for raw in data.get("endorsements", []) or []:
        if not isinstance(raw, dict):
            continue
        cid = str(raw.get("claim_id", "")).strip()
        if cid in endorse_ids:
            choice = (VoteChoice.ACCEPT.value if raw.get("endorse")
                      else VoteChoice.REJECT.value)
            votes.append(Vote(claim_id=cid, model=reviewer,
                              kind=VoteKind.ENDORSEMENT.value, choice=choice))
    return votes


def _abstain_all(reviewer: str, contested_ids: set[str]) -> list[Vote]:
    return [Vote(claim_id=cid, model=reviewer, kind=VoteKind.RESOLUTION.value,
                 choice=VoteChoice.ABSTAIN.value,
                 reason="reviewer call failed after retries")
            for cid in sorted(contested_ids)]


def request_votes(article: Article,
                  contested: list[tuple[Claim, Stance]],
                  conceded: list[Claim],
                  reviewers: list[Provider],
                  max_workers: int = 4) -> list[Vote]:
    """Run the vote wave across all reviewers in parallel."""
    contested_ids = {c.id for c, _ in contested}
    if not contested_ids and not conceded:
        return []

    def one(provider: Provider) -> list[Vote]:
        # Endorsements only for conceded claims this reviewer didn't raise
        to_endorse = [c for c in conceded if provider.name not in c.models]
        if not contested and not to_endorse:
            return []
        prompt = vote_prompt(article, provider.name, contested, to_endorse)
        endorse_ids = {c.id for c in to_endorse}
        last_err: Exception | None = None
        for attempt in range(VOTE_RETRIES + 1):
            try:
                data = provider.structured(prompt, VOTE_RESPONSE_SCHEMA,
                                           VOTE_MAX_TOKENS,
                                           label=f"vote{':retry' if attempt else ''}")
                votes = _parse_votes(data, provider.name, contested_ids, endorse_ids)
                missing = contested_ids - {v.claim_id for v in votes
                                           if v.kind == VoteKind.RESOLUTION.value}
                # Unvoted contested claims from a *successful* call: abstain
                votes.extend(_abstain_all(provider.name, missing))
                return votes
            except Exception as e:  # noqa: BLE001 — per-reviewer isolation
                last_err = e
        # Total failure: abstain on everything so the engine can discount the wave
        _ = last_err
        return _abstain_all(provider.name, contested_ids)

    all_votes: list[Vote] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(one, p) for p in reviewers]
        for fut in as_completed(futures):
            all_votes.extend(fut.result())
    return all_votes
