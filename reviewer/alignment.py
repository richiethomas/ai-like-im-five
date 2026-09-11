"""Cross-model claim alignment: group findings that target the same underlying claim.

Two-signal design (span overlap alone both over- and under-merges):
  1. Candidate generation — quote spans overlap (section/global findings pair
     within the same anchor kind instead).
  2. Confirmation — issue-text similarity plus dimension as a SOFT signal:
     - clearly similar issues -> merge
     - clearly unrelated issues -> keep separate
     - ambiguous band -> one cheap structured adjudication call ("same
       underlying claim?"), when an adjudicator provider is supplied;
       without one, fall back to same-dimension-class + moderate similarity.
Union-find with a mega-claim guard: a merge that would stretch a claim's span
past MAX_CLAIM_SPAN_CHARS is refused, so transitive chaining can't swallow
half the article.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from .providers import Provider
from .schemas import (
    ADJUDICATION_RESPONSE_SCHEMA,
    Anchor,
    AnchorKind,
    Claim,
    Finding,
    same_dimension_class,
)

MAX_CLAIM_SPAN_CHARS = 500

# Issue-similarity decision bands
SIM_MERGE = 0.55      # >= this: same claim, no adjudication needed
SIM_REJECT = 0.05     # < this (no shared content words): different claims
SIM_HEURISTIC = 0.3   # no-adjudicator fallback: merge iff same dim class and >= this

# Function words carry no claim identity — but negations DO (keep not/no/never).
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "of", "in", "to", "and",
    "or", "for", "with", "on", "at", "by", "as", "from", "could", "would",
    "should", "may", "might", "can", "will", "here", "there", "about",
    "does", "do", "has", "have", "which", "who", "into", "than", "then",
}


def _tokens(s: str) -> list[str]:
    words = re.sub(r"[^a-z0-9]+", " ", s.lower()).split()
    return [w for w in words if w not in _STOPWORDS]


def issue_similarity(a: str, b: str) -> float:
    """Content-word similarity: max of set Jaccard (handles reordering) and
    word-sequence ratio (handles near-duplicates). Character-level ratios are
    deliberately NOT used — they inflate on short strings."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    jac = len(sa & sb) / len(sa | sb)
    seq = difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()
    return max(jac, seq)


def _spans_overlap(a: Anchor, b: Anchor) -> bool:
    if a.start is None or b.start is None:
        return False
    return a.start < b.end and b.start < a.end


def _adjudicate(adjudicator: Provider, f1: Finding, f2: Finding) -> bool:
    prompt = f"""Two reviewers flagged issues on a blog article. Decide whether they are raising the SAME underlying claim (one change would address both) or DIFFERENT claims.

Reviewer A flagged{' text: "' + f1.anchor.quote + '"' if f1.anchor.quote else ' (document-level)'}
A's issue ({f1.dimension}): {f1.issue}

Reviewer B flagged{' text: "' + f2.anchor.quote + '"' if f2.anchor.quote else ' (document-level)'}
B's issue ({f2.dimension}): {f2.issue}

Return JSON: {{"same_claim": true/false, "reason": "<10 words>"}}"""
    data = adjudicator.structured(prompt, ADJUDICATION_RESPONSE_SCHEMA, 200,
                                  label="adjudicate")
    return bool(data.get("same_claim"))


@dataclass
class AlignmentResult:
    claims: list[Claim]
    adjudications: list[dict] = field(default_factory=list)  # audit log


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _group_span(findings: list[Finding]) -> tuple[int, int] | None:
    starts = [f.anchor.start for f in findings if f.anchor.start is not None]
    ends = [f.anchor.end for f in findings if f.anchor.end is not None]
    if not starts:
        return None
    return min(starts), max(ends)


def _same_claim(f1: Finding, f2: Finding, adjudicator: Provider | None,
                log: list[dict]) -> bool:
    sim = issue_similarity(f1.issue, f2.issue)
    same_class = same_dimension_class(f1.dimension, f2.dimension)
    if sim >= SIM_MERGE:
        return True
    if sim < SIM_REJECT:
        return False
    if adjudicator is not None:
        verdict = _adjudicate(adjudicator, f1, f2)
        log.append({"a": f1.issue[:60], "b": f2.issue[:60],
                    "sim": round(sim, 3), "verdict": verdict})
        return verdict
    return same_class and sim >= SIM_HEURISTIC


def align_findings(findings: list[Finding], body: str,
                   adjudicator: Provider | None = None) -> AlignmentResult:
    """Group anchored findings into Claims. `body` is the canonical article text
    (used only to materialize the canonical quote for widened spans)."""
    n = len(findings)
    uf = _UnionFind(n)
    adjudication_log: list[dict] = []

    quote_kinds = {AnchorKind.QUOTE.value, AnchorKind.QUOTE_PAIR.value}

    for i in range(n):
        for j in range(i + 1, n):
            fi, fj = findings[i], findings[j]
            ki, kj = fi.anchor.kind, fj.anchor.kind

            if ki in quote_kinds and kj in quote_kinds:
                if not _spans_overlap(fi.anchor, fj.anchor):
                    continue
            elif ki == kj and ki in (AnchorKind.SECTION.value, AnchorKind.GLOBAL.value):
                pass  # no span signal; similarity decides alone
            else:
                continue  # quote-vs-section etc. never merge

            if not _same_claim(fi, fj, adjudicator, adjudication_log):
                continue

            # Mega-claim guard: refuse merges that stretch the span too far
            gi = [findings[k] for k in range(n) if uf.find(k) == uf.find(i)]
            gj = [findings[k] for k in range(n) if uf.find(k) == uf.find(j)]
            merged_span = _group_span(gi + gj)
            if merged_span and merged_span[1] - merged_span[0] > MAX_CLAIM_SPAN_CHARS:
                continue
            uf.union(i, j)

    # Materialize groups
    groups: dict[int, list[Finding]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(findings[i])

    # Deterministic ordering: by span start, then kind, then first issue text
    def group_key(fs: list[Finding]):
        span = _group_span(fs)
        return (span[0] if span else 1 << 30, fs[0].anchor.kind, fs[0].issue)

    claims: list[Claim] = []
    for idx, fs in enumerate(sorted(groups.values(), key=group_key), start=1):
        # Canonical anchor: the widest-span quote finding; else first anchor
        quote_fs = [f for f in fs if f.anchor.start is not None]
        if quote_fs:
            rep = max(quote_fs, key=lambda f: f.anchor.end - f.anchor.start)
            canonical = Anchor(**rep.anchor.to_dict())
            span = _group_span(quote_fs)
            if span and (span[1] - span[0]) > (canonical.end - canonical.start):
                canonical.start, canonical.end = span
                canonical.quote = body[span[0]:span[1]]
        else:
            canonical = Anchor(**fs[0].anchor.to_dict())

        # Representative dimension: most common; ties -> highest-severity finding's
        counts: dict[str, int] = {}
        for f in fs:
            counts[f.dimension] = counts.get(f.dimension, 0) + 1
        top = max(counts.values())
        candidates = {d for d, c in counts.items() if c == top}
        dimension = max(fs, key=lambda f: (f.dimension in candidates, f.severity)).dimension

        claims.append(Claim(id=f"c{idx:03d}", anchor=canonical,
                            dimension=dimension, findings=fs))

    return AlignmentResult(claims=claims, adjudications=adjudication_log)
