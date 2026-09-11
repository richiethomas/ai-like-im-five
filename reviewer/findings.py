"""Finding validation, repair, and quote anchoring.

Repair layer (M2): schema enforcement varies by provider (DeepSeek gets
json_object with no server-side schema; Gemini's schema loses minimum/maximum),
so EVERY provider's output goes through repair: coerce types, clamp ranges,
drop items that can't be salvaged. Every drop is logged with a reason (this
feeds the per-model metrics).

Anchoring (M3): quote anchors are resolved to spans in the canonical article
body — exact match first, then markdown/typography-normalized match, then
ellipsis-fragment match, then fuzzy (difflib >= 0.9). A resolved anchor is
RE-ANCHORED: its quote is replaced with the exact article text at the span,
so downstream alignment works on ground truth, not on the model's paraphrase.
Quotes that can't be located are hallucinations and the finding is dropped.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .schemas import Anchor, AnchorKind, DIMENSIONS, Finding

_ANCHOR_KINDS = {k.value for k in AnchorKind}

# Common dimension misspellings/variants -> canonical wire value
_DIMENSION_ALIASES = {
    "CLICHES": "CLICHÉS",
    "CLICHE": "CLICHÉS",
    "CLICHÉ": "CLICHÉS",
}


@dataclass
class Dropped:
    model: str
    reason: str
    raw: dict


def _coerce_severity(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        sev = value
    elif isinstance(value, float):
        sev = int(value)
    elif isinstance(value, str):
        m = re.search(r"\d+", value)
        if not m:
            return None
        sev = int(m.group())
    else:
        return None
    return max(1, min(10, sev))


def _coerce_dimension(value) -> str | None:
    if not isinstance(value, str):
        return None
    canon = value.strip().upper()
    canon = _DIMENSION_ALIASES.get(canon, canon)
    return canon if canon in DIMENSIONS else None


def repair_findings(raw_findings, model: str) -> tuple[list[Finding], list[Dropped]]:
    """Coerce raw finding dicts from any provider into Finding objects.

    Returns (findings, dropped). Anchors are built but NOT yet validated
    against the article — that's the M3 anchoring pass.
    """
    findings: list[Finding] = []
    dropped: list[Dropped] = []

    if not isinstance(raw_findings, list):
        return findings, [Dropped(model, "findings-not-a-list", {"value": raw_findings})]

    for raw in raw_findings:
        if not isinstance(raw, dict):
            dropped.append(Dropped(model, "item-not-an-object", {"value": raw}))
            continue

        issue = raw.get("issue")
        if not isinstance(issue, str) or not issue.strip():
            dropped.append(Dropped(model, "missing-issue", raw))
            continue

        severity = _coerce_severity(raw.get("severity"))
        if severity is None:
            dropped.append(Dropped(model, "bad-severity", raw))
            continue

        dimension = _coerce_dimension(raw.get("dimension"))
        if dimension is None:
            dropped.append(Dropped(model, "bad-dimension", raw))
            continue

        kind = raw.get("anchor_type")
        if kind not in _ANCHOR_KINDS:
            # Providers without enum enforcement invent values; infer from fields.
            if isinstance(raw.get("quote"), str) and isinstance(raw.get("quote_b"), str):
                kind = AnchorKind.QUOTE_PAIR.value
            elif isinstance(raw.get("quote"), str) and raw.get("quote", "").strip():
                kind = AnchorKind.QUOTE.value
            elif isinstance(raw.get("section"), str) and raw.get("section", "").strip():
                kind = AnchorKind.SECTION.value
            else:
                kind = AnchorKind.GLOBAL.value

        quote = raw.get("quote") if isinstance(raw.get("quote"), str) else None
        quote_b = raw.get("quote_b") if isinstance(raw.get("quote_b"), str) else None
        section = raw.get("section") if isinstance(raw.get("section"), str) else None

        # Kind/field consistency: a quote anchor without a quote degrades.
        if kind == AnchorKind.QUOTE.value and not (quote and quote.strip()):
            kind = AnchorKind.SECTION.value if section else AnchorKind.GLOBAL.value
        if kind == AnchorKind.QUOTE_PAIR.value and not (quote and quote_b):
            kind = AnchorKind.QUOTE.value if quote else AnchorKind.GLOBAL.value

        fix = raw.get("suggested_fix") if isinstance(raw.get("suggested_fix"), str) else None
        if fix is None and isinstance(raw.get("fix"), str):
            fix = raw["fix"]

        findings.append(Finding(
            model=model,
            dimension=dimension,
            issue=issue.strip(),
            severity=severity,
            anchor=Anchor(kind=kind, quote=quote, quote_b=quote_b, section=section),
            suggested_fix=fix,
        ))

    return findings, dropped


# ---------------------------------------------------------------------------
# Quote anchoring (M3)
# ---------------------------------------------------------------------------

_CHAR_TRANSLATE = {
    "‘": "'", "’": "'",   # curly single quotes
    "“": '"', "”": '"',   # curly double quotes
    "–": "-", "—": "-",   # en/em dash
    " ": " ",                  # nbsp
}
_MARKDOWN_STRIP = {"*", "`"}        # emphasis and code markers
_ELLIPSIS_SPLIT = re.compile(r"\.{3}|…")


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Lowercased, typography-straightened, markdown-stripped, space-collapsed
    text plus a map from normalized index -> original index."""
    out: list[str] = []
    idx_map: list[int] = []
    prev_space = True  # collapse leading whitespace too
    for i, ch in enumerate(text):
        ch = _CHAR_TRANSLATE.get(ch, ch)
        if ch in _MARKDOWN_STRIP:
            continue
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            idx_map.append(i)
            prev_space = True
        else:
            out.append(ch.lower())
            idx_map.append(i)
            prev_space = False
    # trailing space
    if out and out[-1] == " ":
        out.pop()
        idx_map.pop()
    return "".join(out), idx_map


class _BodyIndex:
    """Precomputed normalized view of the article body for repeated lookups."""

    def __init__(self, body: str):
        self.body = body
        self.norm, self.idx_map = _normalize_with_map(body)

    def _to_original_span(self, nstart: int, nend: int) -> tuple[int, int]:
        return self.idx_map[nstart], self.idx_map[nend - 1] + 1

    def find(self, quote: str) -> tuple[int, int] | None:
        """Resolve a quote to a span in the original body, or None."""
        # 1) exact
        pos = self.body.find(quote)
        if pos != -1:
            return pos, pos + len(quote)

        nquote, _ = _normalize_with_map(quote)
        if not nquote:
            return None

        # 2) normalized exact
        npos = self.norm.find(nquote)
        if npos != -1:
            return self._to_original_span(npos, npos + len(nquote))

        # 3) ellipsis fragments: anchor first fragment, extend to last if in order
        frags = [f.strip() for f in _ELLIPSIS_SPLIT.split(quote) if f.strip()]
        if len(frags) > 1:
            first = self.find(frags[0])
            if first is not None:
                last = self.find(frags[-1])
                if last is not None and last[0] >= first[1]:
                    return first[0], last[1]
                return first
            return None

        # 4) fuzzy sliding window over normalized text
        return self._fuzzy(nquote)

    def _fuzzy(self, nquote: str, threshold: float = 0.9) -> tuple[int, int] | None:
        """Locate a near-match via difflib matching blocks over the whole body.

        Handles insertions/deletions (model dropped or added a word while
        quoting) that a fixed-size sliding window clips. Guards: the matched
        characters must cover >= threshold of the quote, and the matched body
        span must not exceed 1.5x the quote length (rejects scattered
        common-word matches).
        """
        n = len(nquote)
        if n < 12 or n > len(self.norm):
            return None  # too short to fuzzy-match safely
        matcher = difflib.SequenceMatcher(None, self.norm, nquote, autojunk=False)
        blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
        if not blocks:
            return None
        matched = sum(b.size for b in blocks)
        if matched / n < threshold:
            return None
        start = blocks[0].a
        end = blocks[-1].a + blocks[-1].size
        if end - start > n * 1.5:
            return None
        return self._to_original_span(start, end)


def anchor_findings(findings: list[Finding], body: str) -> tuple[list[Finding], list[Dropped]]:
    """Resolve quote anchors to body spans; drop unresolvable quotes.

    Re-anchors resolved quotes to the exact article text at the span.
    quote_pair anchors degrade to quote when only the first location resolves.
    section/global anchors pass through untouched.
    """
    index = _BodyIndex(body)
    anchored: list[Finding] = []
    dropped: list[Dropped] = []

    for f in findings:
        a = f.anchor
        if a.kind == AnchorKind.QUOTE.value:
            span = index.find(a.quote or "")
            if span is None:
                dropped.append(Dropped(f.model, "quote-not-found",
                                       {"quote": a.quote, "issue": f.issue}))
                continue
            a.start, a.end = span
            a.quote = body[span[0]:span[1]]
            anchored.append(f)

        elif a.kind == AnchorKind.QUOTE_PAIR.value:
            span_a = index.find(a.quote or "")
            span_b = index.find(a.quote_b or "")
            if span_a is None and span_b is None:
                dropped.append(Dropped(f.model, "quote-pair-not-found",
                                       {"quote": a.quote, "quote_b": a.quote_b,
                                        "issue": f.issue}))
                continue
            if span_a is None or span_b is None:
                # one side resolves: degrade to a plain quote anchor
                span = span_a or span_b
                a.kind = AnchorKind.QUOTE.value
                a.start, a.end = span
                a.quote = body[span[0]:span[1]]
                a.quote_b = None
            else:
                a.start, a.end = span_a
                a.quote = body[span_a[0]:span_a[1]]
                a.start_b, a.end_b = span_b
                a.quote_b = body[span_b[0]:span_b[1]]
            anchored.append(f)

        else:  # section / global: nothing to validate against
            anchored.append(f)

    return anchored, dropped
