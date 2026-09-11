"""Finding validation and repair.

M2: the repair layer — schema enforcement varies by provider (DeepSeek gets
json_object with no server-side schema; Gemini's schema loses minimum/maximum),
so EVERY provider's output goes through repair: coerce types, clamp ranges,
drop items that can't be salvaged. Every drop is logged with a reason (this
feeds the per-model metrics).

M3 adds quote anchoring/validation on top.
"""

from __future__ import annotations

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
