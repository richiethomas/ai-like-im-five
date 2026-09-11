"""M4 fixture smoke: align REAL scan output (cached by M3's smoke) with the
real gpt-4o-mini adjudicator. This is the project's biggest empirical risk —
alignment quality is a property of real model outputs, not of logic.

Run with: pytest tests/test_alignment_smoke.py -m smoke -s
Costs pennies (only ambiguous pairs hit the adjudicator).
"""

import pytest

from reviewer.alignment import MAX_CLAIM_SPAN_CHARS, align_findings
from reviewer.article import load_article
from reviewer.costs import CostLedger
from reviewer.findings import anchor_findings, repair_findings
from reviewer.providers import build_provider

from test_scan_smoke import ARTICLE, cached_scans

pytestmark = pytest.mark.smoke


def test_align_real_scans():
    article = load_article(ARTICLE)
    raw = cached_scans(article)

    all_findings = []
    for name, payload in raw.items():
        findings, _ = repair_findings(payload.get("findings", []), name)
        anchored, _ = anchor_findings(findings, article.body)
        all_findings.extend(anchored)

    ledger = CostLedger()
    adjudicator = build_provider("gpt-4o-mini", ledger)
    res = align_findings(all_findings, article.body, adjudicator=adjudicator)

    print(f"\n{len(all_findings)} findings -> {len(res.claims)} claims, "
          f"{len(res.adjudications)} adjudications (${ledger.total_usd:.4f})")
    print("\n--- claims ---")
    for c in res.claims:
        loc = (f"[{c.anchor.start}:{c.anchor.end}]" if c.anchor.start is not None
               else f"({c.anchor.kind})")
        print(f"{c.id} {loc} {c.dimension} sev={c.aggregate_severity} "
              f"models={c.models}")
        for f in c.findings:
            print(f"    {f.model}: ({f.dimension} {f.severity}) {f.issue[:90]}")
    if res.adjudications:
        print("\n--- adjudications ---")
        for a in res.adjudications:
            print(f"  {a['verdict']} (sim {a['sim']}): {a['a']!r} vs {a['b']!r}")

    # Structural invariants
    assert len(res.claims) <= len(all_findings)
    for c in res.claims:
        if c.anchor.start is not None:
            assert c.anchor.end - c.anchor.start <= MAX_CLAIM_SPAN_CHARS
            assert c.anchor.quote == article.body[c.anchor.start:c.anchor.end]
        assert c.findings, "empty claim"
    # Every finding lands in exactly one claim
    assert sum(len(c.findings) for c in res.claims) == len(all_findings)
    # With 4 models scanning the same short article, at least one claim
    # should be multi-model — if not, alignment is under-merging (issue #1)
    multi = [c for c in res.claims if len(c.models) >= 2]
    print(f"\nmulti-model claims: {[c.id for c in multi]}")
    assert multi, "no multi-model claims formed — alignment under-merging"
