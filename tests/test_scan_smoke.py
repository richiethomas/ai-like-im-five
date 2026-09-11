"""M3 smoke test: real 4-model scan of article 1, cached to disk fixtures.

Run with: pytest tests/test_scan_smoke.py -m smoke -s

Raw responses are cached in tests/fixtures/cache/ so M4+ iterates on real
data for $0. Delete the cache files to force a fresh scan.
"""

import json

import pytest

from reviewer.article import load_article
from reviewer.costs import CostLedger
from reviewer.findings import anchor_findings, repair_findings
from reviewer.providers import REVIEWER_NAMES, build_provider
from reviewer.scan import run_scans

from conftest import FIXTURE_CACHE

pytestmark = pytest.mark.smoke

ARTICLE = "src/content/posts/paper-1-part-1-what-does-it-mean-for-a-computer-to-see-a-picture/index.mdx"


def cached_scans(article) -> dict[str, dict]:
    """Load cached raw scan responses, scanning live for any missing model."""
    results: dict[str, dict] = {}
    missing = []
    for name in REVIEWER_NAMES:
        path = FIXTURE_CACHE / f"scan_article1_{name}.json"
        if path.exists():
            results[name] = json.loads(path.read_text())
        else:
            missing.append(name)

    if missing:
        ledger = CostLedger()
        providers = [build_provider(n, ledger) for n in missing]
        live = run_scans(article, providers)
        for name, res in live.items():
            if isinstance(res, Exception):
                print(f"  {name}: SCAN FAILED — {res}")
                continue
            (FIXTURE_CACHE / f"scan_article1_{name}.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2))
            results[name] = res
        print(f"  live scan of {missing}: ${ledger.total_usd:.4f}")

    return results


def test_scan_repair_anchor_pipeline():
    article = load_article(ARTICLE)
    raw = cached_scans(article)

    assert len(raw) >= 3, f"only {len(raw)}/4 models produced a scan: {list(raw)}"

    stats = {}
    all_findings = []
    for name, payload in raw.items():
        findings, repair_dropped = repair_findings(payload.get("findings", []), name)
        anchored, anchor_dropped = anchor_findings(findings, article.body)
        stats[name] = {
            "raw": len(payload.get("findings", [])),
            "repaired": len(findings),
            "anchored": len(anchored),
            "repair_dropped": [d.reason for d in repair_dropped],
            "anchor_dropped": [d.raw.get("quote", "")[:60] for d in anchor_dropped],
            "kinds": {},
        }
        for f in anchored:
            stats[name]["kinds"][f.anchor.kind] = stats[name]["kinds"].get(f.anchor.kind, 0) + 1
        all_findings.extend(anchored)

    print("\n--- scan pipeline stats ---")
    for name, s in stats.items():
        print(f"{name}: raw={s['raw']} repaired={s['repaired']} anchored={s['anchored']} "
              f"kinds={s['kinds']}")
        for reason in s["repair_dropped"]:
            print(f"    repair-drop: {reason}")
        for q in s["anchor_dropped"]:
            print(f"    anchor-drop: {q!r}")

    # Gate: enough total signal to debate
    assert len(all_findings) >= 8, f"only {len(all_findings)} findings total"
    # Gate: at least 3 models contribute
    contributing = [n for n, s in stats.items() if s["anchored"] >= 1]
    assert len(contributing) >= 3, f"only {contributing} contributed findings"
    # Gate: quote-anchor survival rate — anchoring shouldn't slaughter findings
    total_repaired = sum(s["repaired"] for s in stats.values())
    total_anchored = sum(s["anchored"] for s in stats.values())
    assert total_anchored / max(total_repaired, 1) >= 0.7, \
        f"anchor survival {total_anchored}/{total_repaired} below 70%"
    # Spans resolved and re-anchored to exact article text
    for f in all_findings:
        if f.anchor.kind == "quote":
            assert f.anchor.start is not None
            assert article.body[f.anchor.start:f.anchor.end] == f.anchor.quote
