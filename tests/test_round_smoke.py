"""M6 smoke: one live debate round on REAL claims from cached fixtures.

Author (claude-sonnet-5) stances on every open claim, then a full 4-reviewer
vote wave on the contested ones + endorsements on the conceded ones, applied
through the engine. Run with: pytest tests/test_round_smoke.py -m smoke -s
"""

import pytest

from reviewer.alignment import align_findings
from reviewer.article import load_article
from reviewer.author import request_stances
from reviewer.costs import CostLedger
from reviewer.engine import DebateEngine
from reviewer.findings import anchor_findings, repair_findings
from reviewer.providers import AUTHOR_NAME, REVIEWER_NAMES, build_provider
from reviewer.schemas import ClaimStatus, StanceKind
from reviewer.votes import request_votes

from test_scan_smoke import ARTICLE, cached_scans

pytestmark = pytest.mark.smoke


def build_engine(article, ledger):
    raw = cached_scans(article)
    findings = []
    for name, payload in raw.items():
        fs, _ = repair_findings(payload.get("findings", []), name)
        anchored, _ = anchor_findings(fs, article.body)
        findings.extend(anchored)
    adjudicator = build_provider("gpt-4o-mini", ledger)
    res = align_findings(findings, article.body, adjudicator=adjudicator)
    return DebateEngine(res.claims, REVIEWER_NAMES)


def test_one_live_round():
    article = load_article(ARTICLE)
    ledger = CostLedger()
    engine = build_engine(article, ledger)

    open_before = engine.open_claims()
    print(f"\nopen claims: {len(open_before)} "
          f"(+{len(engine.claims) - len(open_before)} recorded nitpicks)")
    assert open_before, "nothing to debate"

    # --- author stances ---
    author = build_provider(AUTHOR_NAME, ledger)
    engine.start_round()
    stances = request_stances(article, open_before, author)

    got = {s.claim_id for s in stances}
    missing = [c.id for c in open_before if c.id not in got]
    print(f"stances: {len(stances)} (missing after re-ask: {missing})")
    for s in stances:
        print(f"  {s.claim_id}: {s.stance} — {s.rationale[:70]}")
    assert len(missing) == 0, f"author skipped {missing}"
    kinds = {s.stance for s in stances}
    assert kinds <= {k.value for k in StanceKind}

    engine.apply_stances(stances)
    contested = engine.contested_claims()
    conceded = engine.conceded_this_round()
    print(f"contested: {[c.id for c in contested]}; conceded: {[c.id for c in conceded]}")

    # --- vote wave ---
    stance_by_id = {s.claim_id: s for s in stances}
    pairs = [(c, stance_by_id[c.id]) for c in contested]
    reviewers = [build_provider(n, ledger) for n in REVIEWER_NAMES]
    votes = request_votes(article, pairs, conceded, reviewers)

    res_votes = [v for v in votes if v.kind == "resolution"]
    end_votes = [v for v in votes if v.kind == "endorsement"]
    print(f"votes: {len(res_votes)} resolution, {len(end_votes)} endorsement")
    if contested:
        assert res_votes, "no resolution votes returned"
        # every reviewer voted (or abstained) on every contested claim
        assert len(res_votes) == len(contested) * len(reviewers)
        abstains = [v for v in res_votes if v.choice == "ABSTAIN"]
        assert len(abstains) < len(res_votes), "every vote was an abstain"

    engine.apply_votes(votes)

    # --- outcome sanity ---
    statuses = {}
    for c in engine.claims.values():
        statuses.setdefault(c.status, []).append(c.id)
    print(f"after round 1: { {k: len(v) for k, v in statuses.items()} }")
    print(f"consensus claims: {[c.id for c in engine.consensus_claims()]}")
    print(f"cost so far: ${ledger.total_usd:.4f}")

    resolved = sum(len(v) for k, v in statuses.items()
                   if k in (ClaimStatus.AGREED.value, ClaimStatus.DISMISSED.value))
    assert resolved >= 1, "a full live round resolved nothing at all"
    assert ledger.total_usd < 1.0, f"one round cost ${ledger.total_usd:.2f} — budget math off"
