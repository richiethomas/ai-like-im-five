"""M9 unit tests: model usefulness metrics — synthetic + real checkpoint. Offline."""

import json

import pytest

from reviewer.costs import CostLedger
from reviewer.engine import DebateEngine
from reviewer.metrics import build_metrics
from reviewer.schemas import Anchor, Claim, ClaimStatus, Event, Finding

from conftest import FIXTURE_CACHE
from test_report_transcript import CHECKPOINT, rebuild_engine


def finding(model, severity=5):
    return Finding(model=model, dimension="CORRECTNESS", issue=f"i-{model}",
                   severity=severity,
                   anchor=Anchor(kind="quote", quote="q", start=0, end=1))


def claim(cid, status, model_sev: dict, endorsements=()):
    c = Claim(id=cid, anchor=Anchor(kind="quote", quote="q", start=0, end=1),
              dimension="CORRECTNESS",
              findings=[finding(m, s) for m, s in model_sev.items()],
              status=status.value)
    c.endorsements = list(endorsements)
    return c


def make_engine(claims, events=()):
    eng = DebateEngine(claims, ["a", "b", "c", "d"], initial_events=[])
    eng.events = list(events)
    return eng


def test_raised_solo_agreed_dismissed_counts():
    claims = [
        claim("c1", ClaimStatus.AGREED, {"a": 5, "b": 6}),
        claim("c2", ClaimStatus.DISMISSED, {"a": 4}),
        claim("c3", ClaimStatus.AGREED, {"b": 7}),
    ]
    m = build_metrics(make_engine(claims), CostLedger())
    assert m["a"]["claims_raised"] == 2
    assert m["a"]["solo_claims"] == 1
    assert m["a"]["agreed_raised"] == 1
    assert m["a"]["dismissed_raised"] == 1
    assert m["b"]["agreed_raised"] == 2
    assert m["b"]["solo_claims"] == 1


def test_consensus_participation_includes_endorsers():
    claims = [claim("c1", ClaimStatus.AGREED, {"a": 5, "b": 5},
                    endorsements=["c"])]  # 3 supporters -> consensus
    m = build_metrics(make_engine(claims), CostLedger())
    for name in ("a", "b", "c"):
        assert m[name]["consensus_participation"] == 1
    assert m["c"]["endorsements_given"] == 1
    assert m["c"]["claims_raised"] == 0


def test_severity_bias_on_shared_claims_only():
    claims = [
        claim("c1", ClaimStatus.AGREED, {"a": 9, "b": 5, "c": 5}),  # agg median 5
        claim("c2", ClaimStatus.AGREED, {"a": 8}),                  # solo: excluded
    ]
    m = build_metrics(make_engine(claims), CostLedger())
    assert m["a"]["severity_bias"] == pytest.approx(4.0)   # 9 - 5
    assert m["b"]["severity_bias"] == pytest.approx(0.0)


def test_event_derived_stats():
    events = [
        Event(0, "scan_complete", {"model": "a", "raw": 10, "anchored": 8}),
        Event(1, "finding_dropped", {"model": "a", "reason": "quote-not-found",
                                     "raw": {}}),
        Event(2, "finding_dropped", {"model": "a", "reason": "quote-not-found",
                                     "raw": {}}),
        Event(3, "vote", {"model": "b", "kind": "resolution", "choice": "ACCEPT",
                          "claim_id": "c1", "round": 1, "reason": None}),
        Event(4, "vote", {"model": "b", "kind": "resolution", "choice": "ABSTAIN",
                          "claim_id": "c2", "round": 1, "reason": None}),
        Event(5, "vote", {"model": "b", "kind": "endorsement", "choice": "REJECT",
                          "claim_id": "c3", "round": 1, "reason": None}),
    ]
    m = build_metrics(make_engine([], events), CostLedger())
    assert m["a"]["scan_raw"] == 10 and m["a"]["scan_anchored"] == 8
    assert m["a"]["anchor_drop_rate"] == pytest.approx(0.2)
    assert m["a"]["findings_dropped"] == {"quote-not-found": 2}
    assert m["b"]["votes_cast"] == 1
    assert m["b"]["abstains"] == 1
    assert m["b"]["endorsement_opportunities"] == 1
    assert m["b"]["endorsements_given"] == 0   # rejected the endorsement


def test_cost_per_agreed_raised():
    claims = [claim("c1", ClaimStatus.AGREED, {"gpt-4o-mini": 5})]
    ledger = CostLedger()
    ledger.record("gpt-4o-mini", 1_000_000, 0)  # $0.15
    m = build_metrics(make_engine(claims), ledger)
    assert m["gpt-4o-mini"]["cost_usd"] == pytest.approx(0.15)
    assert m["gpt-4o-mini"]["cost_per_agreed_raised"] == pytest.approx(0.15)


# --- against the real run ----------------------------------------------------

def test_metrics_from_real_run():
    if not CHECKPOINT.exists():
        pytest.skip("run M7 smoke first")
    state = json.loads(CHECKPOINT.read_text())
    eng = rebuild_engine(state)
    ledger = CostLedger.from_dict(state["ledger"])
    m = build_metrics(eng, ledger)

    reviewers = [k for k in m if k != "claude-sonnet-5"]
    assert len(reviewers) == 4
    total_raised = sum(m[r]["claims_raised"] for r in reviewers)
    assert total_raised > 0
    for r in reviewers:
        assert m[r]["cost_usd"] > 0
    json.dumps(m)  # serializable
    print("\nper-model (real run):")
    for r, v in m.items():
        print(f"  {r}: raised={v['claims_raised']} solo={v['solo_claims']} "
              f"agreed={v['agreed_raised']} cost=${v['cost_usd']}")
