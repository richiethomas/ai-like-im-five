"""M8 unit tests: report outcome logic + renderers, incl. against the REAL
checkpoint fixture saved by M7's smoke run. Offline."""

import json

import pytest

from reviewer.costs import CostLedger
from reviewer.engine import DebateEngine
from reviewer.report import build_report, outcome_for
from reviewer.schemas import Anchor, Claim, ClaimStatus, Event, Finding
from reviewer.transcript import render_transcript

from conftest import FIXTURE_CACHE

CHECKPOINT = FIXTURE_CACHE / "run_article1_checkpoint.json"


def claim(cid, status, severity=5):
    f = Finding(model="m1", dimension="CORRECTNESS", issue="i", severity=severity,
                anchor=Anchor(kind="quote", quote="q", start=0, end=1))
    return Claim(id=cid, anchor=Anchor(kind="quote", quote="q", start=0, end=1),
                 dimension="CORRECTNESS", findings=[f], status=status.value)


# --- outcome logic -----------------------------------------------------------

def test_outcome_pass():
    claims = [claim("c1", ClaimStatus.DISMISSED),
              claim("c2", ClaimStatus.RECORDED, severity=2),
              claim("c3", ClaimStatus.AGREED, severity=2)]  # agreed nitpick ok
    assert outcome_for(claims, stopped=False) == "PASS"


def test_outcome_needs_changes():
    claims = [claim("c1", ClaimStatus.AGREED, severity=5)]
    assert outcome_for(claims, stopped=False) == "NEEDS_CHANGES"


def test_outcome_blocked_beats_needs_changes():
    claims = [claim("c1", ClaimStatus.AGREED, severity=5),
              claim("c2", ClaimStatus.BLOCKED, severity=4)]
    assert outcome_for(claims, stopped=False) == "BLOCKED"


def test_outcome_incomplete_beats_all():
    claims = [claim("c1", ClaimStatus.BLOCKED, severity=5),
              claim("c2", ClaimStatus.UNRESOLVED_BUDGET, severity=4)]
    assert outcome_for(claims, stopped=True) == "INCOMPLETE"


def test_outcome_open_claims_mean_incomplete():
    claims = [claim("c1", ClaimStatus.OPEN)]
    assert outcome_for(claims, stopped=False) == "INCOMPLETE"


# --- against the real M7 run -------------------------------------------------

@pytest.fixture
def real_state():
    if not CHECKPOINT.exists():
        pytest.skip("run M7 smoke first: pytest tests/test_run_smoke.py -m smoke")
    return json.loads(CHECKPOINT.read_text())


def rebuild_engine(state) -> DebateEngine:
    claims = [Claim.from_dict(d) for d in state["engine"]["claims"]]
    events = [Event.from_dict(d) for d in state["engine"]["events"]]
    eng = DebateEngine(claims, state["engine"]["reviewer_names"],
                       initial_events=[])
    # from_dict claims arrive with terminal statuses; engine init only filters
    # OPEN nitpicks, so statuses survive. Restore the rest of the run state.
    eng.events = events
    eng.round = state["engine"]["round"]
    eng.total_turns = state["engine"]["total_turns"]
    eng.stopped = state["engine"]["stopped"]
    return eng


def test_report_from_real_run(real_state):
    eng = rebuild_engine(real_state)
    ledger = CostLedger.from_dict(real_state["ledger"])
    report = build_report(eng, ledger, real_state["article_title"],
                          real_state["article_path"])

    assert report["outcome"] in ("PASS", "NEEDS_CHANGES", "BLOCKED", "INCOMPLETE")
    n_claims = len(real_state["engine"]["claims"])
    buckets = (len(report["agreed_changes"]) + len(report["blockers"])
               + len(report["dismissed"]) + len(report["recorded_nitpicks"])
               + len(report["unresolved"]))
    assert buckets == n_claims          # every claim lands in exactly one bucket
    assert report["cost"]["total_usd"] > 0
    assert report["cost"]["by_model"]   # per-model spend present
    for entry in report["agreed_changes"]:
        assert entry["severity"] >= 1
        assert entry["raised_by"]
    # agreed entries sorted most-severe first
    sevs = [e["severity"] for e in report["agreed_changes"]]
    assert sevs == sorted(sevs, reverse=True)
    # json-serializable end to end
    json.dumps(report)


def test_transcript_from_real_run(real_state):
    eng = rebuild_engine(real_state)
    text = render_transcript(eng.events, eng.claims, real_state["article_title"])

    assert text.startswith("# Roundtable Review:")
    assert "## Claims" in text
    assert "## Round 1" in text
    assert "**Author on " in text            # stances rendered
    assert " votes " in text                 # votes rendered
    assert "## Outcome" in text
    assert "**AGREED**" in text
    # every claim id appears somewhere
    for cid in eng.claims:
        assert cid in text
    # sane size: a real debate reads as pages, not a stub
    assert len(text) > 3000
