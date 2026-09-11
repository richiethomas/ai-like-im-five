"""M7 unit tests: debate loop with fakes — budget stop, loop guard,
checkpoint-on-crash. All offline."""

import json

import pytest

from reviewer.article import Article
from reviewer.costs import CostLedger
from reviewer.engine import DebateEngine
from reviewer.orchestrator import checkpoint, debate_loop
from reviewer.schemas import Anchor, Claim, ClaimStatus, Finding

ARTICLE = Article(path="a.mdx", title="T", body="Body text.", frontmatter={})
REVIEWERS = ["r1", "r2"]


def claim(cid, severity=5):
    f = Finding(model="m1", dimension="CORRECTNESS", issue="i", severity=severity,
                anchor=Anchor(kind="quote", quote="q", start=0, end=1),
                suggested_fix="fix")
    return Claim(id=cid, anchor=Anchor(kind="quote", quote="q", start=0, end=1),
                 dimension="CORRECTNESS", findings=[f])


class FakeAuthor:
    """Always defends everything."""

    name = "claude-sonnet-5"

    def __init__(self, stance="DEFEND", crash_on_call=None):
        self.calls = 0
        self.stance = stance
        self.crash_on_call = crash_on_call

    def structured(self, prompt, schema, max_tokens, label=""):
        self.calls += 1
        if self.crash_on_call and self.calls == self.crash_on_call:
            raise KeyboardInterrupt
        import re
        ids = sorted(set(re.findall(r"CLAIM (c\d+)", prompt)))
        return {"stances": [{"claim_id": cid, "stance": self.stance,
                             "rationale": "r", "proposed_fix": "pf"}
                            for cid in ids]}


class FakeReviewer:
    """Always rejects the author's stance."""

    def __init__(self, name, choice="REJECT"):
        self.name = name
        self.choice = choice

    def structured(self, prompt, schema, max_tokens, label=""):
        import re
        ids = sorted(set(re.findall(r"CLAIM (c\d+)", prompt)))
        return {"votes": [{"claim_id": cid, "choice": self.choice, "reason": "x"}
                          for cid in ids],
                "endorsements": []}


def test_loop_runs_to_blocker():
    c = claim("c001")
    eng = DebateEngine([c], REVIEWERS, max_rounds_per_claim=3)
    ledger = CostLedger(budget_usd=100)
    debate_loop(ARTICLE, eng, FakeAuthor(), [FakeReviewer("r1"), FakeReviewer("r2")],
                ledger)
    assert c.status == ClaimStatus.BLOCKED.value
    assert c.rounds == 3
    assert eng.finished()


def test_loop_resolves_on_accept():
    c = claim("c001")
    eng = DebateEngine([c], REVIEWERS)
    ledger = CostLedger(budget_usd=100)
    debate_loop(ARTICLE, eng, FakeAuthor(),
                [FakeReviewer("r1", "ACCEPT"), FakeReviewer("r2", "ACCEPT")], ledger)
    assert c.status == ClaimStatus.DISMISSED.value
    assert eng.round == 1


def test_budget_stop_before_round():
    c = claim("c001")
    eng = DebateEngine([c], REVIEWERS)
    ledger = CostLedger(budget_usd=0.10)  # below the 0.50 headroom immediately
    author = FakeAuthor()
    debate_loop(ARTICLE, eng, author, [FakeReviewer("r1")], ledger)
    assert author.calls == 0                      # stopped BEFORE spending
    assert c.status == ClaimStatus.UNRESOLVED_BUDGET.value
    assert any(e.type == "budget_stop" for e in eng.events)


def test_loop_guard_stops_uncounted_round_spin():
    # Reviewers always fail -> every wave abstains -> rounds never count ->
    # without the guard this loops forever.
    class DeadReviewer:
        def __init__(self, name):
            self.name = name

        def structured(self, *a, **k):
            raise RuntimeError("down")

    c = claim("c001")
    eng = DebateEngine([c], REVIEWERS)
    ledger = CostLedger(budget_usd=100)
    debate_loop(ARTICLE, eng, FakeAuthor(),
                [DeadReviewer("r1"), DeadReviewer("r2")], ledger)
    assert c.status == ClaimStatus.UNRESOLVED_BUDGET.value
    assert any(e.type == "round_cap_stop" for e in eng.events)


def test_checkpoint_written_per_round_and_on_crash(tmp_path):
    c = claim("c001")
    eng = DebateEngine([c], REVIEWERS, max_rounds_per_claim=5)
    ledger = CostLedger(budget_usd=100)
    tags = []

    def ckpt(tag):
        tags.append(tag)
        checkpoint(tmp_path, tag, ARTICLE, eng, ledger, {})

    author = FakeAuthor(crash_on_call=3)  # dies during round 3's stance call
    with pytest.raises(KeyboardInterrupt):
        try:
            debate_loop(ARTICLE, eng, author,
                        [FakeReviewer("r1"), FakeReviewer("r2")], ledger,
                        checkpoint_fn=ckpt)
        finally:
            ckpt("final")  # what run_review's finally does

    assert tags == ["round1", "round2", "final"]
    state = json.loads((tmp_path / "checkpoint.json").read_text())
    assert state["tag"] == "final"
    assert state["engine"]["round"] == 3
    assert state["engine"]["claims"][0]["rounds"] == 2  # two counted rounds survived
    assert len(state["engine"]["events"]) > 0


def test_defaulted_stances_still_reach_vote_wave():
    """Regression: when the author returns NOTHING parseable, the engine
    defaults every stance to DEFEND — and the vote wave must still run on
    those defaults. The first e2e run silently skipped voting for 4 rounds
    because pairs were built from the author's (empty) list."""
    class SilentAuthor:
        name = "claude-sonnet-5"

        def structured(self, prompt, schema, max_tokens, label=""):
            return {"stances": []}  # parses to nothing, twice (re-ask too)

    c = claim("c001")
    eng = DebateEngine([c], REVIEWERS)
    ledger = CostLedger(budget_usd=100)
    debate_loop(ARTICLE, eng, SilentAuthor(),
                [FakeReviewer("r1", "ACCEPT"), FakeReviewer("r2", "ACCEPT")],
                ledger)
    # Defaulted DEFEND + both reviewers accept -> dismissed in round 1
    assert c.status == ClaimStatus.DISMISSED.value
    assert c.rounds == 1
    assert not any(e.type == "round_not_counted" for e in eng.events)
