"""M5 unit tests: the debate resolution state machine. All offline."""

from reviewer.engine import DebateEngine
from reviewer.schemas import (
    Anchor,
    Claim,
    ClaimStatus,
    Finding,
    Stance,
    Vote,
)

REVIEWERS = ["r1", "r2", "r3", "r4"]


def claim(cid="c001", severities=(5,), models=None, fix="suggested fix"):
    models = models or [f"m{i}" for i in range(len(severities))]
    findings = [
        Finding(model=m, dimension="CORRECTNESS", issue=f"issue {m}", severity=s,
                anchor=Anchor(kind="quote", quote="q", start=0, end=1),
                suggested_fix=fix)
        for m, s in zip(models, severities)
    ]
    return Claim(id=cid, anchor=Anchor(kind="quote", quote="q", start=0, end=1),
                 dimension="CORRECTNESS", findings=findings)


def stance(cid, kind, fix=None):
    return Stance(claim_id=cid, stance=kind, rationale="because", proposed_fix=fix)


def rvote(cid, model, choice, kind="resolution"):
    return Vote(claim_id=cid, model=model, kind=kind, choice=choice)


# --- init / nitpick filter ---------------------------------------------------

def test_nitpick_filter_records_low_severity_regardless_of_raisers():
    c1 = claim("c001", severities=(2, 2, 2))  # 3 models but median 2 -> RECORDED
    c2 = claim("c002", severities=(5,))
    eng = DebateEngine([c1, c2], REVIEWERS)
    assert c1.status == ClaimStatus.RECORDED.value
    assert [c.id for c in eng.open_claims()] == ["c002"]
    assert eng.events[0].type == "claims_formed"
    assert eng.events[0].data["recorded_nitpicks"] == ["c001"]


# --- concede path ------------------------------------------------------------

def test_concede_agrees_immediately_with_fix():
    c = claim()
    eng = DebateEngine([c], REVIEWERS)
    eng.start_round()
    eng.apply_stances([stance("c001", "CONCEDE", fix="the author's fix")])
    assert c.status == ClaimStatus.AGREED.value
    assert c.resolution_fix == "the author's fix"
    assert c.rounds == 0                      # never contested
    assert not eng.contested_claims()
    assert eng.finished()
    assert eng.conceded_this_round() == [c]


def test_concede_without_fix_falls_back_to_highest_severity_suggestion():
    c = claim(severities=(3, 8))
    c.findings[0].suggested_fix = "weak fix"
    c.findings[1].suggested_fix = "strong fix"
    eng = DebateEngine([c], REVIEWERS)
    eng.start_round()
    eng.apply_stances([stance("c001", "CONCEDE")])
    assert c.resolution_fix == "strong fix"


# --- defend path -------------------------------------------------------------

def test_defend_majority_accept_dismisses():
    c = claim()
    eng = DebateEngine([c], REVIEWERS)
    eng.start_round()
    eng.apply_stances([stance("c001", "DEFEND")])
    assert eng.contested_claims() == [c]
    eng.apply_votes([rvote("c001", "r1", "ACCEPT"),
                     rvote("c001", "r2", "ACCEPT"),
                     rvote("c001", "r3", "REJECT")])
    assert c.status == ClaimStatus.DISMISSED.value
    assert c.rounds == 1


def test_defend_majority_reject_continues():
    c = claim()
    eng = DebateEngine([c], REVIEWERS)
    eng.start_round()
    eng.apply_stances([stance("c001", "DEFEND")])
    eng.apply_votes([rvote("c001", "r1", "REJECT"),
                     rvote("c001", "r2", "REJECT"),
                     rvote("c001", "r3", "ACCEPT")])
    assert c.status == ClaimStatus.OPEN.value
    assert c.rounds == 1
    assert not eng.finished()


def test_tie_continues_debate():
    c = claim()
    eng = DebateEngine([c], REVIEWERS)
    eng.start_round()
    eng.apply_stances([stance("c001", "DEFEND")])
    eng.apply_votes([rvote("c001", "r1", "ACCEPT"),
                     rvote("c001", "r2", "REJECT")])
    assert c.status == ClaimStatus.OPEN.value
    assert c.rounds == 1


# --- negotiate path ----------------------------------------------------------

def test_negotiate_majority_accept_agrees_with_negotiated_fix():
    c = claim()
    eng = DebateEngine([c], REVIEWERS)
    eng.start_round()
    eng.apply_stances([stance("c001", "NEGOTIATE", fix="middle ground")])
    eng.apply_votes([rvote("c001", "r1", "ACCEPT"),
                     rvote("c001", "r2", "ACCEPT")])
    assert c.status == ClaimStatus.AGREED.value
    assert c.resolution_fix == "middle ground"


# --- blocker -----------------------------------------------------------------

def test_five_contested_rounds_blocks():
    c = claim()
    eng = DebateEngine([c], REVIEWERS)
    for _ in range(5):
        eng.start_round()
        eng.apply_stances([stance("c001", "DEFEND")])
        eng.apply_votes([rvote("c001", "r1", "REJECT"),
                         rvote("c001", "r2", "REJECT")])
    assert c.status == ClaimStatus.BLOCKED.value
    assert c.rounds == 5
    assert eng.finished()


# --- abstain / failed wave ---------------------------------------------------

def test_too_few_valid_votes_round_not_counted():
    c = claim()
    eng = DebateEngine([c], REVIEWERS)
    eng.start_round()
    eng.apply_stances([stance("c001", "DEFEND")])
    eng.apply_votes([rvote("c001", "r1", "ACCEPT"),
                     rvote("c001", "r2", "ABSTAIN"),
                     rvote("c001", "r3", "ABSTAIN")])
    assert c.status == ClaimStatus.OPEN.value
    assert c.rounds == 0                       # round didn't count
    assert any(e.type == "round_not_counted" for e in eng.events)


def test_abstains_excluded_from_majority():
    c = claim()
    eng = DebateEngine([c], REVIEWERS)
    eng.start_round()
    eng.apply_stances([stance("c001", "DEFEND")])
    # 2 valid (both ACCEPT) + 2 abstain -> dismissed 2-0
    eng.apply_votes([rvote("c001", "r1", "ACCEPT"),
                     rvote("c001", "r2", "ACCEPT"),
                     rvote("c001", "r3", "ABSTAIN"),
                     rvote("c001", "r4", "ABSTAIN")])
    assert c.status == ClaimStatus.DISMISSED.value


# --- endorsements / consensus ------------------------------------------------

def test_endorsements_accumulate_toward_consensus():
    c = claim(severities=(5, 6), models=["m1", "m2"])
    eng = DebateEngine([c], REVIEWERS)
    eng.start_round()
    eng.apply_stances([stance("c001", "CONCEDE", fix="f")])
    eng.apply_votes([rvote("c001", "r3", "ACCEPT", kind="endorsement"),
                     rvote("c001", "r3", "ACCEPT", kind="endorsement"),  # dup ignored
                     rvote("c001", "r4", "REJECT", kind="endorsement")])
    assert c.endorsements == ["r3"]
    assert c.supporter_count == 3              # m1, m2 raisers + r3 endorser
    assert eng.consensus_claims() == [c]


# --- stance validation -------------------------------------------------------

def test_missing_stance_defaults_to_defend():
    c1, c2 = claim("c001"), claim("c002")
    eng = DebateEngine([c1, c2], REVIEWERS)
    eng.start_round()
    eng.apply_stances([stance("c001", "CONCEDE", fix="f")])  # c002 missing
    assert c2.status == ClaimStatus.OPEN.value
    assert eng.contested_claims() == [c2]
    defaulted = [e for e in eng.events if e.type == "stance" and e.data["defaulted"]]
    assert len(defaulted) == 1 and defaulted[0].data["claim_id"] == "c002"


def test_unknown_claim_stance_logged_and_ignored():
    c = claim()
    eng = DebateEngine([c], REVIEWERS)
    eng.start_round()
    eng.apply_stances([stance("c999", "CONCEDE"), stance("c001", "CONCEDE", fix="f")])
    assert any(e.type == "stance_unknown_claim" for e in eng.events)
    assert c.status == ClaimStatus.AGREED.value


# --- stops -------------------------------------------------------------------

def test_global_turn_cap_stops_engine():
    c1, c2 = claim("c001"), claim("c002")
    eng = DebateEngine([c1, c2], REVIEWERS, max_total_turns=2)
    eng.start_round()
    eng.apply_stances([stance("c001", "DEFEND"), stance("c002", "DEFEND")])
    eng.apply_votes([rvote("c001", "r1", "REJECT"), rvote("c001", "r2", "REJECT"),
                     rvote("c002", "r1", "REJECT"), rvote("c002", "r2", "REJECT")])
    # 2 turns consumed -> cap hit -> both forced terminal
    assert eng.finished()
    assert c1.status == ClaimStatus.UNRESOLVED_BUDGET.value
    assert c2.status == ClaimStatus.UNRESOLVED_BUDGET.value
    assert any(e.type == "round_cap_stop" for e in eng.events)


def test_budget_stop_marks_open_claims():
    c = claim()
    eng = DebateEngine([c], REVIEWERS)
    eng.start_round()
    eng.apply_stances([stance("c001", "DEFEND")])
    eng.stop_for_budget(spent_usd=9.97, budget_usd=10.0)
    assert c.status == ClaimStatus.UNRESOLVED_BUDGET.value
    assert eng.finished()
    stop = [e for e in eng.events if e.type == "budget_stop"][0]
    assert stop.data["unresolved_claims"] == ["c001"]


# --- event stream / history --------------------------------------------------

def test_event_stream_for_simple_scenario():
    c = claim()
    eng = DebateEngine([c], REVIEWERS)
    eng.start_round()
    eng.apply_stances([stance("c001", "DEFEND")])
    eng.apply_votes([rvote("c001", "r1", "ACCEPT"), rvote("c001", "r2", "ACCEPT")])
    types = [e.type for e in eng.events]
    assert types == ["claims_formed", "round_start", "stance",
                     "vote", "vote", "resolution"]
    assert [e.seq for e in eng.events] == list(range(6))


def test_history_records_stance_and_votes_per_round():
    c = claim()
    eng = DebateEngine([c], REVIEWERS)
    eng.start_round()
    eng.apply_stances([stance("c001", "DEFEND")])
    eng.apply_votes([rvote("c001", "r1", "REJECT"), rvote("c001", "r2", "ACCEPT")])
    assert len(c.history) == 1
    rec = c.history[0]
    assert rec["round"] == 1 and rec["stance"] == "DEFEND"
    assert {v["model"] for v in rec["votes"]} == {"r1", "r2"}


# --- tests added to kill surviving mutants (mutation-testing pass) -----------

def test_negotiate_majority_reject_stays_open():
    """Kills `accepted OR stance==NEGOTIATE`: a rejected negotiation must NOT
    become AGREED — without this test, that mutation survives the suite."""
    c = claim()
    eng = DebateEngine([c], REVIEWERS)
    eng.start_round()
    eng.apply_stances([stance("c001", "NEGOTIATE", fix="rejected middle ground")])
    eng.apply_votes([rvote("c001", "r1", "REJECT"),
                     rvote("c001", "r2", "REJECT"),
                     rvote("c001", "r3", "ACCEPT")])
    assert c.status == ClaimStatus.OPEN.value
    assert c.resolution_fix is None
    assert c.rounds == 1


def test_conceded_this_round_excludes_prior_rounds_and_contested():
    """Kills the and->or mutants in conceded_this_round's condition chain."""
    c1, c2 = claim("c001"), claim("c002")
    eng = DebateEngine([c1, c2], REVIEWERS)

    eng.start_round()  # round 1
    eng.apply_stances([stance("c001", "CONCEDE", fix="f"),
                       stance("c002", "DEFEND")])
    # contested c002 (OPEN, history round == current) must NOT appear
    assert eng.conceded_this_round() == [c1]
    eng.apply_votes([rvote("c002", "r1", "REJECT"), rvote("c002", "r2", "REJECT")])

    eng.start_round()  # round 2
    eng.apply_stances([stance("c002", "DEFEND")])
    # c001 was conceded in ROUND 1 — must not reappear in round 2's poll
    assert eng.conceded_this_round() == []


def test_conceded_this_round_tolerates_agreed_claim_with_no_history():
    """Kills `history AND ...` -> `history OR ...`: an AGREED claim restored
    from a checkpoint with empty history must not crash the poll."""
    pre_agreed = claim("c001")
    pre_agreed.status = ClaimStatus.AGREED.value  # e.g. restored state
    live = claim("c002")
    eng = DebateEngine([pre_agreed, live], REVIEWERS)
    eng.start_round()
    eng.apply_stances([stance("c002", "DEFEND")])
    assert eng.conceded_this_round() == []  # no IndexError, no false positive
