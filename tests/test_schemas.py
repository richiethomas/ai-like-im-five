"""M1 unit tests: contract shapes, serialization round-trips, aggregates."""

from reviewer.schemas import (
    Anchor,
    AnchorKind,
    BudgetExceededError,
    Claim,
    ClaimStatus,
    DIMENSIONS,
    Dimension,
    Event,
    Finding,
    ReportOutcome,
    SCAN_RESPONSE_SCHEMA,
    STANCE_RESPONSE_SCHEMA,
    Stance,
    StanceKind,
    VOTE_RESPONSE_SCHEMA,
    Vote,
    VoteChoice,
    VoteKind,
    same_dimension_class,
)


def make_finding(model="m1", severity=5, dimension=Dimension.CORRECTNESS.value, quote="some text"):
    return Finding(
        model=model,
        dimension=dimension,
        issue="issue text",
        severity=severity,
        anchor=Anchor(kind=AnchorKind.QUOTE.value, quote=quote),
        suggested_fix="fix it",
    )


# --- serialization round-trips ---------------------------------------------

def test_finding_round_trip():
    f = make_finding()
    f2 = Finding.from_dict(f.to_dict())
    assert f2 == f


def test_claim_round_trip():
    c = Claim(
        id="c001",
        anchor=Anchor(kind=AnchorKind.QUOTE.value, quote="q", start=10, end=11),
        dimension=Dimension.CLARITY.value,
        findings=[make_finding("a"), make_finding("b", severity=7)],
        history=[{"round": 1, "stance": "DEFEND"}],
        endorsements=["c"],
    )
    c2 = Claim.from_dict(c.to_dict())
    assert c2 == c


def test_stance_vote_event_round_trips():
    s = Stance(claim_id="c001", stance=StanceKind.NEGOTIATE.value, rationale="r", proposed_fix="pf")
    assert Stance.from_dict(s.to_dict()) == s
    v = Vote(claim_id="c001", model="m", kind=VoteKind.RESOLUTION.value, choice=VoteChoice.ACCEPT.value)
    assert Vote.from_dict(v.to_dict()) == v
    e = Event(seq=3, type="stance", data={"claim_id": "c001"})
    assert Event.from_dict(e.to_dict()) == e


# --- claim aggregates --------------------------------------------------------

def test_aggregate_severity_is_median_not_max():
    c = Claim(id="c", anchor=Anchor(kind="quote", quote="q"), dimension="CORRECTNESS",
              findings=[make_finding("a", 3), make_finding("b", 4), make_finding("c", 9)])
    assert c.aggregate_severity == 4  # one inflated model doesn't drag it to 9


def test_aggregate_severity_even_count_uses_median_high():
    c = Claim(id="c", anchor=Anchor(kind="quote", quote="q"), dimension="CORRECTNESS",
              findings=[make_finding("a", 3), make_finding("b", 6)])
    assert c.aggregate_severity == 6


def test_aggregate_severity_per_model_dedup():
    # Two findings from the same model count once (at that model's max)
    c = Claim(id="c", anchor=Anchor(kind="quote", quote="q"), dimension="CORRECTNESS",
              findings=[make_finding("a", 2), make_finding("a", 8), make_finding("b", 4)])
    assert c.aggregate_severity == 8  # models: a->8, b->4; median_high([4,8]) = 8


def test_supporter_count_unions_raisers_and_endorsers():
    c = Claim(id="c", anchor=Anchor(kind="quote", quote="q"), dimension="CORRECTNESS",
              findings=[make_finding("a"), make_finding("b")],
              endorsements=["b", "c"])  # b both raised and endorsed: counted once
    assert c.supporter_count == 3
    assert c.models == ["a", "b"]


# --- enums / constants -------------------------------------------------------

def test_dimensions_wire_values():
    assert "CLICHÉS" in DIMENSIONS
    assert "VOICE" in DIMENSIONS
    assert len(DIMENSIONS) == 7


def test_dimension_soft_classes():
    assert same_dimension_class("CLARITY", "PEDAGOGY")
    assert same_dimension_class("CORRECTNESS", "COMPLETENESS")
    assert not same_dimension_class("CLARITY", "CORRECTNESS")
    assert same_dimension_class("CLICHÉS", "CLICHÉS")
    assert same_dimension_class("CLICHÉS", "VOICE")  # both style/tone
    assert not same_dimension_class("VOICE", "CORRECTNESS")


def test_status_and_outcome_enums():
    assert ClaimStatus.UNRESOLVED_BUDGET.value == "UNRESOLVED_BUDGET"
    assert ReportOutcome.NEEDS_CHANGES.value == "NEEDS_CHANGES"
    assert issubclass(BudgetExceededError, Exception)


# --- API schemas -------------------------------------------------------------

def test_scan_schema_shape():
    item = SCAN_RESPONSE_SCHEMA["properties"]["findings"]["items"]
    assert set(item["required"]) == {"anchor_type", "dimension", "issue", "severity"}
    assert item["properties"]["anchor_type"]["enum"] == ["quote", "quote_pair", "section", "global"]
    assert item["properties"]["dimension"]["enum"] == DIMENSIONS


def test_stance_and_vote_schemas_reference_claim_ids():
    stance_item = STANCE_RESPONSE_SCHEMA["properties"]["stances"]["items"]
    assert "claim_id" in stance_item["required"]
    vote_item = VOTE_RESPONSE_SCHEMA["properties"]["votes"]["items"]
    assert "claim_id" in vote_item["required"]
    # ABSTAIN is engine-recorded on failure, never a model-selectable choice
    assert VOTE_RESPONSE_SCHEMA["properties"]["votes"]["items"]["properties"]["choice"]["enum"] == [
        "ACCEPT", "REJECT"]
