"""M4 unit tests: claim alignment. Offline (fake adjudicator where needed)."""

from reviewer.alignment import (
    MAX_CLAIM_SPAN_CHARS,
    SIM_MERGE,
    SIM_REJECT,
    align_findings,
    issue_similarity,
)
from reviewer.schemas import Anchor, AnchorKind, Finding

BODY = "x" * 2000  # spans are synthetic; body only materializes widened quotes


def qf(model, start, end, issue, dimension="CORRECTNESS", severity=5, kind="quote"):
    return Finding(model=model, dimension=dimension, issue=issue, severity=severity,
                   anchor=Anchor(kind=kind, quote=BODY[start:end], start=start, end=end))


def sf(model, issue, dimension="COMPLETENESS", section="Intro", severity=4):
    return Finding(model=model, dimension=dimension, issue=issue, severity=severity,
                   anchor=Anchor(kind=AnchorKind.SECTION.value, section=section))


class FakeAdjudicator:
    """Scripted verdicts; records what it was asked."""

    def __init__(self, verdict=True):
        self.verdict = verdict
        self.calls = 0

    def structured(self, prompt, schema, max_tokens, label=""):
        self.calls += 1
        return {"same_claim": self.verdict, "reason": "scripted"}


# --- similarity --------------------------------------------------------------

def test_issue_similarity_identical_and_disjoint():
    assert issue_similarity("the math is wrong here", "the math is wrong here") == 1.0
    assert issue_similarity("the math is wrong", "analogy may confuse readers") < SIM_REJECT


def test_issue_similarity_reordered_words():
    a = "claim overstates what CNNs can do"
    b = "overstates what CNNs can do, this claim"
    assert issue_similarity(a, b) >= 0.5


# --- merging behavior --------------------------------------------------------

def test_overlapping_spans_similar_issues_merge():
    fs = [qf("m1", 100, 150, "the 1.44 million figure is presented without byte context"),
          qf("m2", 120, 180, "1.44 million figure lacks byte context for readers")]
    res = align_findings(fs, BODY)
    assert len(res.claims) == 1
    assert res.claims[0].models == ["m1", "m2"]


def test_overlapping_spans_unrelated_issues_stay_separate():
    fs = [qf("m1", 100, 150, "arithmetic here is incorrect"),
          qf("m2", 120, 180, "sentence uses jargon a layperson cannot parse",
             dimension="PEDAGOGY")]
    res = align_findings(fs, BODY)
    assert len(res.claims) == 2


def test_disjoint_spans_never_merge():
    fs = [qf("m1", 100, 150, "the math is wrong in this passage"),
          qf("m2", 500, 550, "the math is wrong in this passage")]
    res = align_findings(fs, BODY)
    assert len(res.claims) == 2


def test_mega_claim_guard_blocks_transitive_chain():
    issue = "this whole passage overstates model capability significantly"
    fs = [qf("m1", 0, 200, issue),
          qf("m2", 150, 400, issue),    # m1+m2 union = 400 chars: allowed
          qf("m3", 350, 600, issue)]    # +m3 union = 600 > 500: refused
    res = align_findings(fs, BODY)
    spans = [(c.anchor.start, c.anchor.end) for c in res.claims]
    assert len(res.claims) == 2, spans
    for c in res.claims:
        assert c.anchor.end - c.anchor.start <= MAX_CLAIM_SPAN_CHARS


def test_same_model_overlapping_findings_merge():
    fs = [qf("m1", 100, 160, "figure lacks byte-format context"),
          qf("m1", 110, 170, "the figure needs byte format context for clarity")]
    res = align_findings(fs, BODY)
    assert len(res.claims) == 1
    assert res.claims[0].models == ["m1"]


def test_section_findings_merge_on_similarity():
    fs = [sf("m1", "training process is never mentioned in the classification intro"),
          sf("m2", "classification intro never mentions the training process")]
    res = align_findings(fs, BODY)
    assert len(res.claims) == 1


def test_quote_never_merges_with_section():
    fs = [qf("m1", 100, 150, "training process is never mentioned here",
             dimension="COMPLETENESS"),
          sf("m2", "training process is never mentioned here")]
    res = align_findings(fs, BODY)
    assert len(res.claims) == 2


# --- adjudication ------------------------------------------------------------

def test_ambiguous_pair_goes_to_adjudicator():
    # Overlapping spans, middling similarity (between SIM_REJECT and SIM_MERGE)
    a = "the compression explanation skips lossy versus lossless distinction"
    b = "compression is oversimplified and could mislead about image quality"
    assert SIM_REJECT <= issue_similarity(a, b) < SIM_MERGE
    fs = [qf("m1", 100, 150, a, dimension="COMPLETENESS"),
          qf("m2", 120, 170, b, dimension="CLARITY")]

    adj_yes = FakeAdjudicator(verdict=True)
    res = align_findings(fs, BODY, adjudicator=adj_yes)
    assert adj_yes.calls == 1
    assert len(res.claims) == 1
    assert len(res.adjudications) == 1

    adj_no = FakeAdjudicator(verdict=False)
    res = align_findings(fs, BODY, adjudicator=adj_no)
    assert len(res.claims) == 2


def test_clear_cases_skip_adjudicator():
    adj = FakeAdjudicator()
    fs = [qf("m1", 100, 150, "the 1.44 million figure lacks byte context"),
          qf("m2", 120, 180, "1.44 million figure lacks byte context"),   # clear merge
          qf("m3", 130, 190, "unrelated grammatical problem with tense")]  # clear reject
    align_findings(fs, BODY, adjudicator=adj)
    assert adj.calls == 0


# --- claim assembly ----------------------------------------------------------

def test_claim_ids_deterministic_and_ordered_by_position():
    fs = [qf("m1", 500, 550, "later issue about the analogy"),
          qf("m2", 100, 150, "earlier issue about arithmetic")]
    res = align_findings(fs, BODY)
    assert [c.id for c in res.claims] == ["c001", "c002"]
    assert res.claims[0].anchor.start == 100


def test_canonical_anchor_covers_group_span():
    fs = [qf("m1", 100, 140, "figure lacks byte context for readers"),
          qf("m2", 120, 200, "the figure lacks byte context for the reader")]
    res = align_findings(fs, BODY)
    c = res.claims[0]
    assert (c.anchor.start, c.anchor.end) == (100, 200)
    assert c.anchor.quote == BODY[100:200]


def test_representative_dimension_majority():
    issue = "this figure lacks essential byte context"
    fs = [qf("m1", 100, 150, issue, dimension="CLARITY"),
          qf("m2", 110, 160, issue, dimension="CLARITY"),
          qf("m3", 120, 170, issue, dimension="COMPLETENESS")]
    res = align_findings(fs, BODY)
    assert len(res.claims) == 1
    assert res.claims[0].dimension == "CLARITY"
