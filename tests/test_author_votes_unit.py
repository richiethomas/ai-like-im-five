"""M6 unit tests: stance/vote parsing, re-ask, abstain fallback. Offline."""

from reviewer.author import _parse_stances, request_stances, stance_prompt
from reviewer.votes import request_votes, vote_prompt
from reviewer.article import Article
from reviewer.schemas import Anchor, Claim, Finding, Stance

ARTICLE = Article(path="x", title="T", body="Some article body text.", frontmatter={})


def claim(cid, models=("m1",), severity=5):
    findings = [Finding(model=m, dimension="CORRECTNESS", issue=f"issue by {m}",
                        severity=severity,
                        anchor=Anchor(kind="quote", quote="q", start=0, end=1),
                        suggested_fix="fixit")
                for m in models]
    return Claim(id=cid, anchor=Anchor(kind="quote", quote="q", start=0, end=1),
                 dimension="CORRECTNESS", findings=findings)


class FakeProvider:
    def __init__(self, name, responses):
        self.name = name
        self.responses = list(responses)
        self.calls = 0

    def structured(self, prompt, schema, max_tokens, label=""):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


# --- stance parsing ----------------------------------------------------------

def test_parse_stances_filters_garbage():
    data = {"stances": [
        {"claim_id": "c001", "stance": "concede", "rationale": "r", "proposed_fix": "f"},
        {"claim_id": "c002", "stance": "MAYBE", "rationale": "r"},   # bad enum
        {"stance": "DEFEND", "rationale": "r"},                       # no id
        "junk",
        {"claim_id": "c003", "stance": "DEFEND", "rationale": "r"},
    ]}
    stances = _parse_stances(data)
    assert [(s.claim_id, s.stance) for s in stances] == [
        ("c001", "CONCEDE"), ("c003", "DEFEND")]


def test_request_stances_reasks_only_missing():
    c1, c2 = claim("c001"), claim("c002")
    p = FakeProvider("claude-sonnet-5", [
        {"stances": [{"claim_id": "c001", "stance": "DEFEND", "rationale": "r"}]},
        {"stances": [{"claim_id": "c002", "stance": "CONCEDE", "rationale": "r",
                      "proposed_fix": "f"}]},
    ])
    stances = request_stances(ARTICLE, [c1, c2], p)
    assert p.calls == 2
    assert {s.claim_id for s in stances} == {"c001", "c002"}


def test_request_stances_single_call_when_complete():
    c1 = claim("c001")
    p = FakeProvider("claude-sonnet-5", [
        {"stances": [{"claim_id": "c001", "stance": "NEGOTIATE",
                      "rationale": "r", "proposed_fix": "f"}]},
    ])
    stances = request_stances(ARTICLE, [c1], p)
    assert p.calls == 1 and len(stances) == 1


def test_stance_prompt_contains_history_and_full_body():
    c = claim("c001")
    c.history.append({"round": 1, "stance": "DEFEND", "rationale": "my reasons",
                      "votes": [{"model": "r1", "choice": "REJECT",
                                 "kind": "resolution", "claim_id": "c001",
                                 "reason": "unconvincing"}]})
    prompt = stance_prompt(ARTICLE, [c])
    assert ARTICLE.body in prompt              # full article, never truncated
    assert "Round 1: you chose DEFEND" in prompt
    assert "r1 REJECT (unconvincing)" in prompt


# --- vote wave ---------------------------------------------------------------

def stance_for(c, kind="DEFEND", fix=None):
    return Stance(claim_id=c.id, stance=kind, rationale="because", proposed_fix=fix)


def test_vote_wave_parses_and_partitions():
    c1 = claim("c001", models=("gpt-4o-mini",))
    c2 = claim("c002", models=("other-model",))
    c2.resolution_fix = "agreed fix"
    contested = [(c1, stance_for(c1))]

    r1 = FakeProvider("gpt-4o-mini", [
        {"votes": [{"claim_id": "c001", "choice": "REJECT", "reason": "still wrong"}],
         "endorsements": [{"claim_id": "c002", "endorse": True}]},
    ])
    votes = request_votes(ARTICLE, contested, [c2], [r1])
    res = [v for v in votes if v.kind == "resolution"]
    end = [v for v in votes if v.kind == "endorsement"]
    assert [(v.claim_id, v.choice) for v in res] == [("c001", "REJECT")]
    assert [(v.claim_id, v.choice) for v in end] == [("c002", "ACCEPT")]


def test_reviewer_who_raised_claim_not_asked_to_endorse_it():
    c2 = claim("c002", models=("gpt-4o-mini",))
    r1 = FakeProvider("gpt-4o-mini", [])  # would raise IndexError if called
    votes = request_votes(ARTICLE, [], [c2], [r1])
    assert votes == [] and r1.calls == 0


def test_failed_reviewer_abstains_after_retries():
    c1 = claim("c001")
    contested = [(c1, stance_for(c1))]
    boom = RuntimeError("api down")
    r1 = FakeProvider("gpt-4o-mini", [boom, boom, boom])  # VOTE_RETRIES=2 -> 3 attempts
    votes = request_votes(ARTICLE, contested, [], [r1])
    assert r1.calls == 3
    assert [(v.claim_id, v.choice) for v in votes] == [("c001", "ABSTAIN")]


def test_unvoted_contested_claims_become_abstains():
    c1, c2 = claim("c001"), claim("c002")
    contested = [(c1, stance_for(c1)), (c2, stance_for(c2))]
    r1 = FakeProvider("gpt-4o-mini", [
        {"votes": [{"claim_id": "c001", "choice": "ACCEPT"}]},  # c002 skipped
    ])
    votes = request_votes(ARTICLE, contested, [], [r1])
    by_id = {v.claim_id: v.choice for v in votes}
    assert by_id == {"c001": "ACCEPT", "c002": "ABSTAIN"}


def test_transient_failure_recovers_on_retry():
    c1 = claim("c001")
    contested = [(c1, stance_for(c1))]
    r1 = FakeProvider("gpt-4o-mini", [
        RuntimeError("blip"),
        {"votes": [{"claim_id": "c001", "choice": "ACCEPT", "reason": "ok"}]},
    ])
    votes = request_votes(ARTICLE, contested, [], [r1])
    assert [(v.claim_id, v.choice) for v in votes] == [("c001", "ACCEPT")]


def test_vote_prompt_mentions_stance_and_fix():
    c1 = claim("c001")
    prompt = vote_prompt(ARTICLE, "gpt-4o-mini",
                         [(c1, stance_for(c1, "NEGOTIATE", fix="the fix"))], [])
    assert "AUTHOR'S STANCE: NEGOTIATE" in prompt
    assert "AUTHOR'S PROPOSED FIX: the fix" in prompt
    assert ARTICLE.body in prompt
