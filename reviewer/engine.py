"""Debate resolution engine: pure state machine, no I/O.

Definitions (pinned by the spec and the design review):
- Round N = one (author stance call over all OPEN claims) + (reviewer vote
  wave over contested claims) cycle. The initial scan is round 0.
- A claim's `rounds` counter increments only in rounds where it was contested
  AND received a valid vote wave (>= MIN_VALID_VOTES non-ABSTAIN votes).
- CONCEDE -> AGREED immediately. DEFEND: majority ACCEPT (of valid votes)
  -> DISMISSED. NEGOTIATE: majority ACCEPT -> AGREED with the negotiated fix.
  Same majority threshold for both paths — unanimity would let one stubborn
  reviewer convert every defense into a blocker.
- Majority = strictly more ACCEPT than REJECT. Ties continue the debate.
- MAX_CONTESTED_ROUNDS_PER_CLAIM contested rounds without resolution -> BLOCKED.
- Global safety cap: MAX_TOTAL_DEBATE_TURNS summed per-claim contested rounds.
- Aggregate severity <= NITPICK_SEVERITY_CEILING -> RECORDED at init, never
  debated, regardless of how many models raised it.

The engine emits typed Events as it goes; the transcript is a pure renderer
over that stream. All mutation happens here so the orchestrator stays thin.
"""

from __future__ import annotations

from .schemas import (
    CONSENSUS_MODEL_COUNT,
    MAX_CONTESTED_ROUNDS_PER_CLAIM,
    MAX_TOTAL_DEBATE_TURNS,
    MIN_VALID_VOTES,
    NITPICK_SEVERITY_CEILING,
    Claim,
    ClaimStatus,
    Event,
    Stance,
    StanceKind,
    Vote,
    VoteChoice,
    VoteKind,
)


class DebateEngine:
    def __init__(self, claims: list[Claim], reviewer_names: list[str],
                 max_rounds_per_claim: int = MAX_CONTESTED_ROUNDS_PER_CLAIM,
                 max_total_turns: int = MAX_TOTAL_DEBATE_TURNS,
                 initial_events: list[Event] | None = None):
        self.claims = {c.id: c for c in claims}
        self.reviewer_names = list(reviewer_names)
        self.max_rounds_per_claim = max_rounds_per_claim
        self.max_total_turns = max_total_turns
        self.round = 0            # scan is round 0; first debate round is 1
        self.total_turns = 0      # sum of per-claim contested-round increments
        self.stopped = False
        # Single event stream for the whole run: the orchestrator's pre-engine
        # events (scan, drops) come first, then everything the engine emits.
        self.events: list[Event] = list(initial_events or [])
        # stance currently under debate, per open claim
        self._pending_stance: dict[str, Stance] = {}

        for c in claims:
            if c.status == ClaimStatus.OPEN.value and \
                    c.aggregate_severity <= NITPICK_SEVERITY_CEILING:
                c.status = ClaimStatus.RECORDED.value
        self._emit("claims_formed", {
            "claims": [c.id for c in claims],
            "recorded_nitpicks": [c.id for c in claims
                                  if c.status == ClaimStatus.RECORDED.value],
        })

    # --- event log -----------------------------------------------------------

    def _emit(self, type_: str, data: dict) -> None:
        self.events.append(Event(seq=len(self.events), type=type_, data=data))

    # --- claim views ---------------------------------------------------------

    def open_claims(self) -> list[Claim]:
        return [c for c in self.claims.values()
                if c.status == ClaimStatus.OPEN.value]

    def contested_claims(self) -> list[Claim]:
        """Open claims whose current stance is DEFEND/NEGOTIATE (need votes)."""
        return [c for c in self.open_claims() if c.id in self._pending_stance]

    def pending_stances(self) -> dict[str, Stance]:
        """The stance under debate per contested claim — INCLUDING engine-
        defaulted DEFENDs. The vote wave must be built from this, not from
        the author's returned list, or defaulted stances never get voted on."""
        return dict(self._pending_stance)

    def conceded_this_round(self) -> list[Claim]:
        """Claims that reached AGREED in the current round (for endorsement polling)."""
        out = []
        for c in self.claims.values():
            if c.status == ClaimStatus.AGREED.value and c.history \
                    and c.history[-1].get("round") == self.round \
                    and c.history[-1].get("stance") == StanceKind.CONCEDE.value:
                out.append(c)
        return out

    def finished(self) -> bool:
        return self.stopped or not self.open_claims() \
            or self.total_turns >= self.max_total_turns

    def consensus_claims(self) -> list[Claim]:
        return [c for c in self.claims.values()
                if c.supporter_count >= CONSENSUS_MODEL_COUNT]

    # --- round lifecycle -----------------------------------------------------

    def start_round(self) -> int:
        if self.total_turns >= self.max_total_turns:
            self._stop_open_claims("round_cap_stop",
                                   {"total_turns": self.total_turns})
            raise RuntimeError("round cap reached")
        self.round += 1
        self._pending_stance = {}
        self._emit("round_start", {"round": self.round})
        return self.round

    def apply_stances(self, stances: list[Stance]) -> None:
        """Apply the author's stances for this round. Open claims with no
        stance (after the caller's re-ask) default to DEFEND."""
        by_id = {}
        for s in stances:
            if s.claim_id not in self.claims:
                self._emit("stance_unknown_claim", {"claim_id": s.claim_id})
                continue
            by_id[s.claim_id] = s

        for c in self.open_claims():
            s = by_id.get(c.id)
            defaulted = s is None
            if defaulted:
                s = Stance(claim_id=c.id, stance=StanceKind.DEFEND.value,
                           rationale="(no stance returned; defaulted to DEFEND)")
            self._emit("stance", {"round": self.round, "claim_id": c.id,
                                  "stance": s.stance, "rationale": s.rationale,
                                  "proposed_fix": s.proposed_fix,
                                  "defaulted": defaulted})
            record = {"round": self.round, "stance": s.stance,
                      "rationale": s.rationale, "proposed_fix": s.proposed_fix,
                      "votes": []}
            c.history.append(record)

            if s.stance == StanceKind.CONCEDE.value:
                c.status = ClaimStatus.AGREED.value
                c.resolution_fix = s.proposed_fix or self._best_suggested_fix(c)
                self._emit("resolution", {"round": self.round, "claim_id": c.id,
                                          "status": c.status,
                                          "detail": "author conceded"})
            else:
                self._pending_stance[c.id] = s

    def apply_votes(self, votes: list[Vote]) -> None:
        """Apply one vote wave: resolution votes on contested claims plus
        endorsement votes on claims conceded this round."""
        by_claim: dict[str, list[Vote]] = {}
        for v in votes:
            if v.claim_id not in self.claims:
                self._emit("vote_unknown_claim", {"claim_id": v.claim_id,
                                                  "model": v.model})
                continue
            self._emit("vote", {"round": self.round, "claim_id": v.claim_id,
                                "model": v.model, "kind": v.kind,
                                "choice": v.choice, "reason": v.reason})
            by_claim.setdefault(v.claim_id, []).append(v)

        # Endorsements (consensus counting; no effect on status)
        for cid, vs in by_claim.items():
            c = self.claims[cid]
            for v in vs:
                if v.kind == VoteKind.ENDORSEMENT.value \
                        and v.choice == VoteChoice.ACCEPT.value \
                        and v.model not in c.endorsements:
                    c.endorsements.append(v.model)

        # Resolution votes on contested claims
        for c in self.contested_claims():
            vs = [v for v in by_claim.get(c.id, [])
                  if v.kind == VoteKind.RESOLUTION.value]
            if c.history and c.history[-1].get("round") == self.round:
                c.history[-1]["votes"] = [v.to_dict() for v in vs]
            valid = [v for v in vs if v.choice != VoteChoice.ABSTAIN.value]
            if len(valid) < MIN_VALID_VOTES:
                # Vote wave failed (API errors): round doesn't count against
                # the claim's budget — otherwise dead reviewers manufacture
                # false blockers.
                self._emit("round_not_counted", {
                    "round": self.round, "claim_id": c.id,
                    "valid_votes": len(valid)})
                continue

            c.rounds += 1
            self.total_turns += 1
            accepts = sum(1 for v in valid if v.choice == VoteChoice.ACCEPT.value)
            rejects = len(valid) - accepts
            stance = self._pending_stance[c.id]
            accepted = accepts > rejects

            if accepted and stance.stance == StanceKind.DEFEND.value:
                c.status = ClaimStatus.DISMISSED.value
                detail = f"defense accepted {accepts}-{rejects}"
            elif accepted and stance.stance == StanceKind.NEGOTIATE.value:
                c.status = ClaimStatus.AGREED.value
                c.resolution_fix = stance.proposed_fix or self._best_suggested_fix(c)
                detail = f"negotiation accepted {accepts}-{rejects}"
            elif c.rounds >= self.max_rounds_per_claim:
                c.status = ClaimStatus.BLOCKED.value
                detail = (f"no resolution after {c.rounds} contested rounds "
                          f"(last vote {accepts}-{rejects})")
            else:
                self._emit("round_continue", {"round": self.round,
                                              "claim_id": c.id,
                                              "accepts": accepts,
                                              "rejects": rejects})
                continue

            self._emit("resolution", {"round": self.round, "claim_id": c.id,
                                      "status": c.status, "detail": detail})

        self._pending_stance = {}
        if self.total_turns >= self.max_total_turns and self.open_claims():
            self._stop_open_claims("round_cap_stop",
                                   {"total_turns": self.total_turns})

    def stop_for_budget(self, spent_usd: float, budget_usd: float) -> None:
        self._stop_open_claims("budget_stop", {"spent_usd": round(spent_usd, 4),
                                               "budget_usd": budget_usd})

    def force_stop(self, reason: str) -> None:
        """External safety stop (e.g. orchestrator loop guard)."""
        self._stop_open_claims("round_cap_stop", {"reason": reason})

    def _stop_open_claims(self, event_type: str, data: dict) -> None:
        stopped_ids = []
        for c in self.open_claims():
            # Status name says BUDGET but covers any externally-forced stop
            # (budget or round cap); the stop event carries the actual reason.
            c.status = ClaimStatus.UNRESOLVED_BUDGET.value
            stopped_ids.append(c.id)
        self.stopped = True
        self._emit(event_type, {**data, "unresolved_claims": stopped_ids})

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _best_suggested_fix(c: Claim) -> str | None:
        """Fallback fix when a stance carries none: highest-severity finding's."""
        with_fix = [f for f in c.findings if f.suggested_fix]
        if not with_fix:
            return None
        return max(with_fix, key=lambda f: f.severity).suggested_fix

    # --- checkpointing -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "total_turns": self.total_turns,
            "stopped": self.stopped,
            "claims": [c.to_dict() for c in self.claims.values()],
            "events": [e.to_dict() for e in self.events],
            "reviewer_names": self.reviewer_names,
        }
