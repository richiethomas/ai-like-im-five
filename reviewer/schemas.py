"""Pinned contracts for the review system.

Every module imports its types from here. Changing a shape here is a breaking
change to every milestone built on top of it — extend, don't mutate.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field, asdict
from enum import Enum


# ---------------------------------------------------------------------------
# Enums / constants
# ---------------------------------------------------------------------------

class Dimension(str, Enum):
    CORRECTNESS = "CORRECTNESS"
    CLARITY = "CLARITY"
    COMPLETENESS = "COMPLETENESS"
    CONSISTENCY = "CONSISTENCY"
    PEDAGOGY = "PEDAGOGY"
    CLICHES = "CLICHÉS"  # wire value keeps the accent used in prompts/old data


DIMENSIONS = [d.value for d in Dimension]

# Soft equivalence classes for alignment: a dimension mismatch across these
# groups argues against merging two findings; within a group it does not.
DIMENSION_CLASSES = [
    {Dimension.CLARITY.value, Dimension.PEDAGOGY.value},
    {Dimension.CORRECTNESS.value, Dimension.CONSISTENCY.value, Dimension.COMPLETENESS.value},
    {Dimension.CLICHES.value},
]


def same_dimension_class(a: str, b: str) -> bool:
    for cls in DIMENSION_CLASSES:
        if a in cls and b in cls:
            return True
    return a == b


class AnchorKind(str, Enum):
    QUOTE = "quote"            # verbatim article text; validated + span-resolved
    QUOTE_PAIR = "quote_pair"  # two locations (consistency contradictions)
    SECTION = "section"        # a named/described region (absent-content findings)
    GLOBAL = "global"          # document-level


class ClaimStatus(str, Enum):
    OPEN = "OPEN"
    AGREED = "AGREED"                        # author conceded or negotiation accepted
    DISMISSED = "DISMISSED"                  # author's defense accepted by majority
    BLOCKED = "BLOCKED"                      # 5 contested rounds without resolution
    RECORDED = "RECORDED"                    # nitpick (aggregate severity <= 2): logged, not debated
    UNRESOLVED_BUDGET = "UNRESOLVED_BUDGET"  # debate stopped by budget/round cap


class StanceKind(str, Enum):
    CONCEDE = "CONCEDE"
    DEFEND = "DEFEND"
    NEGOTIATE = "NEGOTIATE"


class VoteChoice(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    ABSTAIN = "ABSTAIN"  # recorded when a reviewer errored out after retries


class VoteKind(str, Enum):
    RESOLUTION = "resolution"    # vote on the author's DEFEND/NEGOTIATE stance
    ENDORSEMENT = "endorsement"  # co-sign of a CONCEDEd claim (consensus counting)


class ReportOutcome(str, Enum):
    PASS = "PASS"                    # nothing open or agreed above severity 2
    NEEDS_CHANGES = "NEEDS_CHANGES"  # agreed changes with severity > 2 pending
    BLOCKED = "BLOCKED"              # at least one blocker
    INCOMPLETE = "INCOMPLETE"        # budget or round-cap stop left claims unresolved


# Debate parameters (from the original spec)
MAX_CONTESTED_ROUNDS_PER_CLAIM = 5
MAX_TOTAL_DEBATE_TURNS = 100
NITPICK_SEVERITY_CEILING = 2      # aggregate severity <= this -> RECORDED, not debated
CONSENSUS_MODEL_COUNT = 3         # models raising/endorsing = consensus (reporting stat)
PASS_BLOCK_SUPPORT_COUNT = 2      # agreed items with this many supporters gate PASS
MIN_VALID_VOTES = 2               # a vote wave with fewer valid votes doesn't count
BUDGET_USD = 10.0


class BudgetExceededError(Exception):
    """Raised when a spend would exceed the per-article budget."""


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Anchor:
    kind: str  # AnchorKind value
    quote: str | None = None       # quote / quote_pair (first location)
    quote_b: str | None = None     # quote_pair (second location)
    section: str | None = None     # section: heading or short description
    # Resolved offsets into Article.body (set by anchoring, None until then)
    start: int | None = None
    end: int | None = None
    start_b: int | None = None
    end_b: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Anchor":
        return cls(**d)


@dataclass
class Finding:
    model: str
    dimension: str  # Dimension value
    issue: str
    severity: int   # 1-10, clamped by the repair layer
    anchor: Anchor
    suggested_fix: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        d = dict(d)
        d["anchor"] = Anchor.from_dict(d["anchor"])
        return cls(**d)


@dataclass
class Claim:
    id: str                 # stable id assigned at alignment ("c001", ...)
    anchor: Anchor          # canonical anchor (representative/widest span)
    dimension: str          # representative dimension (most common among findings)
    findings: list[Finding] = field(default_factory=list)
    status: str = ClaimStatus.OPEN.value
    rounds: int = 0         # contested rounds consumed (increments only when contested)
    history: list[dict] = field(default_factory=list)  # per-round stance/vote records
    resolution_fix: str | None = None  # the fix that was agreed, when status == AGREED
    endorsements: list[str] = field(default_factory=list)  # models co-signing beyond raisers

    @property
    def models(self) -> list[str]:
        return sorted({f.model for f in self.findings})

    @property
    def supporter_count(self) -> int:
        """Models that raised OR endorsed the claim (consensus counting)."""
        return len(set(self.models) | set(self.endorsements))

    @property
    def aggregate_severity(self) -> int:
        """Median of per-model severities (max is gameable by one inflated model).

        For each model, use its highest-severity finding in this claim, then take
        the median across models. Even counts round half up via median_high.
        """
        per_model: dict[str, int] = {}
        for f in self.findings:
            per_model[f.model] = max(per_model.get(f.model, 0), f.severity)
        if not per_model:
            return 0
        return int(statistics.median_high(sorted(per_model.values())))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "anchor": self.anchor.to_dict(),
            "dimension": self.dimension,
            "findings": [f.to_dict() for f in self.findings],
            "status": self.status,
            "rounds": self.rounds,
            "history": self.history,
            "resolution_fix": self.resolution_fix,
            "endorsements": self.endorsements,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Claim":
        d = dict(d)
        d["anchor"] = Anchor.from_dict(d["anchor"])
        d["findings"] = [Finding.from_dict(f) for f in d["findings"]]
        return cls(**d)


@dataclass
class Stance:
    claim_id: str
    stance: str  # StanceKind value
    rationale: str
    proposed_fix: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Stance":
        return cls(**d)


@dataclass
class Vote:
    claim_id: str
    model: str
    kind: str    # VoteKind value
    choice: str  # VoteChoice value
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Vote":
        return cls(**d)


@dataclass
class Event:
    """Typed record for the transcript/metrics. `data` shape depends on `type`.

    Types emitted by the engine (transcript renders these):
      scan_complete, finding_dropped, claims_formed, round_start,
      stance, vote, resolution, budget_stop, round_cap_stop
    """
    seq: int
    type: str
    data: dict

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(**d)


# ---------------------------------------------------------------------------
# JSON Schemas for structured API calls (plain JSON Schema; providers translate
# to their dialect — Gemini strips unsupported keys, DeepSeek gets json_object)
# ---------------------------------------------------------------------------

SCAN_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "anchor_type": {"type": "string",
                                    "enum": [k.value for k in AnchorKind]},
                    "quote": {"type": "string"},
                    "quote_b": {"type": "string"},
                    "section": {"type": "string"},
                    "dimension": {"type": "string", "enum": DIMENSIONS},
                    "issue": {"type": "string"},
                    "severity": {"type": "integer", "minimum": 1, "maximum": 10},
                    "suggested_fix": {"type": "string"},
                },
                "required": ["anchor_type", "dimension", "issue", "severity"],
            },
        }
    },
    "required": ["findings"],
}

STANCE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "stances": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "stance": {"type": "string",
                               "enum": [s.value for s in StanceKind]},
                    "rationale": {"type": "string"},
                    "proposed_fix": {"type": "string"},
                },
                "required": ["claim_id", "stance", "rationale"],
            },
        }
    },
    "required": ["stances"],
}

VOTE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "votes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "choice": {"type": "string",
                               "enum": [VoteChoice.ACCEPT.value, VoteChoice.REJECT.value]},
                    "reason": {"type": "string"},
                },
                "required": ["claim_id", "choice"],
            },
        },
        "endorsements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "endorse": {"type": "boolean"},
                },
                "required": ["claim_id", "endorse"],
            },
        },
    },
    "required": ["votes"],
}

ADJUDICATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "same_claim": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["same_claim"],
}
