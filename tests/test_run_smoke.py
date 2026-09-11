"""M7 smoke: full live debate loop on article 1 with checkpointing.

Run with: pytest tests/test_run_smoke.py -m smoke -s
Cost: roughly $0.20-0.50 depending on how long the debate runs.
"""

import json

import pytest

from reviewer.orchestrator import run_review
from reviewer.schemas import ClaimStatus

from conftest import FIXTURE_CACHE
from test_scan_smoke import ARTICLE

pytestmark = pytest.mark.smoke


def test_full_debate_loop(tmp_path):
    out_dir = tmp_path / "run"
    result = run_review(ARTICLE, out_dir=out_dir, budget_usd=2.0)
    engine, ledger = result.engine, result.ledger

    # --- outcome summary ---
    statuses = {}
    for c in engine.claims.values():
        statuses.setdefault(c.status, []).append(c.id)
    print(f"\nrounds: {engine.round}, turns: {engine.total_turns}, "
          f"cost: ${ledger.total_usd:.4f}")
    print(f"statuses: { {k: sorted(v) for k, v in statuses.items()} }")
    print(f"consensus: {sorted(c.id for c in engine.consensus_claims())}")
    for c in engine.claims.values():
        if c.status == ClaimStatus.AGREED.value:
            print(f"  AGREED {c.id} (sev {c.aggregate_severity}): "
                  f"{(c.resolution_fix or '')[:80]}")
        elif c.status == ClaimStatus.BLOCKED.value:
            print(f"  BLOCKED {c.id} (sev {c.aggregate_severity}) after {c.rounds} rounds")

    # --- gates ---
    assert engine.finished()
    assert not engine.open_claims(), "loop ended with OPEN claims"
    terminal = {ClaimStatus.AGREED.value, ClaimStatus.DISMISSED.value,
                ClaimStatus.BLOCKED.value, ClaimStatus.RECORDED.value,
                ClaimStatus.UNRESOLVED_BUDGET.value}
    assert all(c.status in terminal for c in engine.claims.values())
    assert ledger.total_usd <= 2.0

    # checkpoint exists, is final, and round-trips
    state = json.loads((out_dir / "checkpoint.json").read_text())
    assert state["tag"] == "final"
    assert state["engine"]["round"] == engine.round
    assert len(state["engine"]["events"]) == len(engine.events)
    assert state["ledger"]["total_usd"] == pytest.approx(ledger.total_usd)

    # save the final checkpoint as a fixture for M8/M9 renderer work ($0 iteration)
    fixture = FIXTURE_CACHE / "run_article1_checkpoint.json"
    fixture.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    print(f"checkpoint fixture saved: {fixture}")
