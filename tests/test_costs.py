"""M2 unit tests: cost ledger math, budget enforcement, thread-safety. Offline."""

import threading

import pytest

from reviewer.costs import PRICES, CostLedger
from reviewer.providers import AUTHOR_NAME, REVIEWER_NAMES
from reviewer.schemas import BudgetExceededError


def test_all_roster_models_priced():
    for name in REVIEWER_NAMES + [AUTHOR_NAME]:
        assert name in PRICES, f"no price entry for {name}"


def test_record_math():
    ledger = CostLedger(budget_usd=10.0)
    # gpt-4o-mini: $0.15/M in, $0.60/M out
    cost = ledger.record("gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.75)
    assert ledger.total_usd == pytest.approx(0.75)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        CostLedger().record("gpt-99-ultra", 100, 100)


def test_budget_check_with_headroom():
    ledger = CostLedger(budget_usd=1.0)
    ledger.record("gpt-4o-mini", 4_000_000, 500_000)  # 0.60 + 0.30 = $0.90
    ledger.check()                    # 0.90 <= 1.00: fine
    ledger.check(headroom_usd=0.05)   # 0.95 <= 1.00: fine
    with pytest.raises(BudgetExceededError):
        ledger.check(headroom_usd=0.2)  # 1.10 > 1.00


def test_by_model_aggregation():
    ledger = CostLedger()
    ledger.record("gpt-4o-mini", 100, 50, label="scan")
    ledger.record("gpt-4o-mini", 200, 80, label="vote")
    ledger.record("deepseek-chat", 300, 90, label="scan")
    agg = ledger.by_model()
    assert agg["gpt-4o-mini"]["calls"] == 2
    assert agg["gpt-4o-mini"]["input_tokens"] == 300
    assert agg["deepseek-chat"]["calls"] == 1


def test_round_trip():
    ledger = CostLedger(budget_usd=5.0)
    ledger.record("gpt-4o-mini", 1000, 500, label="x")
    restored = CostLedger.from_dict(ledger.to_dict())
    assert restored.budget_usd == 5.0
    assert restored.total_usd == pytest.approx(ledger.total_usd)
    assert restored.entries == ledger.entries


def test_thread_safety():
    ledger = CostLedger(budget_usd=1e9)
    n_threads, n_calls = 8, 200

    def hammer():
        for _ in range(n_calls):
            ledger.record("gpt-4o-mini", 1000, 1000)

    threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected_calls = n_threads * n_calls
    assert len(ledger.entries) == expected_calls
    per_call = (1000 * 0.15 + 1000 * 0.60) / 1_000_000
    assert ledger.total_usd == pytest.approx(expected_calls * per_call)
