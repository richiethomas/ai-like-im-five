"""Thread-safe cost ledger with per-article budget enforcement.

Budget is enforced at round boundaries WITH headroom (the orchestrator asks
"can I afford one more round?" before starting it) — never mid-wave, so a
parallel vote wave is never killed half-recorded.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .schemas import BUDGET_USD, BudgetExceededError


@dataclass(frozen=True)
class Price:
    input_per_mtok: float   # USD per 1M input tokens
    output_per_mtok: float  # USD per 1M output tokens


# Estimated prices, USD per million tokens. Order-of-magnitude correct for
# budget enforcement; verify against provider pricing pages before trusting
# the dollar figures for invoicing.
PRICES: dict[str, Price] = {
    "gpt-4o-mini": Price(0.15, 0.60),
    "deepseek-chat": Price(0.27, 1.10),
    "gemini-3.5-flash": Price(0.30, 2.50),
    "llama-3.3-70b": Price(0.88, 0.88),
    "claude-sonnet-5": Price(3.00, 15.00),
}


class CostLedger:
    def __init__(self, budget_usd: float = BUDGET_USD):
        self.budget_usd = budget_usd
        self._lock = threading.Lock()
        self._entries: list[dict] = []
        self._total = 0.0

    def record(self, model: str, input_tokens: int, output_tokens: int,
               label: str = "") -> float:
        """Record one API call's usage. Returns the cost of that call."""
        price = PRICES.get(model)
        if price is None:
            raise KeyError(f"No price entry for model {model!r}")
        cost = (input_tokens * price.input_per_mtok
                + output_tokens * price.output_per_mtok) / 1_000_000
        with self._lock:
            self._entries.append({
                "model": model,
                "label": label,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
            })
            self._total += cost
        return cost

    @property
    def total_usd(self) -> float:
        with self._lock:
            return self._total

    @property
    def entries(self) -> list[dict]:
        with self._lock:
            return list(self._entries)

    def check(self, headroom_usd: float = 0.0) -> None:
        """Raise BudgetExceededError if spend + headroom would exceed budget."""
        with self._lock:
            if self._total + headroom_usd > self.budget_usd:
                raise BudgetExceededError(
                    f"spent ${self._total:.4f} + headroom ${headroom_usd:.4f} "
                    f"exceeds budget ${self.budget_usd:.2f}"
                )

    def by_model(self) -> dict[str, dict]:
        """Aggregate spend per model (for metrics/reporting)."""
        agg: dict[str, dict] = {}
        for e in self.entries:
            m = agg.setdefault(e["model"], {"calls": 0, "input_tokens": 0,
                                            "output_tokens": 0, "cost_usd": 0.0})
            m["calls"] += 1
            m["input_tokens"] += e["input_tokens"]
            m["output_tokens"] += e["output_tokens"]
            m["cost_usd"] += e["cost_usd"]
        return agg

    def to_dict(self) -> dict:
        return {"budget_usd": self.budget_usd, "total_usd": self.total_usd,
                "entries": self.entries}

    @classmethod
    def from_dict(cls, d: dict) -> "CostLedger":
        ledger = cls(budget_usd=d["budget_usd"])
        for e in d["entries"]:
            ledger._entries.append(dict(e))
            ledger._total += e["cost_usd"]
        return ledger
