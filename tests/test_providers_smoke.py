"""M2 smoke tests: real API calls (pennies). Run with: pytest -m smoke

Verifies each provider returns schema-conformant structured output and
records usage tokens. DeepSeek additionally runs a full-article scan-sized
request — its json_object mode has no server-side schema and its verbosity
is the known truncation failure mode; a tiny call proves nothing.
"""

import json

import pytest

from reviewer.article import load_article
from reviewer.costs import CostLedger
from reviewer.findings import repair_findings
from reviewer.providers import AUTHOR_NAME, REVIEWER_NAMES, build_provider
from reviewer.schemas import SCAN_RESPONSE_SCHEMA

pytestmark = pytest.mark.smoke

TINY_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "integer", "minimum": 1, "maximum": 10},
    },
    "required": ["answer", "confidence"],
}

TINY_PROMPT = (
    "What color is a clear daytime sky? Respond as JSON: "
    '{"answer": "<one word>", "confidence": <1-10>}'
)

ARTICLE = "src/content/posts/paper-1-part-1-what-does-it-mean-for-a-computer-to-see-a-picture/index.mdx"


@pytest.mark.parametrize("name", REVIEWER_NAMES + [AUTHOR_NAME])
def test_tiny_structured_call(name):
    ledger = CostLedger()
    provider = build_provider(name, ledger)
    data = provider.structured(TINY_PROMPT, TINY_SCHEMA, 200, label="smoke-tiny")
    assert isinstance(data, dict), f"{name}: not a dict: {data!r}"
    assert "answer" in data, f"{name}: missing answer: {data!r}"
    assert "blue" in str(data["answer"]).lower(), f"{name}: {data!r}"
    assert ledger.total_usd > 0, f"{name}: no cost recorded"
    entry = ledger.entries[0]
    assert entry["input_tokens"] > 0 and entry["output_tokens"] > 0, \
        f"{name}: usage not captured: {entry}"
    print(f"\n{name}: {json.dumps(data)} — ${ledger.total_usd:.6f}")


def test_deepseek_full_article_scan_length():
    """The truncation gauntlet: full article, scan-style prompt, findings schema."""
    article = load_article(ARTICLE)
    ledger = CostLedger()
    provider = build_provider("deepseek-chat", ledger)

    prompt = f"""You are a rigorous fact-checker. Review this article and report every significant issue (severity 3+) across CORRECTNESS, CLARITY, COMPLETENESS, CONSISTENCY, PEDAGOGY, CLICHÉS.

Article: {article.title}

{article.body}

Return JSON: {{"findings": [{{"anchor_type": "quote", "quote": "<verbatim text from the article>", "dimension": "<one of the six>", "issue": "<what's wrong>", "severity": <1-10>, "suggested_fix": "<correction>"}}, ...]}}
Keep each issue and fix under 40 words."""

    data = provider.structured(prompt, SCAN_RESPONSE_SCHEMA, 4000, label="smoke-deepseek-full")
    assert isinstance(data, dict) and "findings" in data, f"bad shape: {str(data)[:200]}"

    findings, dropped = repair_findings(data["findings"], "deepseek-chat")
    print(f"\ndeepseek full-length: {len(findings)} findings, {len(dropped)} dropped, "
          f"${ledger.total_usd:.4f}")
    for d in dropped:
        print(f"  dropped ({d.reason}): {str(d.raw)[:100]}")
    # The pass criterion: a full-length response survives parsing end-to-end
    # and yields usable findings even if the tail was truncated+salvaged.
    assert len(findings) >= 3, f"only {len(findings)} usable findings"
