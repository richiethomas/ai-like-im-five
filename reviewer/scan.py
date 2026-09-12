"""Reviewer scans: one structured call per reviewer covering ALL seven dimensions.

(The old system scanned once per dimension — 6x the cost for the same coverage.)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from .article import Article
from .providers import Provider
from .schemas import SCAN_RESPONSE_SCHEMA

SCAN_MAX_TOKENS = 4000


def scan_prompt(article: Article) -> str:
    return f"""You are one of several independent expert reviewers evaluating a blog article written for non-technical business stakeholders. Other reviewers are checking the same article; your findings will be debated with the author.

Article title: {article.title}

Article body:
---
{article.body}
---

Review ALL seven dimensions:
1. CORRECTNESS — factual accuracy: are claims true, definitions right?
2. CLARITY — could a passage mislead or confuse? are analogies apt?
3. COMPLETENESS — missing caveats, absent context a reader needs
4. CONSISTENCY — self-contradictions, terms used inconsistently
5. PEDAGOGY — appropriate for non-technical readers? too dense or too thin? Flag a proper noun or technical term ONLY if its first use has NO inline plain-language explanation at all. A term that already carries any gloss at first use is settled: do not demand a longer or different gloss (rate such wishes severity 2 at most), and never flag later mentions of an already-glossed term.
6. CLICHÉS — vague filler and LLM clichés ("honest", "genuine", "game-changer", "rides on", "shines for", "land" as a verb). Also flag grandiose totalizing frames ("X is the entire problem/point/story of Y", "this is what it's all about") and metaphor flourishes that state no mechanism ("the ways the gap opens up"): rewrite as a direct statement of the underlying fact. Also flag summary or closing flourishes that add no information: empty adjective wrap-ups ("it's simple, powerful, and elegant", "clean and intuitive"), grandiose closers ("this is the foundation for everything that comes next", "and that's the magic of X", "and that changes everything"). The fix is to delete them or replace with a concrete fact. Also flag figurative or non-literal phrasing where a literal word is available: metaphorical verbs like "unpack", "dive into", "dig into", "peel back", "walk through", "under the hood", "tease apart", and similar. The fix is the literal statement ("unpack it" -> "explain it in detail", "dive into" -> "cover in detail"). ANY use of an em-dash (—) is a violation: report every occurrence, severity 3, with a fix that rewrites the sentence using a period, comma, colon, or parentheses.
7. VOICE — the author is a self-described learner writing a public learning log for non-technical readers, NOT an expert. Two things to flag:
   (a) Overclaimed authority: field-wide empirical claims the author cannot have observed ("almost everyone does this", "the most common mistake", "everyone knows"), verdicts stated as personal decree ("the right way and the wrong way", "here's what NOT to do"), or presumptive framings that tell the reader how to feel ("the obvious answer", "obviously", "of course", "as you'd expect"). Fix: attribute the judgment to its real source (the CS231n notes, named experts, standard practice), or reframe as what the author has personally laid out and observed. Quoting experts is fine; claiming their vantage point is not.
   (b) Tone that doesn't fit a humble learner: snark, sarcasm, condescension, forced jokiness, or hype ("Perfect score! You're a genius!", "it's that simple", "buckle up"). Fix: rewrite plainly.

Severity calibration:
- 9-10: factually wrong or seriously misleading
- 6-8: likely to confuse or misinform a non-technical reader
- 3-5: meaningful improvement to accuracy or clarity
- 1-2: nitpick — DO NOT report these

Report ONLY significant issues (severity 3+). No style preferences, no trivia.

For each finding choose the anchor type:
- "quote" — the issue is about specific text. Copy the passage VERBATIM from the article body (exact characters, punctuation and all; no paraphrasing; at most ~200 characters). Findings whose quote does not appear verbatim in the article are discarded.
- "quote_pair" — a contradiction between two passages: put the first in "quote" and the second in "quote_b", both verbatim.
- "section" — something is MISSING from a specific part: name the section or heading in "section".
- "global" — a document-wide issue.

Return JSON:
{{"findings": [{{"anchor_type": "quote", "quote": "<verbatim>", "dimension": "CORRECTNESS", "issue": "<what is wrong, max 40 words>", "severity": 5, "suggested_fix": "<concrete correction, max 40 words>"}}, ...]}}

Keep "issue" and "suggested_fix" each under 40 words. A careful review of an explainer this length typically surfaces several severity-3+ issues across the seven dimensions. Report each one you find. Do not pad with nitpicks to hit a count."""


def run_scans(article: Article, providers: list[Provider],
              max_workers: int = 4) -> dict[str, dict | Exception]:
    """Scan with every reviewer in parallel.

    Returns {provider.name: parsed dict} — or the Exception the call raised,
    so the caller can log per-model failures without losing the others.
    """
    prompt = scan_prompt(article)
    results: dict[str, dict | Exception] = {}

    def one(provider: Provider):
        try:
            return provider.structured(prompt, SCAN_RESPONSE_SCHEMA,
                                       SCAN_MAX_TOKENS, label="scan")
        except Exception:  # noqa: BLE001 — one retry for transient API failures
            return provider.structured(prompt, SCAN_RESPONSE_SCHEMA,
                                       SCAN_MAX_TOKENS, label="scan:retry")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(one, p): p.name for p in providers}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
            except Exception as e:  # noqa: BLE001 — per-model isolation is the point
                results[name] = e

    return results
