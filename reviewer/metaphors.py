"""Propose metaphors for a post's most abstract ideas, roundtable-style.

This is a GENERATIVE companion to the review pipeline, not a critique. Where
reviewer/scan.py flags problems in existing text (and its CLARITY dimension
only judges whether metaphors already present are apt), this looks for the
concepts that would most benefit from a concrete metaphor and proposes one for
each, the way the assembly-line metaphor works for backpropagation.

Pipeline (four models, same providers/cost ledger as the reviewer):
  1. propose  — each model finds the article's most abstract concepts and
                proposes a metaphor for each (parallel).
  2. merge    — one cheap model dedupes/merges them into a canonical list.
  3. score    — each model rates every metaphor on three strength axes
                (aptness, accessibility, faithfulness), 1-5 each (parallel).
  4. select   — median-aggregate the scores into a strength out of 15 and
                surface every metaphor at or above a threshold, for review.

Faithfulness matters because a vivid metaphor that mirrors the wrong mechanism
teaches a wrong intuition. The assembly-line framing is what produced Part 8's
severity-8 error (blame = gradient x local, not loss x factor), so every
metaphor also carries a "breaks_down" note naming where it leaks.

Artifacts: metaphors.json (full data) + metaphors.md (readable, thresholded).

Usage:
    .venv-fact-check/bin/python -m reviewer.metaphors <article.mdx> [--out-dir DIR] [--min-score N]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from .article import Article, load_article
from .costs import CostLedger
from .providers import Provider, REVIEWER_NAMES, build_provider
from .schemas import BUDGET_USD

# One model does the cheap dedupe/merge step.
MERGE_MODEL = "gpt-4o-mini"
METAPHORS_PER_MODEL = 4
PROPOSE_MAX_TOKENS = 2200
MERGE_MAX_TOKENS = 2600
SCORE_MAX_TOKENS = 1600

# Strength is aptness + accessibility + faithfulness, each a 1-5 median, so it
# runs 3 to 15. Default surfaces metaphors averaging roughly 3.7+ across axes.
DEFAULT_MIN_SCORE = 11.0

# Item shape shared by propose + merge.
_ITEM_PROPS = {
    "concept": {"type": "string"},
    "metaphor": {"type": "string"},
    "mapping": {"type": "string"},
    "breaks_down": {"type": "string"},
}

PROPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "metaphors": {
            "type": "array",
            "items": {"type": "object", "properties": _ITEM_PROPS,
                      "required": ["concept", "metaphor", "mapping"]},
        }
    },
    "required": ["metaphors"],
}

MERGE_SCHEMA = PROPOSE_SCHEMA

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "aptness": {"type": "integer", "minimum": 1, "maximum": 5},
                    "accessibility": {"type": "integer", "minimum": 1, "maximum": 5},
                    "faithfulness": {"type": "integer", "minimum": 1, "maximum": 5},
                    "note": {"type": "string"},
                },
                "required": ["id", "aptness", "accessibility", "faithfulness"],
            },
        }
    },
    "required": ["scores"],
}

_AXES = ("aptness", "accessibility", "faithfulness")


def propose_prompt(article: Article) -> str:
    return f"""You are helping an author of a plain-language blog that explains AI concepts to a curious, mostly non-technical audience. The author wants a concrete metaphor for the hardest-to-picture ideas in each post, the way an assembly line (a component moves down a line of stations, and when the output is bad you trace the blame back station by station) makes backpropagation click.

Article title: {article.title}

Article body:
---
{article.body}
---

Find the {METAPHORS_PER_MODEL} concepts in THIS article that are the most abstract or hardest to picture, the passages that would most benefit from a concrete metaphor, and propose one metaphor for each. A strong metaphor draws on something the reader already knows from everyday life and mirrors the actual mechanism, not just the mood.

If a concept already has a good metaphor in the article, skip it, unless you can offer a clearly better one.

For each, give:
- concept: the specific idea from this post the metaphor is for (max 20 words).
- metaphor: the everyday thing you compare it to, stated concretely (max 40 words).
- mapping: how the parts line up (which part of the metaphor stands for which part of the concept) (max 50 words).
- breaks_down: the one place this metaphor is NOT like the concept, a specific mismatch, so it is not used to teach a wrong intuition. Name the mismatch; do not restate what the metaphor shows. (For the assembly line: a real station never sends a corrected part back upstream, but backprop does push blame backward.) (max 30 words).

Return JSON:
{{"metaphors": [{{"concept": "...", "metaphor": "...", "mapping": "...", "breaks_down": "..."}}, ...]}}"""


def merge_prompt(article: Article, proposals: list[dict]) -> str:
    listing = "\n".join(
        f'- ({p.get("concept", "?")}) {p.get("metaphor", "")}'
        for p in proposals
    )
    return f"""Below are metaphors several people proposed for concepts in a blog post titled "{article.title}". Some overlap.

Merge them into a de-duplicated list. Two proposals are "the same" only if they explain the same concept using the same everyday source (for example, two water-flow metaphors for the same idea); in that case combine them into one clear entry, keeping the best wording. Two different sources for the same concept are DISTINCT and both stay. Do NOT collapse different metaphors into one, and do not summarize the whole set into a single item. Aim for roughly 6 to 14 distinct entries. Drop only what is vague or not tied to the post.

Proposals:
{listing}

For each merged metaphor return: concept (max 20 words), metaphor (max 40 words), mapping (max 50 words), breaks_down (the one specific place the metaphor is NOT like the concept, a mismatch, NOT a restatement of what it shows; keep the proposer's mismatch if it named one; max 30 words).

Return JSON:
{{"metaphors": [{{"concept": "...", "metaphor": "...", "mapping": "...", "breaks_down": "..."}}, ...]}}"""


def score_prompt(article: Article, ideas: list[dict]) -> str:
    listing = "\n".join(
        f'{i}. ({idea.get("concept", "?")}) {idea.get("metaphor", "")} '
        f'[maps: {idea.get("mapping", "")}]'
        for i, idea in enumerate(ideas)
    )
    return f"""Rate these proposed metaphors for a plain-language blog post titled "{article.title}", written for a curious, mostly non-technical audience.

For each metaphor, score three things 1-5:
- aptness: how well the metaphor mirrors the actual mechanism of the concept (5 = the parts correspond closely, 1 = only a surface or mood resemblance).
- accessibility: how familiar the everyday source is to a non-technical reader (5 = everyone knows it, 1 = needs its own explanation).
- faithfulness: how little it misleads (5 = its leaks are minor and harmless, 1 = it teaches a wrong intuition about how the concept works). A vivid metaphor that mirrors the WRONG mechanism must score low here.

Metaphors:
{listing}

Return JSON with one entry per metaphor, keyed by its number (id):
{{"scores": [{{"id": 0, "aptness": 4, "accessibility": 5, "faithfulness": 4, "note": "<max 20 words>"}}, ...]}}"""


def _call(provider: Provider, prompt: str, schema: dict, max_tokens: int,
          label: str) -> dict | None:
    """One structured call with a single retry, matching run_scans' isolation:
    a model that fails twice is dropped rather than sinking the whole run."""
    try:
        return provider.structured(prompt, schema, max_tokens, label=label)
    except Exception:  # noqa: BLE001 — retry once for transient API failures
        try:
            return provider.structured(prompt, schema, max_tokens,
                                       label=f"{label}:retry")
        except Exception:  # noqa: BLE001 — per-model isolation
            return None


def run_proposals(article: Article, providers: list[Provider]) -> list[dict]:
    prompt = propose_prompt(article)
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(providers)) as pool:
        futures = {pool.submit(_call, p, prompt, PROPOSE_SCHEMA,
                               PROPOSE_MAX_TOKENS, "propose"): p.name
                   for p in providers}
        for fut in as_completed(futures):
            data = fut.result()
            for item in (data or {}).get("metaphors", []):
                if isinstance(item, dict) and item.get("metaphor"):
                    item["_by"] = futures[fut]
                    out.append(item)
    return out


def _norm(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()


def _dedupe_by_metaphor(proposals: list[dict]) -> list[dict]:
    """Fallback dedupe: same concept + same metaphor source counts as one."""
    seen, deduped = set(), []
    for p in proposals:
        key = (_norm(p.get("concept", "")), _norm(p.get("metaphor", ""))[:60])
        if key[1] and key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped


# Below this count, the LLM merge has almost certainly over-collapsed the list
# (gpt-4o-mini occasionally summarizes many proposals into 1-2 items), so we
# don't trust it and dedupe deterministically instead.
_MIN_TRUSTED_MERGE = 5


def merge_ideas(article: Article, proposals: list[dict],
                merger: Provider) -> list[dict]:
    """Dedupe/merge via one cheap model, guarded against over-merging.

    The merge model is high-variance: it can return a good 6-14 item list, or
    occasionally collapse everything into one. Retry once, then fall back to a
    deterministic dedupe so a bad merge never silently discards ideas.
    """
    for label in ("merge", "merge:retry2"):
        data = _call(merger, merge_prompt(article, proposals), MERGE_SCHEMA,
                     MERGE_MAX_TOKENS, label)
        merged = [i for i in (data or {}).get("metaphors", [])
                  if isinstance(i, dict) and i.get("metaphor")]
        if len(merged) >= _MIN_TRUSTED_MERGE:
            return merged
    return _dedupe_by_metaphor(proposals)


def score_ideas(article: Article, ideas: list[dict],
                providers: list[Provider]) -> dict[int, dict]:
    """Each model scores every metaphor; aggregate to median per axis and a
    strength that is the sum of the three medians (3 to 15)."""
    prompt = score_prompt(article, ideas)
    per_idea: dict[int, dict[str, list[int]]] = {
        i: {ax: [] for ax in _AXES} for i in range(len(ideas))
    }
    with ThreadPoolExecutor(max_workers=len(providers)) as pool:
        futures = [pool.submit(_call, p, prompt, SCORE_SCHEMA,
                               SCORE_MAX_TOKENS, "score") for p in providers]
        for fut in as_completed(futures):
            data = fut.result()
            for s in (data or {}).get("scores", []):
                idx = s.get("id")
                if isinstance(idx, int) and idx in per_idea:
                    for ax in _AXES:
                        v = s.get(ax)
                        if isinstance(v, int) and 1 <= v <= 5:
                            per_idea[idx][ax].append(v)
    agg: dict[int, dict] = {}
    for i, sc in per_idea.items():
        medians = {ax: (statistics.median(sc[ax]) if sc[ax] else 0)
                   for ax in _AXES}
        votes = min(len(sc[ax]) for ax in _AXES)
        agg[i] = {**medians, "votes": votes,
                  "strength": round(sum(medians.values()), 1)}
    return agg


def select_strong(ideas: list[dict], scores: dict[int, dict],
                  min_score: float) -> list[dict]:
    """Every metaphor at or above the strength threshold, strongest first."""
    ranked = sorted(
        ({**idea, "score": scores.get(i, {})} for i, idea in enumerate(ideas)),
        key=lambda e: e["score"].get("strength", 0), reverse=True)
    return [e for e in ranked if e["score"].get("strength", 0) >= min_score]


def render_markdown(article: Article, surfaced: list[dict],
                    min_score: float) -> str:
    lines = [f"# Metaphor opportunities: {article.title}", "",
             "Proposed by a four-model roundtable and scored on strength "
             "(aptness + accessibility + faithfulness, each 1-5, so 15 max). "
             f"Showing every metaphor scoring {min_score:g}+ of 15. Every "
             "metaphor leaks; the \"Breaks down\" line names where, so it is "
             "not used to teach a wrong intuition.", ""]
    if not surfaced:
        lines.append(f"*No metaphor scored {min_score:g} or higher this run. "
                     "Lower --min-score to see the full ranked list in "
                     "metaphors.json.*")
        return "\n".join(lines)
    for e in surfaced:
        sc = e.get("score", {})
        lines.append(f"## {e.get('concept', '')}  "
                     f"(strength {sc.get('strength', 0):g}/15)")
        lines.append(f"*aptness {sc.get('aptness', 0):g}, "
                     f"accessibility {sc.get('accessibility', 0):g}, "
                     f"faithfulness {sc.get('faithfulness', 0):g}, "
                     f"{sc.get('votes', 0)} votes*")
        lines.append("")
        lines.append(f"**Metaphor:** {e.get('metaphor', '')}")
        lines.append("")
        lines.append(f"**Mapping:** {e.get('mapping', '')}")
        if e.get("breaks_down"):
            lines.append("")
            lines.append(f"**Breaks down:** {e['breaks_down']}")
        lines.append("")
    return "\n".join(lines)


def run_metaphors(article_path: str | Path, out_dir: str | Path | None = None,
                  budget_usd: float = BUDGET_USD,
                  min_score: float = DEFAULT_MIN_SCORE) -> dict:
    article = load_article(article_path)
    if out_dir is None:
        out_dir = Path("review-runs") / Path(article_path).parent.name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = CostLedger(budget_usd=budget_usd)
    providers = [build_provider(n, ledger) for n in REVIEWER_NAMES]

    proposals = run_proposals(article, providers)
    merger = build_provider(MERGE_MODEL, ledger)
    ideas = merge_ideas(article, proposals, merger)
    scores = score_ideas(article, ideas, providers) if ideas else {}
    surfaced = select_strong(ideas, scores, min_score) if ideas else []
    all_ranked = sorted(
        [{**idea, "score": scores.get(i, {})} for i, idea in enumerate(ideas)],
        key=lambda e: e["score"].get("strength", 0), reverse=True)

    payload = {
        "article": article.title,
        "article_path": str(article.path),
        "min_score": min_score,
        "proposed_count": len(proposals),
        "merged_count": len(ideas),
        "surfaced": surfaced,
        "all_ranked": all_ranked,
        "cost": {"budget_usd": ledger.budget_usd, "total_usd": ledger.total_usd},
    }
    (out_dir / "metaphors.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "metaphors.md").write_text(
        render_markdown(article, surfaced, min_score))
    payload["out_dir"] = str(out_dir)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Propose metaphors for a blog article's abstract concepts.")
    parser.add_argument("article", help="path to the .mdx/.md article")
    parser.add_argument("--out-dir", default=None,
                        help="artifact directory (default review-runs/<article-slug>)")
    parser.add_argument("--budget", type=float, default=BUDGET_USD)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                        help=f"surface metaphors scoring this or higher, out of "
                             f"15 (default {DEFAULT_MIN_SCORE:g})")
    args = parser.parse_args(argv)

    load_dotenv(Path(__file__).parent.parent / ".env")
    result = run_metaphors(args.article, out_dir=args.out_dir,
                           budget_usd=args.budget, min_score=args.min_score)

    print(f"\n{'=' * 60}")
    print(f"Metaphors: {result['article']}")
    print(f"{'=' * 60}")
    print(f"Proposed: {result['proposed_count']}  "
          f"merged: {result['merged_count']}  "
          f"surfaced (>= {result['min_score']:g}): {len(result['surfaced'])}")
    print(f"Cost: ${result['cost']['total_usd']:.4f} "
          f"of ${result['cost']['budget_usd']:.2f}")
    print(f"\nSurfaced (strength >= {result['min_score']:g} of 15):")
    for e in result["surfaced"]:
        sc = e.get("score", {})
        print(f"  [{sc.get('strength', 0):>4}/15] {e.get('concept', '')}")
    if not result["surfaced"]:
        print("  (none above threshold; see all_ranked in metaphors.json)")
    print(f"\nArtifacts in {result['out_dir']}/: metaphors.md, metaphors.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
