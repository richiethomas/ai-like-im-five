"""Generate 'try-it-yourself' experiment ideas for a post, roundtable-style.

This is a GENERATIVE companion to the review pipeline, not a critique. Where
reviewer/scan.py flags problems in existing text, this proposes hands-on
experiments a reader could run to internalize the post's concept.

Pipeline (four models, same providers/cost ledger as the reviewer):
  1. propose  — each model proposes several experiments (parallel).
  2. merge    — one cheap model dedupes/merges them into a canonical list.
  3. score    — each model rates every canonical idea on feasibility and how
                much it illuminates the concept (parallel).
  4. rank     — median-aggregate the scores, rank, and guarantee a couple of
                Python experiments make the shortlist.

Artifacts: experiments.json (full data) + experiments.md (readable shortlist).

Usage:
    .venv-fact-check/bin/python -m reviewer.experiments <article.mdx> [--out-dir DIR] [--top N]
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
IDEAS_PER_MODEL = 5
PROPOSE_MAX_TOKENS = 2200
MERGE_MAX_TOKENS = 2600
SCORE_MAX_TOKENS = 1600

SETUP_LEVELS = ["none", "browser", "python", "dataset"]

# Item shape shared by propose + merge.
_ITEM_PROPS = {
    "title": {"type": "string"},
    "steps": {"type": "string"},
    "shows": {"type": "string"},
    "setup": {"type": "string", "enum": SETUP_LEVELS},
    "python": {"type": "string"},
}

PROPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "experiments": {
            "type": "array",
            "items": {"type": "object", "properties": _ITEM_PROPS,
                      "required": ["title", "steps", "shows", "setup"]},
        }
    },
    "required": ["experiments"],
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
                    "feasibility": {"type": "integer", "minimum": 1, "maximum": 5},
                    "illumination": {"type": "integer", "minimum": 1, "maximum": 5},
                    "note": {"type": "string"},
                },
                "required": ["id", "feasibility", "illumination"],
            },
        }
    },
    "required": ["scores"],
}

_SETUP_GUIDE = (
    'setup is one of: "none" (a thought experiment or pen-and-paper, no tools), '
    '"browser" (an online demo/tool, nothing to install), '
    '"python" (a few lines a curious reader could run), '
    '"dataset" (needs downloading real data, more effort). '
    'For "python" experiments, put a short runnable snippet in "python".'
)


def propose_prompt(article: Article) -> str:
    return f"""You are helping design "try-it-yourself" experiments for readers of a plain-language blog that explains AI concepts to a curious, mostly non-technical audience (some readers will run a little Python, most won't).

Article title: {article.title}

Article body:
---
{article.body}
---

Propose {IDEAS_PER_MODEL} hands-on experiments a reader could do to build intuition for THIS post's specific concept. Good experiments are concrete, produce an observable result, and make the post's idea click. Bias toward low setup, but include at least one or two that use a few lines of Python (this blog does ship Python-based experiments).

For each experiment give:
- title: short name.
- steps: what the reader actually does, concrete (max 60 words).
- shows: the specific idea from this post it makes concrete (max 30 words).
- setup: {_SETUP_GUIDE}
- python: for setup "python" only, a short runnable snippet (a few lines); omit otherwise.

Avoid vague suggestions ("play around with the model"). Each must have a clear action and a clear payoff tied to this post.

Return JSON:
{{"experiments": [{{"title": "...", "steps": "...", "shows": "...", "setup": "none|browser|python|dataset", "python": "<snippet or empty>"}}, ...]}}"""


def merge_prompt(article: Article, proposals: list[dict]) -> str:
    listing = "\n".join(
        f'- [{p.get("setup", "?")}] {p.get("title", "")}: {p.get("steps", "")} '
        f'(shows: {p.get("shows", "")})'
        for p in proposals
    )
    return f"""Below are experiment ideas several people proposed for a blog post titled "{article.title}". Many overlap.

Merge them into a de-duplicated list. Two proposals are "the same" only if a reader would take the same actions AND learn the same thing; in that case combine them into one clear entry (keep the best steps and payoff). Otherwise keep them as separate entries. Do NOT collapse different experiments into one, and do not summarize the whole set into a single item. Aim for roughly 6 to 12 distinct entries. Drop only what is vague or not tied to the post. Preserve the best Python experiments, with their snippets.

Proposals:
{listing}

For each merged experiment return: title, steps (max 60 words), shows (max 30 words), setup (one of none/browser/python/dataset), and python (a short runnable snippet, for python setup only).

Return JSON:
{{"experiments": [{{"title": "...", "steps": "...", "shows": "...", "setup": "...", "python": "<snippet or empty>"}}, ...]}}"""


def score_prompt(article: Article, ideas: list[dict]) -> str:
    listing = "\n".join(
        f'{i}. [{idea.get("setup", "?")}] {idea.get("title", "")} — '
        f'{idea.get("steps", "")} (shows: {idea.get("shows", "")})'
        for i, idea in enumerate(ideas)
    )
    return f"""Rate these proposed "try-it-yourself" experiments for a plain-language blog post titled "{article.title}", written for a curious, mostly non-technical audience.

For each experiment, score two things 1-5:
- feasibility: can a motivated reader actually do this without expert knowledge? (5 = anyone can, 1 = needs real expertise/hardware)
- illumination: how much does doing it make THIS post's concept click? (5 = a genuine aha, 1 = busywork)

Experiments:
{listing}

Return JSON with one entry per experiment, keyed by its number (id):
{{"scores": [{{"id": 0, "feasibility": 4, "illumination": 5, "note": "<max 20 words>"}}, ...]}}"""


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
            for item in (data or {}).get("experiments", []):
                if isinstance(item, dict) and item.get("title"):
                    item["_by"] = futures[fut]
                    out.append(item)
    return out


def _norm_title(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()


def _dedupe_by_title(proposals: list[dict]) -> list[dict]:
    seen, deduped = set(), []
    for p in proposals:
        key = _norm_title(p.get("title", ""))
        if key and key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped


# Below this count, the LLM merge has almost certainly over-collapsed the list
# (gpt-4o-mini occasionally summarizes 20 proposals into 1-2 items), so we don't
# trust it and dedupe deterministically instead.
_MIN_TRUSTED_MERGE = 5


def merge_ideas(article: Article, proposals: list[dict],
                merger: Provider) -> list[dict]:
    """Dedupe/merge via one cheap model, guarded against over-merging.

    The merge model is high-variance: it can return a good 6-12 item list, or
    occasionally collapse everything into one. Retry once, then fall back to a
    deterministic title-dedupe so a bad merge never silently discards ideas.
    """
    for label in ("merge", "merge:retry2"):
        data = _call(merger, merge_prompt(article, proposals), MERGE_SCHEMA,
                     MERGE_MAX_TOKENS, label)
        merged = [i for i in (data or {}).get("experiments", [])
                  if isinstance(i, dict) and i.get("title")]
        if len(merged) >= _MIN_TRUSTED_MERGE:
            return merged
    return _dedupe_by_title(proposals)


def score_ideas(article: Article, ideas: list[dict],
                providers: list[Provider]) -> dict[int, dict]:
    """Each model scores every idea; aggregate to median feasibility/illumination."""
    prompt = score_prompt(article, ideas)
    per_idea: dict[int, dict[str, list[int]]] = {
        i: {"feasibility": [], "illumination": []} for i in range(len(ideas))
    }
    with ThreadPoolExecutor(max_workers=len(providers)) as pool:
        futures = [pool.submit(_call, p, prompt, SCORE_SCHEMA,
                               SCORE_MAX_TOKENS, "score") for p in providers]
        for fut in as_completed(futures):
            data = fut.result()
            for s in (data or {}).get("scores", []):
                idx = s.get("id")
                if isinstance(idx, int) and idx in per_idea:
                    for k in ("feasibility", "illumination"):
                        v = s.get(k)
                        if isinstance(v, int) and 1 <= v <= 5:
                            per_idea[idx][k].append(v)
    agg: dict[int, dict] = {}
    for i, sc in per_idea.items():
        feas = statistics.median(sc["feasibility"]) if sc["feasibility"] else 0
        illum = statistics.median(sc["illumination"]) if sc["illumination"] else 0
        agg[i] = {"feasibility": feas, "illumination": illum,
                  "votes": len(sc["feasibility"]),
                  "combined": round(feas + illum, 1)}
    return agg


MIN_PYTHON = 2  # experiments a reader could run in a few lines, guaranteed in the shortlist


def rank(ideas: list[dict], scores: dict[int, dict], top: int) -> list[dict]:
    """Rank by combined score, but guarantee up to MIN_PYTHON Python experiments
    in the shortlist (when that many exist) by displacing the weakest *non-Python*
    entries, so the reader always gets some runnable ones."""
    def combined(i: int) -> float:
        return scores.get(i, {}).get("combined", 0)

    def is_py(i: int) -> bool:
        return ideas[i].get("setup") == "python"

    order = sorted(range(len(ideas)), key=combined, reverse=True)
    chosen = order[:top]
    py_chosen = [i for i in chosen if is_py(i)]
    if len(py_chosen) < MIN_PYTHON:
        extra_py = [i for i in order[top:] if is_py(i)][:MIN_PYTHON - len(py_chosen)]
        if extra_py:
            non_py_chosen = [i for i in chosen if not is_py(i)]
            drop = set(non_py_chosen[-len(extra_py):])  # weakest non-Python entries
            chosen = [i for i in chosen if i not in drop] + extra_py
            chosen.sort(key=combined, reverse=True)
    return [{**ideas[i], "score": scores.get(i, {})} for i in chosen]


_SETUP_LABEL = {"none": "no setup", "browser": "browser",
                "python": "a little Python", "dataset": "needs a dataset"}


def render_markdown(article: Article, shortlist: list[dict]) -> str:
    lines = [f"# Try-it-yourself ideas: {article.title}", "",
             "Ranked by a four-model roundtable (feasibility + how much it "
             "illuminates the concept). Setup tag in brackets.", ""]
    for e in shortlist:
        sc = e.get("score", {})
        tag = _SETUP_LABEL.get(e.get("setup", ""), e.get("setup", ""))
        lines.append(f"## {e.get('title', '')}  ({tag})")
        lines.append(f"*Score {sc.get('combined', 0)} "
                     f"(feasibility {sc.get('feasibility', 0)}, "
                     f"illumination {sc.get('illumination', 0)}, "
                     f"{sc.get('votes', 0)} votes)*")
        lines.append("")
        lines.append(f"**Do this:** {e.get('steps', '')}")
        lines.append("")
        lines.append(f"**Shows:** {e.get('shows', '')}")
        if e.get("setup") == "python" and e.get("python"):
            lines.append("")
            lines.append("```python")
            lines.append(e["python"].strip())
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


def run_experiments(article_path: str | Path, out_dir: str | Path | None = None,
                    budget_usd: float = BUDGET_USD, top: int = 6) -> dict:
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
    shortlist = rank(ideas, scores, top) if ideas else []
    all_ranked = sorted(
        [{**idea, "score": scores.get(i, {})} for i, idea in enumerate(ideas)],
        key=lambda e: e["score"].get("combined", 0), reverse=True)

    payload = {
        "article": article.title,
        "article_path": str(article.path),
        "proposed_count": len(proposals),
        "merged_count": len(ideas),
        "shortlist": shortlist,
        "all_ranked": all_ranked,
        "cost": {"budget_usd": ledger.budget_usd, "total_usd": ledger.total_usd},
    }
    (out_dir / "experiments.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "experiments.md").write_text(
        render_markdown(article, shortlist))
    payload["out_dir"] = str(out_dir)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate try-it-yourself experiments for a blog article.")
    parser.add_argument("article", help="path to the .mdx/.md article")
    parser.add_argument("--out-dir", default=None,
                        help="artifact directory (default review-runs/<article-slug>)")
    parser.add_argument("--budget", type=float, default=BUDGET_USD)
    parser.add_argument("--top", type=int, default=6,
                        help="how many experiments in the shortlist (default 6)")
    args = parser.parse_args(argv)

    load_dotenv(Path(__file__).parent.parent / ".env")
    result = run_experiments(args.article, out_dir=args.out_dir,
                             budget_usd=args.budget, top=args.top)

    print(f"\n{'=' * 60}")
    print(f"Experiments: {result['article']}")
    print(f"{'=' * 60}")
    print(f"Proposed: {result['proposed_count']}  "
          f"merged: {result['merged_count']}  "
          f"shortlist: {len(result['shortlist'])}")
    print(f"Cost: ${result['cost']['total_usd']:.4f} "
          f"of ${result['cost']['budget_usd']:.2f}")
    print("\nShortlist:")
    for e in result["shortlist"]:
        sc = e.get("score", {})
        tag = _SETUP_LABEL.get(e.get("setup", ""), e.get("setup", ""))
        print(f"  [{sc.get('combined', 0):>4}] ({tag:>14}) {e.get('title', '')}")
    print(f"\nArtifacts in {result['out_dir']}/: experiments.md, experiments.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
