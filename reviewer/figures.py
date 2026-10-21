"""Figure check: render each <img> in an article and have image-capable models
look at it alongside the text around it.

The text review only ever sees an <img> tag's src and alt text, so it cannot
notice clipped labels, a figure drawn in the wrong direction, numbers that
disagree with the text, or a figure that makes no clear point. This pass fills
that gap. It runs after the review (like the metaphor pass), is advisory, and
never affects the review's outcome.

Artifacts (in the run's out dir):
  figures/<n>-<name>.png  — the rendered images the models saw
  figures.json            — every figure, each model's issues, merged view
  figures.md              — readable summary, most severe first
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .article import Article, load_article
from .costs import CostLedger
from .providers import FIGURE_REVIEWER_NAMES, Provider, build_provider
from .schemas import BUDGET_USD, FIGURE_RESPONSE_SCHEMA

FIGURE_MAX_TOKENS = 2000
RENDER_WIDTH = 1200

IMG_RE = re.compile(r"<img\b[^>]*?/>", re.DOTALL)
ATTR_RE = r'{}="([^"]*)"'


@dataclass
class Figure:
    index: int            # 1-based position in the article
    src: str
    alt: str
    heading: str          # nearest preceding "## " heading ("" if none)
    before: str           # up to 2 paragraphs before the image
    after: str            # 1 paragraph after the image
    png: str = ""         # rendered PNG path ("" if rendering failed)
    skipped: str = ""     # reason the figure was not checked
    results: dict = field(default_factory=dict)  # model -> issues list | error str


def _attr(tag: str, name: str) -> str:
    m = re.search(ATTR_RE.format(name), tag)
    return m.group(1) if m else ""


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def extract_figures(article: Article) -> list[Figure]:
    body = article.body
    figures: list[Figure] = []
    for i, m in enumerate(IMG_RE.finditer(body), start=1):
        tag = m.group(0)
        head_before = body[:m.start()]
        headings = re.findall(r"^##\s+(.+)$", head_before, re.MULTILINE)
        before_paras = [p for p in _paragraphs(head_before)
                        if not p.startswith("<img")][-2:]
        after_paras = [p for p in _paragraphs(body[m.end():])
                       if not p.startswith("<img")][:1]
        figures.append(Figure(
            index=i, src=_attr(tag, "src"), alt=_attr(tag, "alt"),
            heading=headings[-1].strip() if headings else "",
            before="\n\n".join(before_paras), after="\n\n".join(after_paras)))
    return figures


def _public_dir(article_path: str) -> Path:
    # Posts live at <repo>/src/content/posts/<slug>/index.mdx; images are
    # served from <repo>/public/.
    for parent in Path(article_path).resolve().parents:
        if (parent / "public").is_dir():
            return parent / "public"
    return Path("public").resolve()


def render_png(fig: Figure, article_path: str, out_dir: Path) -> None:
    """Fill fig.png, or fig.skipped with the reason it could not be rendered."""
    src_path = _public_dir(article_path) / fig.src.lstrip("/")
    if not src_path.is_file():
        fig.skipped = f"image file not found: {src_path}"
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dest = fig_dir / f"{fig.index:02d}-{src_path.stem}.png"
    suffix = src_path.suffix.lower()
    if suffix == ".png":
        shutil.copyfile(src_path, dest)
    elif suffix == ".svg":
        if shutil.which("rsvg-convert") is None:
            fig.skipped = "rsvg-convert is not installed"
            return
        proc = subprocess.run(
            ["rsvg-convert", "-w", str(RENDER_WIDTH), str(src_path), "-o", str(dest)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            fig.skipped = f"rsvg-convert failed: {proc.stderr.strip()[:200]}"
            return
    else:
        fig.skipped = f"unsupported image type: {suffix}"
        return
    fig.png = str(dest)


def figure_prompt(article: Article, fig: Figure) -> str:
    return f"""You are reviewing one figure from a blog article written for non-technical readers. The image is attached. Below is the text around it in the article, and the figure's alt text.

Article title: {article.title}
Section: {fig.heading or "(none)"}

Text before the figure:
---
{fig.before or "(none)"}
---

Text after the figure:
---
{fig.after or "(none)"}
---

Alt text:
---
{fig.alt or "(none)"}
---

Look at the image itself and report problems in these categories:
- MATCHES_TEXT: the figure does not show what the surrounding text says it shows.
- NUMBERS: a number, label, or arithmetic in the figure is wrong, or disagrees with the text.
- LEGIBILITY: text is clipped, cut off at an edge, overlapping other text or shapes, or too small to read.
- ONE_POINT: the figure does not make one clear point, or includes elements unrelated to it.
- DIRECTION: the figure reads in an unexpected direction (this blog draws flows top to bottom or left to right; inputs first, outputs last).
- ALT_TEXT: the alt text describes something different from what is drawn.

Rules:
- Report only problems you can see in this image. Do not restate what the text already explains, and do not suggest adding new figures.
- Check every number and sum shown in the figure.
- Severity: 9-10 wrong or misleading; 6-8 likely to confuse a reader; 3-5 a meaningful improvement; do not report 1-2.
- If the figure has no problems, return an empty list.

Return JSON: {{"issues": [{{"category": "LEGIBILITY", "issue": "<what is wrong, max 40 words>", "severity": 5, "suggested_fix": "<concrete fix, max 40 words>"}}]}}"""


def _check_one(provider: Provider, article: Article, fig: Figure):
    image = Path(fig.png).read_bytes()
    prompt = figure_prompt(article, fig)
    try:
        data = provider.structured(prompt, FIGURE_RESPONSE_SCHEMA, FIGURE_MAX_TOKENS,
                                   label="figure", images=[image])
    except Exception:  # noqa: BLE001 — one retry for transient API failures
        data = provider.structured(prompt, FIGURE_RESPONSE_SCHEMA, FIGURE_MAX_TOKENS,
                                   label="figure:retry", images=[image])
    issues = data.get("issues", []) if isinstance(data, dict) else []
    clean = []
    for it in issues:
        if not isinstance(it, dict):
            continue
        try:
            sev = int(it.get("severity", 0))
        except (TypeError, ValueError):
            continue
        if sev < 3:
            continue
        clean.append({"category": str(it.get("category", "")),
                      "issue": str(it.get("issue", "")),
                      "severity": sev,
                      "suggested_fix": str(it.get("suggested_fix", ""))})
    return clean


def merge_issues(fig: Figure) -> list[dict]:
    """Flatten every model's issues for one figure, most severe first.

    Each issue records how many models flagged its category on this figure
    ("category_models"). That is a hint about where to look, not agreement:
    two models can flag the same category for unrelated (or wrong) reasons."""
    by_cat: dict[str, set] = {}
    flat: list[dict] = []
    for model, res in fig.results.items():
        if not isinstance(res, list):
            continue
        for it in res:
            flat.append({**it, "model": model})
            by_cat.setdefault(it["category"], set()).add(model)
    for it in flat:
        it["category_models"] = len(by_cat.get(it["category"], ()))
    flat.sort(key=lambda it: -it["severity"])
    return flat


def failed_models(figures: list[Figure]) -> dict[str, str]:
    """model -> first error message, for every model that failed on any figure."""
    out: dict[str, str] = {}
    for fig in figures:
        for model, res in fig.results.items():
            if not isinstance(res, list) and model not in out:
                out[model] = str(res)
    return out


def render_markdown(article: Article, figures: list[Figure], cost: float) -> str:
    lines = [f"# Figure check: {article.title}", "",
             f"{len(figures)} figure(s). Cost ${cost:.4f}.", ""]
    failed = failed_models(figures)
    if failed:
        lines.append("**WARNING: these models failed and did not check the figures:**")
        for m, err in failed.items():
            lines.append(f"- {m}: {err[:200]}")
        lines.append("")
    for fig in figures:
        lines.append(f"## Figure {fig.index}: `{fig.src}`")
        if fig.heading:
            lines.append(f"Section: {fig.heading}")
        if fig.png:
            lines.append(f"Rendered: `{fig.png}`")
        if fig.skipped:
            lines += [f"Skipped: {fig.skipped}", ""]
            continue
        errors = {m: r for m, r in fig.results.items() if not isinstance(r, list)}
        merged = merge_issues(fig)
        if not merged:
            lines.append("No issues reported.")
        for it in merged:
            also = (f"; {it['category_models']} models flagged {it['category']}"
                    if it["category_models"] > 1 else "")
            lines.append(f"- sev {it['severity']} {it['category']} "
                         f"({it['model']}{also}): {it['issue']}")
            if it["suggested_fix"]:
                lines.append(f"  - Fix: {it['suggested_fix']}")
        for m, err in errors.items():
            lines.append(f"- ({m} failed: {err})")
        lines.append("")
    return "\n".join(lines)


def run_figure_check(article_path: str | Path, out_dir: str | Path | None = None,
                     budget_usd: float = BUDGET_USD,
                     names: list[str] | None = None,
                     providers: list[Provider] | None = None) -> dict:
    """Run the figure check. `providers` overrides `names` (used by tests)."""
    article = load_article(article_path)
    if out_dir is None:
        out_dir = Path("review-runs") / Path(article_path).parent.name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = CostLedger(budget_usd=budget_usd)
    if providers is None:
        providers = [build_provider(n, ledger) for n in (names or FIGURE_REVIEWER_NAMES)]

    figures = extract_figures(article)
    for fig in figures:
        render_png(fig, str(article_path), out_dir)

    jobs = [(p, fig) for fig in figures if fig.png for p in providers]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_check_one, p, article, fig): (p.name, fig)
                   for p, fig in jobs}
        for fut in as_completed(futures):
            name, fig = futures[fut]
            try:
                fig.results[name] = fut.result()
            except Exception as e:  # noqa: BLE001 — per-model isolation
                fig.results[name] = f"{type(e).__name__}: {e}"

    payload = {
        "article": article.title,
        "failed_models": failed_models(figures),
        "article_path": str(article_path),
        "figures": [{
            "index": f.index, "src": f.src, "alt": f.alt, "heading": f.heading,
            "png": f.png, "skipped": f.skipped,
            "results": f.results,
            "merged": merge_issues(f) if not f.skipped else [],
        } for f in figures],
        "cost": {"budget_usd": ledger.budget_usd, "total_usd": ledger.total_usd},
    }
    (out_dir / "figures.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "figures.md").write_text(render_markdown(article, figures, ledger.total_usd))

    issues = [it for f in payload["figures"] for it in f["merged"]]
    return {
        "out_dir": str(out_dir),
        "figures": len(figures),
        "checked": sum(1 for f in figures if f.png),
        "issues": len(issues),
        "models": [p.name for p in providers],
        "failed_models": failed_models(figures),
        "cost_usd": ledger.total_usd,
    }


def print_summary(summary: dict) -> None:
    print(f"\nFigure check: {summary['checked']} of {summary['figures']} figure(s) "
          f"checked by {', '.join(summary['models'])}: {summary['issues']} issue(s), "
          f"${summary['cost_usd']:.4f}")
    for model, err in summary["failed_models"].items():
        print(f"  WARNING: {model} failed and did not check the figures: {err[:160]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Figure check for one article.")
    parser.add_argument("article")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--budget", type=float, default=BUDGET_USD)
    args = parser.parse_args(argv)
    summary = run_figure_check(args.article, out_dir=args.out_dir,
                               budget_usd=args.budget)
    print_summary(summary)
    print(f"Artifacts in {summary['out_dir']}/: figures.md, figures.json, figures/")
    return 0
