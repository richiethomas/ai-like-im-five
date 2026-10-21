"""Figure check unit tests: extraction, rendering, merging, full run with fakes.
Offline (fake providers); the SVG render test needs rsvg-convert."""

import json
import shutil

import pytest

from reviewer.article import Article
from reviewer.figures import (
    Figure,
    extract_figures,
    merge_issues,
    render_png,
    run_figure_check,
)

BODY = """Intro paragraph.

## First section

Paragraph one before.

Paragraph two before.

<img src="/experiments/demo/a.svg" alt="A demo figure." style={{ width: "100%" }} />

Paragraph after.

## Second section

<img src="/experiments/demo/missing.svg" alt="Missing." />
"""

SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 20">'
       '<rect width="40" height="20" fill="#2563eb"/></svg>')


def article(path="x/index.mdx"):
    return Article(path=path, title="Demo", body=BODY, frontmatter={})


def test_extract_figures_reads_src_alt_and_context():
    figs = extract_figures(article())
    assert [f.src for f in figs] == ["/experiments/demo/a.svg",
                                     "/experiments/demo/missing.svg"]
    a = figs[0]
    assert a.index == 1
    assert a.alt == "A demo figure."
    assert a.heading == "First section"
    assert "Paragraph one before." in a.before and "Paragraph two before." in a.before
    assert a.after == "Paragraph after."
    assert figs[1].heading == "Second section"


def make_repo(tmp_path):
    (tmp_path / "public/experiments/demo").mkdir(parents=True)
    (tmp_path / "public/experiments/demo/a.svg").write_text(SVG)
    post = tmp_path / "src/content/posts/demo/index.mdx"
    post.parent.mkdir(parents=True)
    post.write_text("---\ntitle: Demo\n---\n" + BODY)
    return post


def test_missing_image_is_skipped(tmp_path):
    post = make_repo(tmp_path)
    fig = Figure(index=2, src="/experiments/demo/missing.svg", alt="", heading="",
                 before="", after="")
    render_png(fig, str(post), tmp_path / "out")
    assert fig.png == "" and "not found" in fig.skipped


@pytest.mark.skipif(shutil.which("rsvg-convert") is None, reason="needs rsvg-convert")
def test_svg_is_rendered(tmp_path):
    post = make_repo(tmp_path)
    fig = extract_figures(Article(str(post), "Demo", BODY, {}))[0]
    render_png(fig, str(post), tmp_path / "out")
    assert fig.skipped == ""
    assert open(fig.png, "rb").read(4) == b"\x89PNG"


def test_merge_marks_agreement_and_sorts():
    fig = Figure(index=1, src="s", alt="", heading="", before="", after="")
    fig.results = {
        "m1": [{"category": "LEGIBILITY", "issue": "clipped", "severity": 5,
                "suggested_fix": ""},
               {"category": "NUMBERS", "issue": "sum wrong", "severity": 8,
                "suggested_fix": ""}],
        "m2": [{"category": "LEGIBILITY", "issue": "label cut", "severity": 4,
                "suggested_fix": ""}],
        "m3": "RuntimeError: boom",
    }
    merged = merge_issues(fig)
    assert [m["severity"] for m in merged] == [8, 5, 4]
    counts = {(m["model"], m["category"]): m["category_models"] for m in merged}
    assert counts == {("m1", "NUMBERS"): 1, ("m1", "LEGIBILITY"): 2,
                      ("m2", "LEGIBILITY"): 2}
    from reviewer.figures import failed_models
    assert failed_models([fig]) == {"m3": "RuntimeError: boom"}


class FakeVision:
    def __init__(self, name, issues):
        self.name = name
        self.issues = issues
        self.saw_image = False

    def structured(self, prompt, schema, max_tokens, label="", images=None):
        self.saw_image = bool(images) and images[0][:4] == b"\x89PNG"
        return {"issues": self.issues}


@pytest.mark.skipif(shutil.which("rsvg-convert") is None, reason="needs rsvg-convert")
def test_full_run_with_fakes(tmp_path):
    post = make_repo(tmp_path)
    fakes = [FakeVision("m1", [{"category": "DIRECTION", "issue": "goes up",
                                "severity": 5, "suggested_fix": "flip"},
                               {"category": "ONE_POINT", "issue": "nitpick",
                                "severity": 2, "suggested_fix": ""}]),
             FakeVision("m2", [{"category": "DIRECTION", "issue": "bottom-up",
                                "severity": 4, "suggested_fix": ""}])]
    out = tmp_path / "out"
    summary = run_figure_check(post, out_dir=out, providers=fakes)
    assert summary["figures"] == 2 and summary["checked"] == 1
    assert summary["issues"] == 2  # severity-2 issue dropped
    assert summary["models"] == ["m1", "m2"] and summary["failed_models"] == {}
    assert all(f.saw_image for f in fakes)
    data = json.loads((out / "figures.json").read_text())
    assert data["figures"][1]["skipped"]
    md = (out / "figures.md").read_text()
    assert "sev 5 DIRECTION (m1; 2 models flagged DIRECTION)" in md and "Skipped:" in md


def test_reviewer_failure_warnings():
    from reviewer.cli import reviewer_failure_warnings
    from reviewer.schemas import Event
    events = [Event(seq=0, type="scan_failed", data={"model": "gemini-3.5-flash",
                                                     "error": "429 credits depleted"}),
              Event(seq=1, type="vote", data={"model": "llama-3.3-70b",
                                              "reason": "reviewer call failed after retries"}),
              Event(seq=2, type="vote", data={"model": "gpt-4o-mini", "reason": ""})]
    lines = reviewer_failure_warnings(events)
    assert any("gemini-3.5-flash scan failed" in l for l in lines)
    assert any("llama-3.3-70b failed to vote on 1" in l for l in lines)
    assert not any("gpt-4o-mini" in l for l in lines)
