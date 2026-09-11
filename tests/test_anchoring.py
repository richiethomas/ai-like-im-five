"""M3 unit tests: quote anchoring against the canonical body. Offline."""

from reviewer.findings import _BodyIndex, anchor_findings
from reviewer.schemas import Anchor, AnchorKind, Finding

BODY = """A computer doesn't work that way. A computer can only manipulate numbers.

For a color image, it's actually three numbers per cell: one for red, one for green, one for blue (RGB). So an 800 × 600 image is really a 800 × 600 × 3 grid of numbers. That's **1.44 million numbers** to represent one photo.

When you scroll through your photos, your phone is actually shuffling around millions of numbers. “Viewpoint variation.” A dog viewed from the front looks different from a dog viewed from the side — the grid of numbers is wildly different."""


def qf(quote, model="m1", kind=AnchorKind.QUOTE.value, quote_b=None):
    return Finding(model=model, dimension="CORRECTNESS", issue="i", severity=5,
                   anchor=Anchor(kind=kind, quote=quote, quote_b=quote_b))


# --- _BodyIndex.find ---------------------------------------------------------

def test_exact_match():
    idx = _BodyIndex(BODY)
    span = idx.find("A computer can only manipulate numbers.")
    assert span is not None
    assert BODY[span[0]:span[1]] == "A computer can only manipulate numbers."


def test_markdown_bold_stripped():
    idx = _BodyIndex(BODY)
    # Model quotes without the ** markers
    span = idx.find("That's 1.44 million numbers to represent one photo.")
    assert span is not None
    assert "1.44 million numbers" in BODY[span[0]:span[1]]


def test_curly_vs_straight_quotes():
    idx = _BodyIndex(BODY)
    span = idx.find('"Viewpoint variation."')  # article has curly quotes
    assert span is not None
    assert "Viewpoint variation" in BODY[span[0]:span[1]]


def test_em_dash_vs_hyphen():
    idx = _BodyIndex(BODY)
    span = idx.find("from the side - the grid of numbers")  # article has em dash
    assert span is not None


def test_whitespace_collapse():
    idx = _BodyIndex(BODY)
    span = idx.find("A computer  doesn't\nwork that way.")
    assert span is not None
    assert BODY[span[0]:span[1]].startswith("A computer doesn't work")


def test_case_insensitive():
    idx = _BodyIndex(BODY)
    assert idx.find("a computer can ONLY manipulate numbers.") is not None


def test_ellipsis_fragments():
    idx = _BodyIndex(BODY)
    span = idx.find("For a color image... grid of numbers.")
    assert span is not None
    start, end = span
    assert BODY[start:].startswith("For a color image")
    assert BODY[:end].endswith("grid of numbers.")


def test_fuzzy_small_difference():
    idx = _BodyIndex(BODY)
    # "is actually shuffling" -> "is shuffling" (word dropped by the model)
    span = idx.find("your phone is shuffling around millions of numbers")
    assert span is not None
    assert "shuffling around millions" in BODY[span[0]:span[1]]


def test_absent_text_returns_none():
    idx = _BodyIndex(BODY)
    assert idx.find("gradient descent converges to a local minimum") is None


def test_short_garbage_not_fuzzy_matched():
    idx = _BodyIndex(BODY)
    assert idx.find("qzx") is None


# --- anchor_findings ---------------------------------------------------------

def test_anchor_resolves_and_reanchors():
    f = qf("That's 1.44 million numbers to represent one photo.")
    anchored, dropped = anchor_findings([f], BODY)
    assert len(anchored) == 1 and not dropped
    a = anchored[0].anchor
    assert a.start is not None and a.end is not None
    assert a.quote == BODY[a.start:a.end]      # re-anchored to exact article text
    assert "**1.44 million numbers**" in a.quote  # markdown restored from body


def test_hallucinated_quote_dropped():
    f = qf("The transformer architecture eliminates recurrence entirely.")
    anchored, dropped = anchor_findings([f], BODY)
    assert not anchored
    assert dropped[0].reason == "quote-not-found"
    assert dropped[0].model == "m1"


def test_quote_pair_both_resolve():
    f = qf("A computer can only manipulate numbers.",
           kind=AnchorKind.QUOTE_PAIR.value,
           quote_b="shuffling around millions of numbers")
    anchored, dropped = anchor_findings([f], BODY)
    assert len(anchored) == 1 and not dropped
    a = anchored[0].anchor
    assert a.kind == "quote_pair"
    assert a.start is not None and a.start_b is not None
    assert a.start_b > a.start


def test_quote_pair_degrades_to_quote_when_one_side_missing():
    f = qf("A computer can only manipulate numbers.",
           kind=AnchorKind.QUOTE_PAIR.value,
           quote_b="text that is not in the article at all whatsoever")
    anchored, dropped = anchor_findings([f], BODY)
    assert len(anchored) == 1 and not dropped
    a = anchored[0].anchor
    assert a.kind == "quote"
    assert a.quote_b is None


def test_quote_pair_both_missing_dropped():
    f = qf("nope not here", kind=AnchorKind.QUOTE_PAIR.value, quote_b="also absent entirely")
    anchored, dropped = anchor_findings([f], BODY)
    assert not anchored
    assert dropped[0].reason == "quote-pair-not-found"


def test_section_and_global_pass_through():
    fs = [
        Finding(model="m", dimension="COMPLETENESS", issue="missing caveat", severity=4,
                anchor=Anchor(kind=AnchorKind.SECTION.value, section="The intro")),
        Finding(model="m", dimension="CLICHÉS", issue="overuses filler", severity=3,
                anchor=Anchor(kind=AnchorKind.GLOBAL.value)),
    ]
    anchored, dropped = anchor_findings(fs, BODY)
    assert len(anchored) == 2 and not dropped
