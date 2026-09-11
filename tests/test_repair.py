"""M2 unit tests: the finding repair layer. Offline."""

from reviewer.findings import repair_findings


def raw(**kw):
    base = {"anchor_type": "quote", "quote": "some text", "dimension": "CORRECTNESS",
            "issue": "an issue", "severity": 5}
    base.update(kw)
    return base


def test_clean_finding_passes():
    findings, dropped = repair_findings([raw(suggested_fix="do X")], "m1")
    assert len(findings) == 1 and not dropped
    f = findings[0]
    assert f.model == "m1"
    assert f.severity == 5
    assert f.anchor.kind == "quote"
    assert f.suggested_fix == "do X"


def test_severity_coercions():
    findings, dropped = repair_findings([
        raw(severity="7/10"),      # string with number
        raw(severity="high"),      # no number -> dropped
        raw(severity=15),          # clamped to 10
        raw(severity=0),           # clamped to 1
        raw(severity=6.7),         # float -> int
        raw(severity=True),        # bool -> dropped
    ], "m1")
    assert [f.severity for f in findings] == [7, 10, 1, 6]
    assert [d.reason for d in dropped] == ["bad-severity", "bad-severity"]


def test_dimension_aliases_and_invalid():
    findings, dropped = repair_findings([
        raw(dimension="cliches"),       # alias, wrong case -> CLICHÉS
        raw(dimension="CLICHÉS"),       # exact
        raw(dimension="correctness"),   # case fix
        raw(dimension="VIBES"),         # invalid -> dropped
    ], "m1")
    assert [f.dimension for f in findings] == ["CLICHÉS", "CLICHÉS", "CORRECTNESS"]
    assert dropped[0].reason == "bad-dimension"


def test_missing_issue_dropped():
    findings, dropped = repair_findings([raw(issue=""), raw(issue=None), {"severity": 5}], "m1")
    assert not findings
    assert all(d.reason == "missing-issue" for d in dropped)


def test_anchor_kind_inference_when_missing():
    findings, _ = repair_findings([
        raw(anchor_type=None),                                  # has quote -> quote
        raw(anchor_type=None, quote=None, section="Intro"),     # -> section
        raw(anchor_type=None, quote=None),                      # -> global
        raw(anchor_type="banana", quote="q", quote_b="q2"),     # both quotes -> quote_pair
    ], "m1")
    assert [f.anchor.kind for f in findings] == ["quote", "section", "global", "quote_pair"]


def test_quote_anchor_without_quote_degrades():
    findings, _ = repair_findings([
        raw(anchor_type="quote", quote="", section="The intro"),
        raw(anchor_type="quote", quote=None, section=None),
        raw(anchor_type="quote_pair", quote="only one", quote_b=None),
    ], "m1")
    assert [f.anchor.kind for f in findings] == ["section", "global", "quote"]


def test_legacy_fix_field_accepted():
    findings, _ = repair_findings([raw(fix="legacy fix key")], "m1")
    assert findings[0].suggested_fix == "legacy fix key"


def test_not_a_list():
    findings, dropped = repair_findings("garbage", "m1")
    assert not findings
    assert dropped[0].reason == "findings-not-a-list"


def test_non_dict_items_dropped():
    findings, dropped = repair_findings([raw(), "junk", 42], "m1")
    assert len(findings) == 1
    assert [d.reason for d in dropped] == ["item-not-an-object"] * 2
