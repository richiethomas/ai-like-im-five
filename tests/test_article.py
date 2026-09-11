"""M1 unit tests: article loading edge cases."""

import pytest

from reviewer.article import ArticleParseError, load_article


def write(tmp_path, content):
    p = tmp_path / "index.mdx"
    p.write_text(content, encoding="utf-8")
    return p


def test_standard_frontmatter(tmp_path):
    p = write(tmp_path, '---\ntitle: "Hello World"\ncategory: "Vision"\n---\n\nBody text here.\n')
    a = load_article(p)
    assert a.title == "Hello World"
    assert a.body == "Body text here."
    assert a.frontmatter["category"] == "Vision"


def test_apostrophe_in_quoted_title(tmp_path):
    p = write(tmp_path, "---\ntitle: \"What Does It Mean for a Computer to 'See' a Picture?\"\n---\nBody.\n")
    a = load_article(p)
    assert a.title == "What Does It Mean for a Computer to 'See' a Picture?"


def test_unquoted_title(tmp_path):
    p = write(tmp_path, "---\ntitle: Plain Unquoted Title\n---\nBody.\n")
    a = load_article(p)
    assert a.title == "Plain Unquoted Title"


def test_hrule_in_body_not_treated_as_fence(tmp_path):
    p = write(tmp_path, '---\ntitle: "T"\n---\nFirst part.\n\n---\n\nSecond part after hrule.\n')
    a = load_article(p)
    assert "---" in a.body
    assert "Second part after hrule." in a.body
    assert a.body.startswith("First part.")


def test_nested_frontmatter_keys_skipped(tmp_path):
    p = write(tmp_path, '---\ntitle: "T"\nauthor:\n  name: "Richie"\n  role: "Engineer"\ntags: ["a", "b"]\n---\nBody.\n')
    a = load_article(p)
    assert a.title == "T"
    assert "name" not in a.frontmatter  # indented lines don't leak to top level
    assert a.frontmatter["tags"] == '["a", "b"]'


def test_body_is_exact_text(tmp_path):
    body = "Line one.\n\n## Heading\n\nLine with **bold** and `code`."
    p = write(tmp_path, f'---\ntitle: "T"\n---\n{body}\n')
    a = load_article(p)
    assert a.body == body  # anchors index into exactly this string


def test_missing_frontmatter_raises(tmp_path):
    p = write(tmp_path, "Just a body, no fences.\n")
    with pytest.raises(ArticleParseError, match="no frontmatter fence"):
        load_article(p)


def test_unclosed_fence_raises(tmp_path):
    p = write(tmp_path, '---\ntitle: "T"\nBody without closing fence.\n')
    with pytest.raises(ArticleParseError, match="never closes"):
        load_article(p)


def test_missing_title_raises(tmp_path):
    p = write(tmp_path, "---\ncategory: X\n---\nBody.\n")
    with pytest.raises(ArticleParseError, match="no title"):
        load_article(p)


def test_empty_body_raises(tmp_path):
    p = write(tmp_path, '---\ntitle: "T"\n---\n\n')
    with pytest.raises(ArticleParseError, match="empty body"):
        load_article(p)


def test_real_article_1():
    """The actual article this system will run against first."""
    a = load_article(
        "src/content/posts/paper-1-part-1-what-does-it-mean-for-a-computer-to-see-a-picture/index.mdx"
    )
    assert a.title.startswith("Paper #1, Part 1")
    assert "'See'" in a.title  # apostrophes survived
    assert len(a.body) > 3000
    assert not a.body.startswith("---")
