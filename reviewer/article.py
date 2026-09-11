"""Article loading: frontmatter parsing and the canonical body text.

The `body` returned here is the single source of truth for "the text every
model saw." Quote anchors resolve to offsets into this exact string, and the
full body (never truncated) is what goes into every prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


class ArticleParseError(Exception):
    pass


@dataclass
class Article:
    path: str
    title: str
    body: str          # canonical text — anchors are offsets into this string
    frontmatter: dict  # shallow key -> raw string value


def _parse_frontmatter_block(block: str) -> dict:
    """Line-based YAML-lite parse: top-level `key: value` pairs only.

    Handles quoted values (single/double) including apostrophes inside double
    quotes, and unquoted values. Nested structures (e.g. `author:` with indented
    children) keep their raw remainder as the value ('' for pure parents).
    """
    out: dict = {}
    for line in block.splitlines():
        if not line or line[0] in (" ", "\t", "#"):
            continue  # skip nested/indented lines, blanks, comments
        m = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2).strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
            raw = raw[1:-1]
        out[key] = raw
    return out


def load_article(path: str | Path) -> Article:
    """Parse an .mdx/.md file with `---` frontmatter fences.

    Only the FIRST fence pair delimits frontmatter; any later `---` lines
    (markdown hrules) belong to the body. Robust to apostrophes in quoted
    titles and to unquoted frontmatter values.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    if not lines or lines[0].strip() != "---":
        raise ArticleParseError(f"{path}: no frontmatter fence on line 1")

    close_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close_idx = i
            break
    if close_idx is None:
        raise ArticleParseError(f"{path}: frontmatter fence never closes")

    frontmatter = _parse_frontmatter_block("\n".join(lines[1:close_idx]))
    body = "\n".join(lines[close_idx + 1:]).strip()
    if not body:
        raise ArticleParseError(f"{path}: empty body")

    title = frontmatter.get("title", "").strip()
    if not title:
        raise ArticleParseError(f"{path}: frontmatter has no title")

    return Article(path=str(path), title=title, body=body, frontmatter=frontmatter)
