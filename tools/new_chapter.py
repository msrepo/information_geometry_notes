#!/usr/bin/env python3
"""Scaffold chapters/chNN-<slug>/notes.md for a chapter of the book.

Fills the front matter from tools/bookmap.json and lays out the chapter's section outline as headings to
fill in. The outline is only the book's own section titles; the notes themselves are yours.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOK_MAP = json.loads((ROOT / "tools" / "bookmap.json").read_text(encoding="utf-8"))

TEMPLATE = """---
title: "{title}"
short_title: "Ch. {n} — {short}"
chapter: {n}
category: "{part}"
book_pages: "TODO"
url: "{url}"
tags: [todo]
status: draft
---

## Links

- The book: [{book}]({url}), Chapter {n}, starting on p. {page}. Equation numbers such as (1.69) refer to the book.
- `code/` holds the runnable checks (`make verify`); `figures/` the themed SVGs and any interactive page.

## In one paragraph

TODO: what the chapter builds, in plain language, before any formulas.

## The spine of the argument

1. TODO

## Setup and notation

| Symbol | Meaning |
|---|---|
| $x$ | TODO |

{sections}
## Questions and doubts

- TODO: what is still unclear, what the book glosses over, and what would settle it.

## Takeaways

- TODO.

---

*Notes started {today}.*
"""


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit("usage: make new CH=<chapter number> SLUG=<short-topic>")
    n, slug = int(sys.argv[1]), re.sub(r"[^a-z0-9-]+", "-", sys.argv[2].strip().lower()).strip("-")
    chapters = {c["n"]: c for c in BOOK_MAP["chapters"]}
    if n not in chapters:
        sys.exit(f"no chapter {n}; the book has chapters 1..{max(chapters)}")
    c = chapters[n]
    part = next(p["key"] for p in BOOK_MAP["parts"] if n in p["chapters"])
    target = ROOT / "chapters" / f"ch{n:02d}-{slug}"
    notes = target / "notes.md"
    if notes.exists():
        sys.exit(f"{notes.relative_to(ROOT)} already exists; nothing written.")
    sections = "".join(f"## {s['num']} {s['title']}\n\nTODO\n\n" for s in c["sections"])
    target.mkdir(parents=True, exist_ok=True)
    notes.write_text(TEMPLATE.format(
        title=f"Chapter {n}: {c['title']}", short=slug.replace("-", " "), n=n, part=part, page=c["book_page"],
        url=BOOK_MAP["book"]["url"], book=BOOK_MAP["book"]["title"], sections=sections, today=date.today().isoformat()), encoding="utf-8")
    print(f"Created {notes.relative_to(ROOT)}")
    print("Next: fill in book_pages and the notes, then `make` (or `make serve`).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
