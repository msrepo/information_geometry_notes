#!/usr/bin/env python3
"""Render every chapters/<slug>/notes.md (and foundations/<slug>/notes.md) to build/<slug>/index.html.

The index page doubles as a map of the whole book: every chapter, grouped by Part, linking to its
notes when they exist. The map comes from tools/bookmap.json (titles and page ranges only).

No third-party dependencies: the YAML front matter is restricted to a small flat subset (scalars and
inline lists) that is parsed here directly. pandoc does the Markdown -> HTML conversion.

The book itself is under copyright and is never part of this repository or of the built site: nothing
here copies a PDF, and the build fails if a PDF ends up in build/.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Pin the KaTeX CDN explicitly. Bare `--katex` uses whatever the local pandoc build was configured with,
# and Debian's package points at the filesystem path /usr/share/javascript/katex/, which 404s once the
# site is served from GitHub Pages.
KATEX_CDN = "https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/"

SITE_TITLE = "Information Geometry Notes"
SITE_URL = "https://msrepo.github.io/information_geometry_notes/"

BOOK_MAP = json.loads((ROOT / "tools" / "bookmap.json").read_text(encoding="utf-8"))
BOOK = BOOK_MAP["book"]
CHAPTERS = {c["n"]: c for c in BOOK_MAP["chapters"]}
PARTS = BOOK_MAP["parts"]

# Roots holding notes. Same front matter and pipeline everywhere; separate roots only so the index can
# group them. Everything renders into a flat build/<slug>/, so slugs must be unique across roots and
# cross-links are always ../<slug>/index.html.
COLLECTIONS = [
    (ROOT / "chapters", "chapter"),          # one folder per chapter of the book
    (ROOT / "foundations", "foundations"),   # background maths that several chapters lean on
]

# Sidebar and index order. A page picks its group with `category:` in the front matter: "Part I" ...
# "Part IV" for chapters, "Foundations" for background pages. Anything else lands under "Unsorted".
FOUNDATIONS = "Foundations"
UNSORTED = "Unsorted"
CATEGORY_ORDER = [p["key"] for p in PARTS] + [FOUNDATIONS, UNSORTED]
CATEGORY_TITLE = {p["key"]: f'{p["key"]}: {p["title"]}' for p in PARTS}
CATEGORY_TITLE[FOUNDATIONS] = "Foundations"

BUILD = ROOT / "build"
STYLE = ROOT / "tools" / "style.css"


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Split a leading `---` fenced block off the document.

    Supports `key: value`, `key: "value"` and `key: [a, b, c]`. Anything more elaborate belongs in the
    body, not the metadata.
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    end = next((i for i, l in enumerate(lines[1:], start=1) if l.strip() == "---"), None)
    if end is None:
        return {}, text
    meta: dict = {}
    for line in lines[1:end]:
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [v.strip().strip("\"'") for v in value[1:-1].split(",")]
            meta[key] = [v for v in items if v]
        else:
            meta[key] = value.strip("\"'")
    return meta, "\n".join(lines[end + 1:])


def discover() -> list[tuple[Path, dict]]:
    found = []
    for root, kind in COLLECTIONS:
        for notes in sorted(root.glob("*/notes.md")):
            meta, _ = parse_front_matter(notes.read_text(encoding="utf-8"))
            meta.setdefault("title", notes.parent.name)
            meta["slug"] = notes.parent.name
            meta["kind"] = kind
            found.append((notes, meta))
    slugs = [m["slug"] for _, m in found]
    dupes = {s for s in slugs if slugs.count(s) > 1}
    if dupes:
        sys.exit(f"duplicate slug(s) across roots: {', '.join(sorted(dupes))}")
    return found


def chapter_number(meta: dict) -> int | None:
    v = str(meta.get("chapter", "")).strip()
    return int(v) if v.isdigit() else None


def meta_line(meta: dict) -> str:
    bits = []
    n = chapter_number(meta)
    if n is not None:
        bits.append(f"Chapter {n}")
        if meta.get("book_pages"):
            bits.append(f'book pp. {html.escape(str(meta["book_pages"]))}')
        bits.append(f'{html.escape(BOOK["author"])}, <i>{html.escape(BOOK["title"])}</i> ({BOOK["year"]})')
    else:
        bits += [html.escape(str(meta[f])) for f in ("authors", "venue") if meta.get(f)]
    return " &middot; ".join(bits)


def tag_html(meta: dict) -> str:
    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    return "".join(f'<span class="tag">{html.escape(t)}</span>' for t in tags)


def render_one(notes: Path, meta: dict, entries: list) -> None:
    slug = meta["slug"]
    out_dir = BUILD / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-chapter figures are published beside the page so notes.md can reference them at figures/<name>.
    # Themed SVGs and the self-contained interactive pages live here. PDFs never do.
    figures = notes.parent / "figures"
    if figures.is_dir():
        shutil.copytree(figures, out_dir / "figures", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.pdf", "*.djvu", ".DS_Store"))

    links = []
    if meta.get("url"):
        links.append(f'<a href="{html.escape(meta["url"])}">the book</a>')
    links.append('<a href="../index.html">all chapters</a>')

    header = (
        '<header class="paper-head">'
        f'<h1>{html.escape(str(meta["title"]))}</h1>'
        f'<p class="meta">{meta_line(meta)}</p>'
        f'<p class="tags">{tag_html(meta)}</p>'
        f'<p class="links">{" &middot; ".join(links)}</p>'
        "</header>"
    )
    # before-body opens the two-column layout and emits the shared nav; after-body closes it. Pandoc drops
    # the TOC and content in between, so they land inside <main>.
    head_file = out_dir / ".head.html"
    head_file.write_text(HEAD_SCRIPT, encoding="utf-8")
    header_file = out_dir / ".header.html"
    header_file.write_text(
        TOGGLE_BUTTON + '<div class="layout">'
        + sidebar_html(entries, slug, depth=1) + "<main>" + header, encoding="utf-8")
    after_file = out_dir / ".after.html"
    after_file.write_text("</main></div>" + TOGGLE_SCRIPT, encoding="utf-8")

    # Feed pandoc the body only. Left in place, the front matter's `title` would make pandoc emit its own
    # title block on top of the header above.
    _, body = parse_front_matter(notes.read_text(encoding="utf-8"))
    body_file = out_dir / ".body.md"
    body_file.write_text(body, encoding="utf-8")

    cmd = [
        "pandoc", str(body_file),
        "--from", "markdown+tex_math_dollars+tex_math_single_backslash",
        "--to", "html5",
        "--standalone",
        f"--katex={KATEX_CDN}",
        "--toc", "--toc-depth=2",
        "--css", "../style.css",
        # pagetitle sets <title> without emitting pandoc's own title block, which would duplicate the
        # header injected above.
        "--metadata", f"pagetitle={meta['title']} · {SITE_TITLE}",
        "--include-in-header", str(head_file),
        "--include-before-body", str(header_file),
        "--include-after-body", str(after_file),
        "--output", str(out_dir / "index.html"),
    ]
    subprocess.run(cmd, check=True)
    for f in (head_file, header_file, after_file, body_file):
        f.unlink()
    print(f"  built  {slug}")


def sort_key(m: dict) -> tuple:
    n = chapter_number(m)
    return (n if n is not None else 10 ** 6, str(m["title"]).lower())


def _grouped(entries: list[tuple[Path, dict]]) -> list[tuple[str, list[dict]]]:
    """[(category, [meta, ...]), ...] in display order; pages sorted by chapter number, then title."""
    by_cat: dict[str, list[dict]] = {}
    for _, m in entries:
        cat = m.get("category") or UNSORTED
        by_cat.setdefault(cat if cat in CATEGORY_ORDER else UNSORTED, []).append(m)
    return [(c, sorted(by_cat[c], key=sort_key)) for c in CATEGORY_ORDER if c in by_cat]


BURGER = ('<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">'
          '<path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor" stroke-width="1.5" '
          'stroke-linecap="round" fill="none"/></svg>')

# Applied before first paint so a collapsed sidebar never flashes open. Defaults to collapsed on narrow
# viewports, expanded otherwise, then remembers the choice. Wrapped in try/catch because localStorage
# throws in private mode.
HEAD_SCRIPT = """<script>
(function(){try{
  var v=localStorage.getItem('sidebar-collapsed');
  if(v===null)v=window.matchMedia('(max-width: 66rem)').matches?'1':'0';
  if(v==='1')document.documentElement.classList.add('nav-collapsed');
}catch(e){}})();
</script>"""

TOGGLE_SCRIPT = """<script>
(function(){
  var b=document.querySelector('.nav-toggle');if(!b)return;
  function sync(){b.setAttribute('aria-expanded',
    String(!document.documentElement.classList.contains('nav-collapsed')));}
  sync();
  b.addEventListener('click',function(){
    var c=document.documentElement.classList.toggle('nav-collapsed');
    try{localStorage.setItem('sidebar-collapsed',c?'1':'0');}catch(e){}
    sync();
  });
})();
</script>"""

TOGGLE_BUTTON = ('<button class="nav-toggle" type="button" aria-controls="site-nav" '
                 f'aria-expanded="true" aria-label="Show or hide the contents sidebar">{BURGER}</button>')


def sidebar_html(entries: list[tuple[Path, dict]], current: str | None, depth: int) -> str:
    """Nav shared by every page. `depth` is how many levels up the site root is."""
    up = "../" * depth
    parts = ['<nav class="sidebar" id="site-nav">',
             f'<a class="nav-home" href="{up}index.html">{html.escape(SITE_TITLE)}</a>']
    for cat, metas in _grouped(entries):
        parts.append(f'<div class="nav-cat">{html.escape(cat)}</div>')
        parts.append("<ul>")
        for m in metas:
            cur = ' aria-current="page"' if m["slug"] == current else ""
            short = m.get("short_title") or m["title"]
            parts.append(f'<li><a href="{up}{html.escape(m["slug"])}/index.html"{cur}>'
                         f'{html.escape(str(short))}</a></li>')
        parts.append("</ul>")
    parts.append("</nav>")
    return "".join(parts)


def _anchor(*parts: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", "-".join(p for p in parts if p).lower()).strip("-")


def _chapter_row(n: int, notes_meta: dict | None) -> str:
    c = CHAPTERS[n]
    title = html.escape(c["title"])
    if notes_meta:
        status = notes_meta.get("status", "")
        badge = (f'<span class="status status-{html.escape(status)}">{html.escape(status)}</span>' if status else "")
        head = (f'<a class="title" href="{html.escape(notes_meta["slug"])}/index.html">'
                f'Chapter {n}. {title}</a>{badge}')
        tags = f'<div class="tags">{tag_html(notes_meta)}</div>'
    else:
        head = f'<span class="title todo">Chapter {n}. {title}</span><span class="status status-todo">notes to come</span>'
        tags = ""
    secs = "".join(f'<li><span class="sec-num">{html.escape(s["num"])}</span> {html.escape(s["title"])}</li>' for s in c["sections"])
    pages = f'book p. {c["book_page"]} &middot; PDF pp. {c["pdf_pages"][0]}&ndash;{c["pdf_pages"][1]}'
    return (f'<li class="chapter-row">{head}<div class="meta">{pages}</div>{tags}'
            f'<details><summary>sections</summary><ul class="secs">{secs}</ul></details></li>')


def render_index(entries: list[tuple[Path, dict]]) -> None:
    by_chapter = {chapter_number(m): m for _, m in entries if m.get("kind") == "chapter" and chapter_number(m)}
    rows = []
    for part in PARTS:
        rows.append(f'<h2 id="{_anchor(part["key"])}">{html.escape(part["key"])}: {html.escape(part["title"])}</h2>')
        rows.append("<ul class='papers'>")
        rows += [_chapter_row(n, by_chapter.get(n)) for n in part["chapters"]]
        rows.append("</ul>")
    found = [m for _, m in entries if m.get("kind") == "foundations"]
    if found:
        rows.append(f'<h2 id="{_anchor(FOUNDATIONS)}">Foundations</h2><ul class="papers">')
        for m in sorted(found, key=sort_key):
            rows.append(f'<li><a class="title" href="{html.escape(m["slug"])}/index.html">{html.escape(str(m["title"]))}</a>'
                        f'<div class="meta">{meta_line(m)}</div><div class="tags">{tag_html(m)}</div></li>')
        rows.append("</ul>")
    n_done = len(by_chapter)
    intro = (f'Study notes on <a href="{html.escape(BOOK["url"])}">{html.escape(BOOK["author"])}, '
             f'<i>{html.escape(BOOK["title"])}</i></a> ({html.escape(BOOK["publisher"])}, {BOOK["year"]}): the fundamentals worked '
             'out from scratch, with checks you can run and visualisations you can poke at. Heavy use of AI went into these notes: '
             'they reflect my effort to understand this book using Claude as a tutor. The book itself is not included here.')
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(SITE_TITLE)}</title>
<link rel="stylesheet" href="style.css">
{HEAD_SCRIPT}</head>
<body>{TOGGLE_BUTTON}<div class="layout">
{sidebar_html(entries, None, depth=0)}
<main>
<header class="paper-head"><h1>{html.escape(SITE_TITLE)}</h1>
<p class="meta">{n_done} of {len(CHAPTERS)} chapters annotated &middot; {len(found)} background page(s)</p>
<p>{intro}</p></header>
{"".join(rows)}
</main></div>{TOGGLE_SCRIPT}</body></html>
"""
    (BUILD / "index.html").write_text(doc, encoding="utf-8")
    print(f"  built  index.html ({len(entries)} page(s))")


def main() -> int:
    if shutil.which("pandoc") is None:
        sys.exit("pandoc not found. Install it: brew install pandoc")
    entries = discover()
    if not entries:
        sys.exit("No <slug>/notes.md found under chapters/ or foundations/. Scaffold one with: make new CH=1 SLUG=dually-flat")
    if BUILD.exists():
        shutil.rmtree(BUILD)                      # never publish stale pages from an earlier build
    BUILD.mkdir()
    shutil.copy2(STYLE, BUILD / "style.css")
    for notes, meta in entries:
        render_one(notes, meta, entries)
    render_index(entries)
    leaked = [p for p in BUILD.rglob("*") if p.suffix.lower() in (".pdf", ".djvu")]
    if leaked:                                    # belt and braces: the book must never reach the published site
        sys.exit("refusing to publish: found " + ", ".join(str(p.relative_to(BUILD)) for p in leaked))
    print(f"\nOpen {BUILD / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
