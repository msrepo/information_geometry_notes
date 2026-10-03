# Information Geometry Notes

**Rendered notes: <https://msrepo.github.io/information_geometry_notes/>**

Study notes on Shun-ichi Amari, *Information Geometry and Its Applications* (Springer, 2016,
[DOI 10.1007/978-4-431-55978-8](https://doi.org/10.1007/978-4-431-55978-8)): the fundamentals worked out from
scratch, one folder per chapter, with runnable checks of every claim and visualisations to poke at.

Heavy use of AI went into these notes: they reflect my effort to understand this book using Claude as a tutor.
**The book itself is not in this repository and is never published**: it is under copyright, so it lives in a gitignored folder on the author's machine.

## Layout

```
.
├── Makefile                 build entry point
├── chapters/
│   └── ch01-dually-flat-structure/
│       ├── notes.md         the notes: YAML front matter + Markdown with LaTeX maths
│       ├── figures/         themed SVGs and an optional self-contained interactive.html
│       └── code/            runnable checks that print every number quoted in the notes
├── foundations/             optional background pages that several chapters lean on (same layout)
├── tools/
│   ├── build.py             notes.md -> build/<slug>/index.html, plus the index / book map
│   ├── new_chapter.py       scaffolds a chapter from the book's own section outline
│   ├── bookmap.json         chapter and section titles with page ranges (no text of the book)
│   └── style.css            shared stylesheet (light and dark)
├── book/                    gitignored: put your own copy of the PDF here
├── local/                   gitignored: scratch that is never tracked or published
└── build/                   gitignored render output
```

Slugs are `chNN-<short-topic>`. Everything renders into a flat `build/<slug>/`, so slugs must be unique across
`chapters/` and `foundations/`, and cross-links between pages are always `../<slug>/index.html`.

## Requirements

- [`pandoc`](https://pandoc.org): `brew install pandoc`
- Python 3.9+ (standard library only for the build; the chapter checks also use `numpy`)

Maths is rendered with KaTeX loaded from a CDN, so the built pages need a network connection the first time they are
opened.

## Usage

```sh
make                                       # render everything into build/
make serve                                 # render, then serve on http://localhost:8000
make new CH=2 SLUG=exponential-families    # scaffold chapters/ch02-exponential-families/notes.md
make list                                  # list the pages in the repo
make check                                 # pandoc present, notes parse, no PDF tracked
make verify                                # run every chapters/*/code/*.py
make clean                                 # remove build/
```

`make` is incremental: it only re-runs when a `notes.md`, a figure, the build script, the stylesheet or the book map
has changed.

## Adding a chapter

```sh
make new CH=2 SLUG=exponential-families
```

This creates `chapters/ch02-exponential-families/notes.md` with the front matter filled in from `tools/bookmap.json`
and the chapter's section titles laid out as headings to fill in. The front matter the build reads:

```yaml
---
title: "Chapter 2: ..."
short_title: "Ch. 2 — Exponential families"   # shown in the sidebar
chapter: 2
category: "Part I"                             # Part I..IV; "Foundations" for background pages
book_pages: "31–50"
url: "https://doi.org/10.1007/978-4-431-55978-8"
tags: [exponential-family, mixture-family]
status: draft                                  # draft | reading | read
---
```

The parser in `tools/build.py` deliberately supports only scalars and inline lists. The sidebar groups chapters by
Part and the index page is a map of the whole book: chapters without notes show as "notes to come".

A loose section order that works: *In one paragraph* → *The spine of the argument* → *Setup and notation* → the
results, in the book's order → *Questions and doubts* → *Takeaways*. The doubts section is the point of the exercise.

### Code and figures

Where the book makes a claim worth checking, `chapters/<slug>/code/` holds a small self-contained script that checks it,
with a `__main__` printing exactly the numbers quoted in the notes. `make verify` runs them all, so a claim in the prose
cannot quietly drift from the code that produced it. Standard library and numpy only. Scripts regenerate their figures
with `--figures`. A page `figures/interactive.html` is published beside the notes and linked from them.

## Publishing

Every push to `main` triggers `.github/workflows/pages.yml`, which installs pandoc, runs `make`, checks that the built
HTML has no filesystem-absolute asset paths, and deploys `build/` to GitHub Pages. Two guards keep the book out of the
site: the workflow fails if any PDF or DjVu file is tracked by git, and `tools/build.py` refuses to finish if one ends
up in `build/`.

## The book

Amari, S. *Information Geometry and Its Applications*. Applied Mathematical Sciences 194, Springer Japan, 2016
(corrected publication 2020). To work along with the notes, get the book from the publisher and keep your copy at
`book/Information_Geometry_and_Its_Applications.pdf` (ignored by git).

## Chapters

| Ch. | Title | Notes |
|---:|---|---|
| 1 | Manifold, Divergence and Dually Flat Structure | [read online](https://msrepo.github.io/information_geometry_notes/ch01-dually-flat-structure/) · [interactive](https://msrepo.github.io/information_geometry_notes/ch01-dually-flat-structure/figures/interactive.html) · [source](chapters/ch01-dually-flat-structure/notes.md) · [code](chapters/ch01-dually-flat-structure/code/) |
| 2 | Exponential Families and Mixture Families of Probability Distributions | [read online](https://msrepo.github.io/information_geometry_notes/ch02-exponential-and-mixture-families/) · [source](chapters/ch02-exponential-and-mixture-families/notes.md) |
| 3 | Invariant Geometry of Manifold of Probability Distributions | [read online](https://msrepo.github.io/information_geometry_notes/ch03-invariant-geometry/) · [interactive](https://msrepo.github.io/information_geometry_notes/ch03-invariant-geometry/figures/interactive.html) · [source](chapters/ch03-invariant-geometry/notes.md) · [code](chapters/ch03-invariant-geometry/code/) |
| 4 | α-Geometry, Tsallis q-Entropy and Positive-Definite Matrices | [read online](https://msrepo.github.io/information_geometry_notes/ch04-alpha-geometry/) · [interactive](https://msrepo.github.io/information_geometry_notes/ch04-alpha-geometry/figures/interactive.html) · [source](chapters/ch04-alpha-geometry/notes.md) · [code](chapters/ch04-alpha-geometry/code/) |
| 5 | Elements of Differential Geometry | [read online](https://msrepo.github.io/information_geometry_notes/ch05-elements-of-differential-geometry/) · [interactive](https://msrepo.github.io/information_geometry_notes/ch05-elements-of-differential-geometry/figures/interactive.html) · [source](chapters/ch05-elements-of-differential-geometry/notes.md) · [code](chapters/ch05-elements-of-differential-geometry/code/) |
| 6 | Dual Affine Connections and Dually Flat Manifold | [read online](https://msrepo.github.io/information_geometry_notes/ch06-dual-connections/) · [interactive](https://msrepo.github.io/information_geometry_notes/ch06-dual-connections/figures/interactive.html) · [source](chapters/ch06-dual-connections/notes.md) · [code](chapters/ch06-dual-connections/code/) |
| 7 | Asymptotic Theory of Statistical Inference | [read online](https://msrepo.github.io/information_geometry_notes/ch07-asymptotic-theory-of-inference/) · [interactive](https://msrepo.github.io/information_geometry_notes/ch07-asymptotic-theory-of-inference/figures/interactive.html) · [source](chapters/ch07-asymptotic-theory-of-inference/notes.md) · [code](chapters/ch07-asymptotic-theory-of-inference/code/) |
| 8 | Estimation in the Presence of Hidden Variables | [read online](https://msrepo.github.io/information_geometry_notes/ch08-hidden-variables/) · [interactive](https://msrepo.github.io/information_geometry_notes/ch08-hidden-variables/figures/interactive.html) · [source](chapters/ch08-hidden-variables/notes.md) · [code](chapters/ch08-hidden-variables/code/) |
| 9 | Neyman–Scott Problem: Estimating Function and Semiparametric Statistical Model | [read online](https://msrepo.github.io/information_geometry_notes/ch09-neyman-scott-and-semiparametrics/) · [interactive](https://msrepo.github.io/information_geometry_notes/ch09-neyman-scott-and-semiparametrics/figures/interactive.html) · [source](chapters/ch09-neyman-scott-and-semiparametrics/notes.md) · [code](chapters/ch09-neyman-scott-and-semiparametrics/code/) |
| 10 | Linear Systems and Time Series | [read online](https://msrepo.github.io/information_geometry_notes/ch10-linear-systems-and-time-series/) · [interactive](https://msrepo.github.io/information_geometry_notes/ch10-linear-systems-and-time-series/figures/interactive.html) · [source](chapters/ch10-linear-systems-and-time-series/notes.md) · [code](chapters/ch10-linear-systems-and-time-series/code/) |
| 11 | Machine Learning | [read online](https://msrepo.github.io/information_geometry_notes/ch11-machine-learning/) · [interactive](https://msrepo.github.io/information_geometry_notes/ch11-machine-learning/figures/interactive.html) · [source](chapters/ch11-machine-learning/notes.md) · [code](chapters/ch11-machine-learning/code/) |
| 12 | Natural Gradient Learning and Its Dynamics in Singular Regions | [read online](https://msrepo.github.io/information_geometry_notes/ch12-natural-gradient-and-singular-regions/) · [interactive](https://msrepo.github.io/information_geometry_notes/ch12-natural-gradient-and-singular-regions/figures/interactive.html) · [source](chapters/ch12-natural-gradient-and-singular-regions/notes.md) · [code](chapters/ch12-natural-gradient-and-singular-regions/code/) |
| 13 | Signal Processing and Optimization | [read online](https://msrepo.github.io/information_geometry_notes/ch13-signal-processing-and-optimization/) · [interactive](https://msrepo.github.io/information_geometry_notes/ch13-signal-processing-and-optimization/figures/interactive.html) · [source](chapters/ch13-signal-processing-and-optimization/notes.md) · [code](chapters/ch13-signal-processing-and-optimization/code/) |

Related: the [theory-inclined papers, with annotations](https://msrepo.github.io/theory_inclined_papers_with_annotations/)
site, which has background pages on the [Fisher information matrix](https://msrepo.github.io/theory_inclined_papers_with_annotations/fisher-information/)
and [EM](https://msrepo.github.io/theory_inclined_papers_with_annotations/expectation-maximization/) that these notes lean on.
