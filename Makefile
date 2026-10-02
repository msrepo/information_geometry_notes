# Build the Information Geometry Notes site from chapters/*/notes.md
#
#   make                              render everything into build/
#   make new CH=2 SLUG=exponential-families
#                                     scaffold chapters/ch02-exponential-families/notes.md
#   make serve                        render, then serve build/ on $(PORT)
#   make open                         render, then open build/index.html
#   make list                         list the pages in the repo
#   make check                        verify pandoc, that notes parse, and that no PDF is tracked
#   make verify                       run every chapters/*/code/*.py
#   make clean                        remove build/

PYTHON ?= python3
PORT   ?= 8000

TOOLS := tools
ROOTS := chapters foundations
NOTES := $(foreach r,$(ROOTS),$(wildcard $(r)/*/notes.md))
FIGS  := $(foreach r,$(ROOTS),$(wildcard $(r)/*/figures/*))
CODE  := $(foreach r,$(ROOTS),$(wildcard $(r)/*/code/*.py))
STAMP := .build-stamp

.DEFAULT_GOAL := html
.PHONY: html new serve open list check verify clean help

html: $(STAMP)

$(STAMP): $(NOTES) $(FIGS) $(TOOLS)/build.py $(TOOLS)/style.css $(TOOLS)/bookmap.json
	@$(PYTHON) $(TOOLS)/build.py
	@touch $@

new:
	@test -n "$(CH)" -a -n "$(SLUG)" || { echo "usage: make new CH=2 SLUG=exponential-families"; exit 1; }
	@$(PYTHON) $(TOOLS)/new_chapter.py "$(CH)" "$(SLUG)"

serve: html
	@echo "Serving http://localhost:$(PORT)  (ctrl-c to stop)"
	@cd build && $(PYTHON) -m http.server $(PORT)

open: html
	@open build/index.html 2>/dev/null || xdg-open build/index.html

list:
	@$(PYTHON) -c "import sys; sys.path.insert(0,'$(TOOLS)'); import build; \
	  [print('%-40s %s' % (m['slug'], m.get('title',''))) for _, m in build.discover()]"

# The book is under copyright: it lives in book/ (gitignored) and must never be committed.
check:
	@command -v pandoc >/dev/null || { echo "pandoc missing: brew install pandoc"; exit 1; }
	@$(PYTHON) -c "import sys; sys.path.insert(0,'$(TOOLS)'); import build; \
	  e = build.discover(); print('pandoc ok; %d note(s) parse' % len(e))"
	@if git ls-files 2>/dev/null | grep -iE '\.(pdf|djvu)$$'; then echo "a PDF/DjVu file is tracked by git: the book must not be committed"; exit 1; fi
	@echo "no PDF or DjVu tracked"

verify:
	@test -n "$(CODE)" || { echo "no code to run"; exit 0; }
	@for f in $(CODE); do \
	  echo "=== $$f ==="; $(PYTHON) $$f || exit 1; echo; \
	done
	@echo "all code ran"

clean:
	@rm -rf build $(STAMP)
	@echo "removed build/"

help:
	@sed -n '2,11p' Makefile | sed 's/^#//'
