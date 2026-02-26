# ACL 2026 Demo Paper (LaTeX Draft)

Files:

- `main.tex`: paper body.
- `references.bib`: bibliography entries.

## Compile

Use ACL style files from the official template package (e.g., `acl.sty`, `acl_natbib.bst`) in the same directory or your TeX path, then run:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If `main.pdf` is open/locked in your PDF viewer, compile with a separate jobname:

```bash
pdflatex -jobname=main_submit main.tex
bibtex main_submit
pdflatex -jobname=main_submit main.tex
pdflatex -jobname=main_submit main.tex
```

## Before submission

- Confirm single-author metadata: `Chen Xi` (Independent Researcher, China).
- Confirm code URL and branch: `https://github.com/A-Dawn/A_memorix` (`acl2026-demo-final`).
- Ensure release asset `video.mp4` is uploaded so the URL in `main.tex` resolves.
- Verify ACL 2026 Demo page limit (4 pages for main paper, references excluded).
- Recheck tables/figures after final edits.
