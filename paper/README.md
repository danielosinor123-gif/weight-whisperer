# Paper

- `main.tex` — full draft (Weight-Whisperer)
- `refs.bib` — bibliography
- `main.pdf` — compiled output (6 pages), built locally with Tectonic 0.17.0
- `weight-whisperer-arxiv.tar.gz` — submission-ready source package for arXiv

## Build locally

```bash
tectonic main.tex        # or pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Requires only standard packages: `amsmath`, `amssymb`, `booktabs`, `geometry`, `hyperref`.

## Submit to arXiv

1. Go to https://arxiv.org/submit (logged in).
2. Upload `weight-whisperer-arxiv.tar.gz` as the source (arXiv compiles it themselves).
3. Category: `cs.CL` (primary), `cs.LG` / `cs.AI` as secondary.
4. Title/author as in `main.tex`; license: CC BY 4.0 (recommended for the code+paper release).
5. Verify their auto-compiled PDF, then submit.
