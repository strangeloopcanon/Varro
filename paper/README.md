This folder contains the evolving manuscript and artefacts for the paper.

Structure:

* `paper_draft.md` – the markdown source of the draft.
* `figs/` – generated figures to be referenced from the paper (currently empty; run `python scripts/make_figs.py` once data is ready).
* `references.bib` – BibTeX bibliography for citation keys in the draft.

To regenerate figures:

```bash
python scripts/make_figs.py
```

The script will read training logs in `TRAINING_SUMMARY_REPORT.md` or CSV exports and save PNG/SVG figures into `paper/figs/`.

When compiling LaTeX or a camera-ready PDF, copy or symlink this directory.
