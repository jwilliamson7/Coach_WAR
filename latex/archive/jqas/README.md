# JQAS Submission (archived)

Submitted to the *Journal of Quantitative Analysis in Sports*, revised after two rounds of review, and
rejected in October 2026. Archived here as the last complete version of the manuscript; the work moved
to the SSAC27 Research Paper Competition (`latex/ssac/`).

## Contents

| Path | What it is |
|---|---|
| `2026-Williamson-Jon-Portfolio-Coach-WAR.tex` / `.pdf` | Final revised manuscript (De Gruyter `dgruyter` class, blind review, tracked changes: `\rev{}` additions, `\del{}` deletions) |
| `dgruyter.sty`, `dgruyter.ist` | JQAS LaTeX template files the manuscript needs |
| `figures/` | Manuscript figures |
| `submission/` | Cover letter, title page, ethics/legal declaration, submission checklist |
| `reviewer_responses/` | Responses to both review rounds, plus editor-response drafts (model choice, reliability, scheme obsolescence) |
| `notes/` | Working notes from the original Word-to-LaTeX conversion |

## Compiling

The manuscript compiles in place (figures and style file are resolved relative to this folder):

```bash
cd latex/archive/jqas
pdflatex 2026-Williamson-Jon-Portfolio-Coach-WAR.tex
pdflatex 2026-Williamson-Jon-Portfolio-Coach-WAR.tex
```

The bibliography is an inline `thebibliography`, so no BibTeX pass is needed.

## Still wired to this folder

- `analysis/validate_report_consistency.py` reads this manuscript (`PAPER_DIR`).
- `analysis/create_trajectory_figure.py`, `analysis/create_career_distribution_figure.py`, and
  `analysis/regenerate_latex_figures.py` write their figures to `figures/` here.
