# Papers and Submissions

```
latex/
├── ssac/                  # ACTIVE: SSAC27 Research Paper Competition (Football track)
│   ├── SSAC2027-Williamson-Jon-Abstract.docx   # abstract, built from the SSAC Word template
│   ├── SSAC2027-Williamson-Jon-Abstract.pdf    # exported from Word; this is what gets uploaded
│   └── figures/                                # abstract figures (see below)
└── archive/
    └── jqas/              # ARCHIVED: JQAS submission, rejected Oct 2026 (see its README)
```

## SSAC27 (active)

- Abstract limits: fewer than 500 words including title and body, at most two tables/figures combined,
  Introduction / Methods / Results / Conclusion.
- If selected, the full paper is due Dec. 4, 2026. The archived JQAS manuscript is the starting point.
- Figure 1 of the abstract: `python analysis/create_ssac_abstract_figure.py`
  (writes `latex/ssac/figures/ssac_decade_background_gap.png`).

## Paper consistency check

`python analysis/validate_report_consistency.py` checks every number in the manuscript against the
canonical CSV outputs. It currently reads the archived JQAS manuscript
(`latex/archive/jqas/2026-Williamson-Jon-Portfolio-Coach-WAR.tex`); repoint `PAPER_DIR` when a new
manuscript becomes the working copy.
