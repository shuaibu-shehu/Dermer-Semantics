# ML4H 2026 submission — build instructions

**Paper:** *Faithful and Fair? A Skin-Tone Equity Audit of the Predictions and Explanations of a Concept-Based Dermatology Model.*

## Files

| File | What it is |
|------|-----------|
| `main.tex` | The paper. This is the file you compile. |
| `ref.bib` | Bibliography (all cited keys defined; see the one `% NOTE` about the DermLIP citation). |
| `figures/*.pdf` | The five figures, vector PDF with embedded TrueType fonts. |
| `make_pipeline_figure.py` | Regenerates `pipeline.pdf` (Fig. 1, the method schematic). Pure diagram, no data. |
| `make_figures.py` | Regenerates `audit/faithfulness/mitigation.pdf` from `../../results/nbC2_fitz_results.json`. |
| `make_diagnostic_figures.py` | Regenerates `diagnostic.pdf` (per-concept separability + violations) directly from the cached features in `../../caches/fairness_cache/`; also writes `../../results/nbC2_fitz_diagnostic.json`. Needs `torch` (retrains the ERM heads on CPU, ~seconds). |
| `ML4H 2026 Template.tex` | The original ML4H template, kept for reference only. |

## Compile (use Overleaf — the `jmlr` class is not installed locally)

1. Go to the **ML4H 2026 template on Overleaf** (or upload the official template, which includes `jmlr.cls`). The class is *not* in this folder and is not installable here, so a local `pdflatex` will not work.
2. Add `main.tex`, `ref.bib`, and the `figures/` folder to the project.
3. Set the **main document** to `main.tex`, compiler **pdfLaTeX**, and compile. Run it twice (or let Overleaf auto-run) so BibTeX resolves the references.

## Before you submit — checklist

- **Track.** The file is set to `\mlhtrack{proceedings}` — the archival PMLR track, **8-page** main-content limit (references and appendices excluded). Switch to `\mlhtrack{findings}` (4-page limit, non-archival) only if you decide to aim shorter.
- **Page limit (8pp).** The diagnostic per-concept figure + its analysis now live in the **appendix** (`\appendix`, after the bibliography) so they don't count toward the 8 pages; the main body keeps 4 figures + 2 tables. **Verify the exact page count on Overleaf.** If still over 8, trim in this order: (1) tighten Related Work, (2) shorten the "What a convincing version would need" paragraph, (3) trim the Discussion opening. Do not cut the cohort table or the statistics paragraph — reviewers want those.
- **Anonymous.** `\author{Anonymous Author(s)}` and the code link must stay anonymized for review. The Data/Code Availability paragraph promises an anonymized repo — attach one (e.g., an anonymized GitHub or a zip) at submission.
- **Camera-ready only.** After acceptance, set `\finaltrue`, fill in authors/affiliations, add `\acks{...}`, and de-anonymize the code link.
- **Citations.** All references are real; the `dermlip2025` key now points to the canonical Derm1M paper (Yan et al., ICCV 2025, arXiv:2503.14911). If your compiled PDF shows citations as `?`, that's just BibTeX not having run — recompile from scratch on Overleaf (LaTeX → BibTeX → LaTeX → LaTeX).

## Regenerate figures

```
python "papers/ML4H 2026/make_pipeline_figure.py"      # Fig. 1 (schematic)
python "papers/ML4H 2026/make_figures.py"              # audit / faithfulness / mitigation
python "papers/ML4H 2026/make_diagnostic_figures.py"   # diagnostic (per-concept)
```
