# Faithful & Fair Concept-Based Dermatology

Research project on the **faithfulness** and **skin-tone equity** of concept-based skin-lesion diagnosis, built on the frozen [DermLIP](https://huggingface.co/redlessone/DermLIP_ViT-B-16) vision-language model.

A concept-bottleneck classifier predicts clinical concepts (the **7-point checklist**) and maps *concepts → diagnosis*, so the explanation is causal by construction. The work studies two questions most derm AI ignores:

- **Faithfulness** — how to make the concept-grounded reasoning clinically valid.
- **Equity** — whether that reasoning is equally faithful across skin tones.

> 📄 **Full write-up, objectives, results, and next steps: [`docs/PROJECT.md`](docs/PROJECT.md).**

## Key findings

- **Faithfulness method (H1):** rewarding the model to match the discrete 7-point *rule* is harmful (collapses melanoma sensitivity); a soft **monotonicity** constraint drives intervention consistency **0.74 → 1.00 at no accuracy cost**.
- **Equity audit (significant):** DermLIP's malignancy detection is **significantly worse on dark skin** — light–dark AUROC gap **0.162, permutation p < 0.0001** on Fitzpatrick17k (n≈16.5k), and **0.75 vs 0.56** with disjoint CIs on biopsy-proven DDI.
- **Faithfulness gap (H3, novel but preliminary):** explanation faithfulness is **significantly lower on dark skin in-distribution** (0.650 vs 0.603, disjoint 95% CIs on Fitzpatrick) — but does not significantly replicate on the small external set; a first, qualified look at *fairness of explanations*.
- **Honest negatives:** Group-DRO does **not** close the tone gap (worst-group change CI spans 0); cross-dataset transfer collapses to near-chance (domain shift).

## Repository

```
notebooks/            research pipeline (01–05); see docs/PROJECT.md §4
notebooks/legacy/     pre-pivot BioMedCLIP demo notebooks
caches/               frozen-feature caches (regenerable)
results/              experiment result JSONs
legacy_demo/          old Streamlit demo (app.py)
docs/PROJECT.md       project documentation
```

## Approach in one line

Encode images **once** with frozen DermLIP → cache embeddings + concept scores → train tiny heads on cached vectors. Only the one-time encoding needs a GPU, so the whole study fits **Kaggle free tier**; all modeling runs on CPU in minutes.

## Quickstart

1. **Encode (Kaggle GPU):** run `notebooks/01…` (Derm7pt) and `notebooks/03…` (Fitzpatrick17k mirror + DDI, Internet on).
2. **Model (local CPU):** run `notebooks/02…` (faithfulness), `notebooks/04…` (fairness), `notebooks/05…` (audit) against the caches.

Local deps: `numpy pandas pyarrow` + CPU `torch`.

---
*Legacy: the original BioMedCLIP + Streamlit demo lives in [`legacy_demo/`](legacy_demo/) and [`notebooks/legacy/`](notebooks/legacy/).*
