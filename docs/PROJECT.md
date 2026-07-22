# Faithful & Fair Concept-Based Dermatology

*An equity audit of a dermatology vision-language model, a faithfulness-reward method, and an honest test of whether standard fairness training and explanation faithfulness hold up across skin tones.*

---

## 1. What this project is

Most dermatology AI is judged only on whether it predicts the right label. This project instead **audits** a dermatology foundation model through two lenses that the field usually ignores, and reports the results — positive and negative — with confidence intervals and significance tests rather than point estimates.

1. **Equity** — does the model detect malignancy equally well across skin tones, and does a standard fairness intervention fix it if not?
2. **Faithfulness** — is the model's stated clinical reasoning (the concepts it claims to see) actually driving its diagnosis, and is that reasoning *equally* faithful across skin tones?

The vehicle is a **concept-bottleneck classifier**: a frozen dermatology vision-language model ([DermLIP](https://huggingface.co/redlessone/DermLIP_ViT-B-16)) produces clinical concept scores (the 7-point checklist criteria), and a small head maps *concepts → diagnosis*. Because the diagnosis sees only the concepts, the explanation is causal by construction; the research is about whether that explanation is **faithful** and **fair**.

> **Narrative note.** This began as an "RL reasoning" idea. The reinforcement-learning element is a single-step policy gradient over 7 cached features and is *not* the contribution — treat it as an optimization detail. The contribution is the **audit + method + honest negative results**, framed as *faithful & fair concept-based dermatology*.

---

## 2. Contributions (what we can actually defend)

1. **A faithfulness-reward method (H1).** On Derm7pt (which has concept ground truth), adding a soft **monotonicity / intervention-consistency** objective drives intervention consistency from **0.74 → 1.00 at no accuracy cost** (accuracy 0.80 → 0.83, melanoma AUROC unchanged at ~0.87). Rewarding the model to match the *discrete* 7-point rule instead is harmful — it caps the model at the rule and collapses melanoma sensitivity. *The right way to reward faithfulness is directional, not rule-equality.*

2. **A powered equity audit — the disparity is real and significant.** DermLIP's malignancy detection is significantly worse on dark skin, shown two ways with non-overlapping/CI-excluded evidence:
   - Concept-bottleneck on **Fitzpatrick17k** (n≈2,365 test): light–dark AUROC gap **0.162, 95% CI [0.071, 0.262], permutation p < 0.0001**.
   - Zero-shot on **biopsy-proven DDI**: light **0.752 [0.674, 0.826]** vs dark **0.564 [0.467, 0.664]** — disjoint 95% CIs.

3. **First (preliminary) evidence that *explanation faithfulness itself* is inequitable (H3).** On the powered Fitzpatrick test set, monotonicity is significantly lower on dark skin: light **0.650 [0.643, 0.658]** vs dark **0.603 [0.589, 0.618]** — **disjoint 95% CIs**. This is a genuinely novel observation (fairness of *explanations*, not just predictions). It is **qualified**, not a slam dunk — see §7.

4. **Two honest negative results.** (a) **Group-DRO does not fix the tone gap:** worst-group AUROC change is +0.014 [-0.011, 0.041] in-distribution and -0.034 [-0.074, 0.001] external — both CIs span (or sit below) 0. (b) **Cross-dataset transfer collapses:** a bottleneck trained on Fitzpatrick tested on biopsy-proven DDI drops to near chance (AUROC 0.46–0.55), a domain-shift result worth reporting.

---

## 3. Objectives / research questions

| ID | Question | Status |
|----|----------|--------|
| **H1** | What is the right way to *reward* explanation faithfulness in a concept bottleneck? | ✅ **Answered** (Derm7pt) — monotonicity, not rule-matching |
| **Audit** | Does DermLIP's malignancy detection differ by skin tone? | ✅ **Answered — significant** (Fitzpatrick p<1e-4; DDI disjoint CIs) |
| **H2** | Does Group-DRO raise worst-skin-tone-group AUROC vs ERM? | ✅ **Answered — NEGATIVE** (no significant gain either dataset) |
| **H3** | Does explanation **faithfulness itself** degrade on darker skin? | 🟡 **Significant in-distribution, qualified** (disjoint CIs on Fitzpatrick; n.s. on small external DDI; not closed by training) |

**The honest shape of the paper:** a rigorous audit with a significant disparity, a novel-but-preliminary faithfulness-gap finding, and a clear negative on the standard mitigation. That is a coherent **workshop paper**, not a main-conference headline.

---

## 4. Method

- **Backbone:** [DermLIP ViT-B/16](https://huggingface.co/redlessone/DermLIP_ViT-B-16) (trained on Derm1M; dermatology-specific CLIP), used **frozen**. Chosen over BioMedCLIP (kept as a baseline) — derm-specific, lightweight, ships a clinical concept ontology.
- **Concepts:** the **7-point checklist** criteria. On Derm7pt these are supervised probes (ground-truth labels exist); on clinical-photo datasets (Fitzpatrick17k, DDI) they are DermLIP zero-shot scores.
- **Faithfulness (H1):** a soft **monotonicity** constraint — *a melanoma criterion must never lower P(melanoma)* — which guarantees directional faithfulness; contrasted against a discrete rule-matching reward (harmful).
- **Equity:** **Group-DRO** (Sagawa et al.) over skin-tone groups {light, mid, dark}, optimizing the worst group; evaluated against ERM.
- **Statistics (this is what makes it publishable):** per-group AUROC with bootstrap 95% CIs; a **permutation test** on the light–dark AUROC gap; a **paired bootstrap** on the Group-DRO − ERM worst-group difference; and **bootstrap CIs on the monotonicity metric** so the H3 gap is significance-gated, not eyeballed.
- **Compute strategy (key enabler):** encode each image **once** with the frozen backbone, cache embeddings + concept scores, then run all modeling on cached vectors + tiny heads. Every experiment is CPU-fast; only the one-time encoding needs a GPU, so the whole study fits **Kaggle free GPU**.

---

## 5. Pipeline (notebooks)

| Notebook | Role | Runs on |
|----------|------|---------|
| `01_dermlip_cache_derm7pt.ipynb` | Encode Derm7pt → embeddings + 7-pt concept scores + labels | Kaggle GPU (once) |
| `02_concept_policy_faithfulness.ipynb` | **H1** — concept→diagnosis head; correctness vs rule-reward vs monotonicity | PC / CPU |
| `03_cache_fitz_ddi_dermlip.ipynb` | Encode fairness datasets (Fitzpatrick17k / DDI) → caches | Kaggle GPU (once) |
| `04_fairness_groupdro.ipynb` | **Audit / H2 / H3** — ERM vs Group-DRO, per-tone AUROC + faithfulness, with CIs + significance tests | PC / CPU |
| `05_concept_transfer_fairness.ipynb` | Concept-transfer test + bootstrap-CI audit (negative: transfer fails) | PC / CPU |

`notebooks/legacy/` holds the pre-pivot BioMedCLIP demo notebooks.

---

## 6. Datasets

| Dataset | Use | Notes |
|---------|-----|-------|
| **Derm7pt** | H1 training (has 7-pt concept ground truth) | ~1,011 cases, dermoscopy + clinical |
| **Fitzpatrick17k** | Powered audit / H2 / H3 training set | 16,577 images cached; tones grouped light 7,755 / mid 6,089 / dark 2,168; malignant 2,263 (dark 208). Images `mobaswiralfarabi/fitzpatrick17k_original`; C1 auto-downloads the labels CSV |
| **DDI** | Equity audit + external test | 656 biopsy-proven, tone-balanced (208/241/207) |
| ~~PAD-UFES-20~~ | ❌ ruled out for equity | only 11 dark-skin images |
| ~~SCIN~~ | ❌ ruled out | only ~1.35% malignant → can't power a malignancy study |

---

## 7. Results (with CIs + significance)

**H1 (Derm7pt, 5-class, non-linear head, 3 seeds):** concept bottleneck is nearly free on melanoma (AUROC ≈ 0.87). Adding the faithfulness objective raises **intervention consistency 0.74 → 1.00** with accuracy 0.80 → 0.83 and AUROC unchanged. Rule-matching reward is harmful (melanoma sensitivity collapses). Saved: `results/nbB_results.json`.

**Audit — significant skin-tone disparity:**
| Setting | light | mid | dark | gap (light−dark) | test |
|---|---|---|---|---|---|
| Fitzpatrick17k bottleneck | 0.841 [0.810, 0.871] | 0.802 | 0.679 [0.594, 0.765] | **0.162 [0.071, 0.262]** | perm **p < 1e-4** |
| DDI zero-shot (biopsy) | 0.752 [0.674, 0.826] | 0.715 | 0.564 [0.467, 0.664] | disjoint CIs | — |

**H2 — Group-DRO is a negative result:** worst-group AUROC change **+0.014, 95% CI [−0.011, 0.041]** (Fitzpatrick) and **−0.034, 95% CI [−0.074, 0.001]** (DDI external). Group-DRO does **not** significantly raise the worst group in either setting.

**H3 — faithfulness gap, significant in-distribution but qualified:**
| Test set | faithfulness light | faithfulness dark | verdict |
|---|---|---|---|
| Fitzpatrick17k | 0.650 [0.643, 0.658] | 0.603 [0.589, 0.618] | **disjoint CIs → significant** |
| DDI external | 0.662 [0.651, 0.674] | 0.648 [0.634, 0.661] | overlapping → n.s. |

**Negative results worth keeping:** cross-dataset transfer (train Fitzpatrick → test DDI) collapses to near-chance (domain shift); Derm7pt→DDI concept transfer also fails (`05`).

Saved numbers: `results/nbC2_fitz_results.json` (full CIs + p-values), `results/nbB_results.json`.

---

## 8. Honest status & limitations

- **Solid:** the H1 method, and the powered, significant equity audit (two datasets).
- **Novel but preliminary (H3):** the faithfulness gap is statistically significant *in-distribution* (Fitzpatrick), but (a) it does **not** significantly replicate on the small biopsy-proven external set (DDI, n=207 dark — underpowered), (b) absolute faithfulness is low for all groups (~0.6, because zero-shot concepts are weak on clinical photos), and (c) the monotonicity CI captures test-set resampling with 3 fixed seeds, not full retraining variance. Report it as *preliminary evidence that motivates the problem*, not a settled result.
- **Negative (H2):** Group-DRO does not close the gap — useful for a community that often assumes it will.
- **Known weaknesses:** the "RL" is a thin bandit; "multimodal" is image + concept text only; the 7-pt concepts are dermoscopy-oriented and transfer poorly to clinical photos; Fitzpatrick17k skin tone is image-estimated (not ground truth).
- **Scope discipline:** do **not** add datasets or a fifth hypothesis. The write-up is the remaining work.

---

## 9. Repository map

```
docs/PROJECT.md                 ← this document
notebooks/                      ← research pipeline (01–05)
notebooks/legacy/               ← old BioMedCLIP demo notebooks
caches/derm7pt_dermlip_cache/   ← Notebook A output (regenerable)
caches/fairness_cache/          ← Notebook C1 output (features_fitz/ddi, regenerable)
results/                        ← nbB_results.json, nbC2_fitz_results.json
legacy_demo/                    ← old Streamlit app (app.py)
requirements.txt
```

---

## 10. Reproduce

1. **Encode (Kaggle GPU, once):** run `01` (Derm7pt) and `03` (Fitzpatrick17k image mirror + DDI; Internet on so DermLIP + labels CSV download). Save each output as a Kaggle Dataset.
2. **Model (local CPU, minutes):** run `02` (H1), then `04` (audit / H2 / H3 with CIs + significance tests) and `05` (transfer audit) against the caches. `04` finishes in ~1 min and writes `results/nbC2_fitz_results.json`.

Local runs need `numpy pandas pyarrow` + CPU `torch`.

---

## 11. Next step

**Write the paper — no more compute is required.** The rigorous numbers are banked in `results/nbC2_fitz_results.json`. Target **ML4H 2026** (Machine Learning for Health Symposium, Sydney, Dec 6–7; **submission deadline Sept 10, 2026**; non-archival Findings track welcomes audits + negative results). The perfect-topic MICCAI workshops (ISIC Skin Image Analysis, FAIMI) closed ~July 1, 2026 — target them for the MICCAI 2027 cycle if H3 is strengthened first. **CHIL 2027** (deadline ~Feb 2027) is the archival backup. Frame the paper around the four contributions in §2, with H3 stated as significant-but-preliminary per §8.
