# From the ML4H workshop paper to a main-conference / journal paper

The ML4H 2026 submission (`papers/ML4H 2026/`) is deliberately a **workshop-tier** contribution: a rigorous audit with a *preliminary* faithfulness-gap finding and an honest Group-DRO negative. This note records what it would take to turn it into a **MICCAI-main / *Medical Image Analysis*-tier** paper, so the workshop version can be cited as the preliminary study and extended rather than restarted.

The through-line: **turn H3 (fairness of explanation faithfulness) from an observation into a robust, mechanism-backed contribution with a working fix.** Three parts.

## 1. Robustness — make the faithfulness gap un-dismissable
Right now the gap is significant only *in-distribution* (Fitzpatrick17k), on a single backbone, with a single metric. A reviewer will call that an artifact. Close that door:
- **≥2 backbones**: DermLIP + BiomedCLIP + one larger dermatology encoder (e.g. a ViT-L). Show the gap is not model-specific.
- **≥2 faithfulness metrics**: monotonicity/intervention-consistency *plus* e.g. concept-deletion/completeness or a sufficiency measure. Show it is not metric-specific.
- **A powered external set**: the current external test (DDI, 207 dark) is too small to detect the gap. Find or assemble a larger tone-diverse, biopsy-labeled set so external significance is achievable.

## 2. Mechanism — explain *why*
Separate the two candidate causes of the in-distribution gap:
- the **concepts themselves** are noisier on darker skin (a backbone problem), vs.
- the **head's response surface** is genuinely less monotone in the region of concept space where dark-skin lesions fall (a bottleneck problem).
The missing ingredient is **concept ground truth on diverse clinical images**, which does not exist at scale — so a smaller expert-annotated diverse concept set (even a few hundred images) would be a real, fundable data contribution on its own.

## 3. A mitigation that works — Group-DRO does not
Since worst-group training does not close the gap, propose a method that does, e.g.:
- a **per-group monotonicity penalty** (equalize faithfulness directly, not just loss), or
- **concept-space reweighting / calibration** per tone.
A method that provably narrows the faithfulness gap without hurting accuracy is the single biggest lever toward a main-track accept.

## Timing & compute
- This is a **6–12 month** project, and item (1)/(3) will exceed Kaggle free-tier — budget for a modest cloud GPU allocation or a lab machine.
- **Venue targets**: MICCAI 2027 workshops (FAIMI / ISIC Skin Image Analysis, deadlines ~June–July 2027) as the stepping stone; MICCAI-main or *Medical Image Analysis* / *npj Digital Medicine* for the full version.
- **Sequence**: land the ML4H 2026 workshop paper first (Sept 10 2026), then build the extension citing it.
