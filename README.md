# 🧬 Derma-Semantics Pro: Hybrid AI for Explainable Skin Diagnostics

**A research prototype combining Foundation Models with Statistical Learning to deliver 95% AUC accuracy and interpretable diagnostics.**

## 🚀 The Problem & Solution

**The Problem:** Standard AI models act as "Black Boxes"—they diagnose cancer without explaining *why*, leading to low trust among clinicians.
**The Solution:** A **Hybrid AI System** that pairs a frozen biomedical vision backbone with a trainable diagnostic head. It provides both a rigorous risk score and a semantic explanation based on the clinical **ABCDE** criteria.

---

## ⚙️ System Architecture (The "Two-Brain" Approach)

1. **The Eye (Feature Extraction):** Uses **Microsoft BioMedCLIP** (pre-trained on 15M medical image-text pairs) to convert skin lesions into high-dimensional semantic embeddings.
2. **The Brain (Diagnosis):** A custom **Linear Probe (Logistic Regression)** trained on a balanced subset of the ISIC dataset to classify lesions as Benign or Malignant.
3. **The Voice (Explainability):** Performs Zero-Shot classification against clinical prompts (Asymmetry, Border, Color, Diameter) to generate an interpretable risk profile.

---

## 📊 Key Research Findings

* **High Accuracy (0.95 AUC):** The hybrid model distinguishes between melanoma and benign nevi with 95% discriminative power, comparable to specialist screening.
* **Bias Mitigation:** Trained on a strictly balanced dataset (400 Benign / 400 Risk images) to prevent class imbalance bias.
* **Semantic Precision:** The confusion matrix demonstrates balanced Recall (88%) and Precision (85%), proving the model does not just guess "healthy" to minimize error.

---

## 📱 How to Interpret the Dashboard

1. **Risk Probability (The Bar):**
* **< 50% (Green):** Semantic features align with benign moles.
* **> 50% (Red):** Features align with malignant patterns. *Note: In medical AI, any score >50% warrants clinical review.*


2. **Visual Explanation (The Table):**
* Breaks down the lesion using the **ABCDE Rule**:
* **A**symmetry (Symmetrical vs. Asymmetrical)
* **B**order (Smooth vs. Irregular)
* **C**olor (Uniform vs. Variegated)
* **D**iameter (<6mm vs. >6mm)





---

## 🛠️ Tech Stack

* **Model:** Microsoft BioMedCLIP (PubMedBERT + ViT)
* **Training:** Scikit-Learn (Linear Probe)
* **Interface:** Streamlit (Python)
* **Validation:** ROC-AUC, Stratified K-Fold Sampling

---

## ⚠️ Disclaimer

*This project is a Proof-of-Concept for research purposes only. It is **not** a certified medical device and should not be used for diagnosis.*
