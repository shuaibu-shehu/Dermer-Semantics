
import streamlit as st
import torch
import torch.nn as nn
import open_clip
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model

# --- PAGE CONFIG ---
st.set_page_config(page_title="Derma-Semantics Pro", layout="wide", page_icon="🧬")

st.title("🧬 Derma-Semantics Pro: Specialized Diagnostic")
st.markdown("""
**System Status:** ✅ Loaded "Turbocharged" Contrastive Model (Rank 32)
""")

# --- 1. LOAD SYSTEM (SPECIALIZED CONTRASTIVE MODEL) ---
@st.cache_resource
def load_system():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # A. Load Base BioMedCLIP Structure
    print("⏳ Loading BioMedCLIP Base...")
    model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    model.to(device)

    # B. Inject LoRA Layers (Re-creating the architecture from training)
    def get_linear_layer_names(module):
        target_names = set()
        for name, mod in module.named_modules():
            if isinstance(mod, torch.nn.Linear):
                target_names.add(name.split('.')[-1])
        return list(target_names)

    # 1. Apply to Vision
    vision_targets = get_linear_layer_names(model.visual)

    # === CRITICAL UPDATE: MATCH TRAINING CONFIG ===
    # r=32, lora_alpha=64 (Matches Option G Training)
    config_vision = LoraConfig(r=32, lora_alpha=64, target_modules=vision_targets, lora_dropout=0.1, bias="none")
    model.visual = get_peft_model(model.visual, config_vision)

    # 2. Text Encoder was unfrozen during training (state_dict handles the weights)

    # C. Load Your New Specialized Weights
    # weights_path = "/content/drive/MyDrive/Colab Notebooks/biomedclip_balanced_best.pt"
    weights_path = "/content/drive/MyDrive/Colab Notebooks/biomedclip_contrastive_finetuned (1).pt"
    try:
        if os.path.exists(weights_path):
            print("📂 Loading Specialized Contrastive Weights...")
            checkpoint = torch.load(weights_path, map_location=device)

            # Handle state_dict key structure
            sd = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

            # Load weights (strict=False because CLIP has some unused parameters sometimes)
            msg = model.load_state_dict(sd, strict=False)
            print(f"✅ Weights Loaded! (Missing keys expected for frozen parts: {len(msg.missing_keys)})")
        else:
            st.warning("⚠️ Fine-tuned weights not found. Using generic BioMedCLIP (Results will be less accurate).")
    except Exception as e:
        st.error(f"Error loading weights: {e}")

    model.eval()
    return model, preprocess, tokenizer, device

model, preprocess, tokenizer, device = load_system()

# --- 2. DYNAMIC SALIENCY FUNCTION ---
def get_saliency_map(image, model, preprocess, tokenizer, device, target_text):
    model.eval()

    img_input = preprocess(image).unsqueeze(0).to(device)
    img_input.requires_grad_() # Track gradients

    text_input = tokenizer([target_text]).to(device)

    # Forward Pass
    image_features = model.encode_image(img_input)
    text_features = model.encode_text(text_input)

    # Normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Similarity & Backward
    similarity = (image_features @ text_features.T)[0, 0]
    model.zero_grad()
    similarity.backward()

    # Gradients
    gradients = img_input.grad.data.cpu().numpy()[0]
    saliency = np.max(np.abs(gradients), axis=0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    return saliency, similarity.item()

def plot_heatmap_overlay(original_image, heatmap):
    heatmap_resized = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    overlay = np.array(original_image) / 255.0 * 0.6 + heatmap_colored * 0.4

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(original_image); ax[0].set_title("Original"); ax[0].axis('off')
    ax[1].imshow(heatmap_resized, cmap='jet'); ax[1].set_title("AI Attention"); ax[1].axis('off')
    ax[2].imshow(overlay); ax[2].set_title("Overlay"); ax[2].axis('off')
    return fig

# --- 3. MAIN UI ---
col1, col2 = st.columns([1, 1.5])

# DEFINING THE 7 CLASSES (Must match your training logic)
isic_classes = {
    "Melanoma (Cancer)": "High risk melanoma skin cancer",
    "Basal Cell Carcinoma (Cancer)": "Basal cell carcinoma skin cancer",
    "Actinic Keratosis (Pre-Cancer)": "Actinic keratosis pre-cancerous lesion",
    "Melanocytic Nevus (Benign)": "Benign melanocytic nevus mole",
    "Benign Keratosis (Benign)": "Benign keratosis-like lesion",
    "Dermatofibroma (Benign)": "Benign dermatofibroma skin lesion",
    "Vascular Lesion (Benign)": "Benign vascular skin lesion"
}

# Define which are "Bad" for the summary risk alert
malignant_keys = ["Melanoma (Cancer)", "Basal Cell Carcinoma (Cancer)", "Actinic Keratosis (Pre-Cancer)"]

with col1:
    st.subheader("1. Patient Input")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Clinical View", use_column_width=True)

with col2:
    st.subheader("2. AI Analysis")

    if uploaded_file and st.button("Run Diagnostic"):
        with st.spinner("Analyzing Lesion Pattern..."):

            # A. SEMANTIC PROFILE (Zero-Shot Classification)
            img_tensor = preprocess(image).unsqueeze(0).to(device)
            sim_scores = {}

            with torch.no_grad():
                img_feat = model.encode_image(img_tensor)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)

                for label, prompt in isic_classes.items():
                    txt = tokenizer([prompt]).to(device)
                    txt_feat = model.encode_text(txt)
                    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
                    score = (img_feat @ txt_feat.T).item()
                    sim_scores[label] = score

            # Sort Results
            df_scores = pd.DataFrame(list(sim_scores.items()), columns=["Condition", "Score"])
            df_scores = df_scores.sort_values(by="Score", ascending=False)
            top_condition = df_scores.iloc[0]["Condition"]

            # B. DIAGNOSTIC SUMMARY
            st.divider()
            if top_condition in malignant_keys:
                 st.error(f"**Primary Diagnosis: {top_condition}**")
                 st.caption("⚠️ The model detected patterns consistent with malignancy.")
            else:
                 st.success(f"**Primary Diagnosis: {top_condition}**")
                 st.caption("✅ The model detected patterns consistent with benign lesions.")

            # C. BAR CHART
            st.markdown("### 📊 Semantic Similarity Profile")
            st.caption("Match confidence for each specific condition:")
            st.bar_chart(df_scores.set_index("Condition"))

            # D. DYNAMIC X-RAY
            st.divider()
            st.markdown("### 👁️ Dynamic X-Ray Vision")

            # Selectbox defaults to the top prediction
            target_class = st.selectbox("Show features for:", list(isic_classes.keys()), index=list(isic_classes.keys()).index(top_condition))
            target_prompt = isic_classes[target_class]

            with st.spinner(f"Generating heatmap for '{target_class}'..."):
                try:
                    heatmap, score = get_saliency_map(image, model, preprocess, tokenizer, device, target_prompt)
                    fig = plot_heatmap_overlay(image, heatmap)
                    st.pyplot(fig)
                    st.info(f"Visualizing pixels that match description: **'{target_prompt}'**")
                except Exception as e:
                    st.error(f"Error: {e}")

