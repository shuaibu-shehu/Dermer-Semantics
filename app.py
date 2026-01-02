
import streamlit as st
import torch
import open_clip
import joblib
import numpy as np
import pandas as pd
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Derma-Semantics Pro", layout="wide", page_icon="🧬")

st.title("🧬 Derma-Semantics Pro")
st.markdown("""
**Hybrid Architecture:** 1.  **Feature Extractor:** BioMedCLIP (Microsoft)
2.  **Diagnostic Head:** Linear Probe (Trained on ISIC-2018 Balanced Subset)
""")

# --- LOAD MODELS ---
@st.cache_resource
def load_system():
    # 1. Load Vision Backbone & Transforms
    model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    
    # 2. Load the Tokenizer Explicitly
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    
    # 3. Load Trained Classifier Head
    try:
        classifier = joblib.load('skin_cancer_classifier.pkl')
    except:
        st.error("⚠️ Model file not found! Please run the 'Save Model' step above.")
        return None, None, None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    return model, preprocess, tokenizer, classifier

model, preprocess, tokenizer, classifier = load_system()

# --- SIDEBAR INFO ---
with st.sidebar:
    st.header("Model Specs")
    st.info("**Backbone:** ViT-Base (PubMedBERT)")
    st.info("**Classifier:** Logistic Regression")
    st.info("**Validation AUC:** 0.95")
    st.divider()
    st.warning("RESEARCH PROTOTYPE. NOT FOR MEDICAL USE.")

# --- MAIN INTERFACE ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Patient Input")
    uploaded_file = st.file_uploader("Upload Dermoscopy Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Clinical View", use_column_width=True)

with col2:
    st.subheader("2. AI Analysis")
    
    if uploaded_file and st.button("Run Hybrid Diagnostic"):
        if classifier is None:
            st.error("Classifier failed to load.")
            st.stop()
            
        with st.spinner("Extracting Semantic Features..."):
            # A. PREPROCESSING
            device = "cuda" if torch.cuda.is_available() else "cpu"
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            # B. FEATURE EXTRACTION (BioMedCLIP)
            with torch.no_grad():
                features = model.encode_image(image_input)
                features /= features.norm(dim=-1, keepdim=True)
                
                # Convert to numpy for the classifier
                features_np = features.cpu().numpy().flatten().reshape(1, -1)

            # C. DIAGNOSTIC PREDICTION
            prob_risk = classifier.predict_proba(features_np)[0][1] # Class 1 = Risk
            
            # --- DISPLAY MAIN RESULT ---
            st.divider()
            st.markdown("### 🏥 Diagnostic Prediction")
            
            if prob_risk > 0.5:
                st.error(f"**High Risk Probability: {prob_risk:.1%}**")
                st.progress(prob_risk)
                st.caption("The linear probe classifies this semantic pattern as consistent with malignant features.")
            else:
                st.success(f"**Low Risk Probability: {prob_risk:.1%}**")
                st.progress(prob_risk)
                st.caption("The linear probe classifies this as benign.")

            # D. EXPLAINABILITY (Zero-Shot)
            st.divider()
            st.markdown("### 🔬 Visual Feature Explanation")
            
            # FIXED: Added Diameter back to the dictionary
            prompts = {
                "Asymmetry": ["Symmetrical lesion", "Asymmetrical lesion"],
                "Border": ["Smooth borders", "Irregular, jagged borders"],
                "Color": ["Uniform skin color", "Multiple variegated colors"],
                "Diameter": ["Small lesion under 6mm", "Large lesion over 6mm"]
            }
            
            results = []
            for criterion, texts in prompts.items():
                text_tokens = tokenizer(texts).to(device)
                
                with torch.no_grad():
                    text_features = model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    probs = (100.0 * features @ text_features.T).softmax(dim=-1)
                    risk_score = probs[0][1].item()
                
                results.append({
                    "Feature": criterion,
                    "Observation": texts[1] if risk_score > 0.5 else texts[0],
                    "Confidence": risk_score
                })
            
            df = pd.DataFrame(results)
            st.dataframe(df.style.background_gradient(subset=["Confidence"], cmap="Oranges", vmin=0, vmax=1))

