import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ui_utils import plot_overlay_analysis
from model_loader import load_methane_model, get_prediction
from data_utils import load_test_metadata, preprocess_for_inference, get_images_from_id_for_display, get_images_from_id_for_inference, get_rgb_stacked
import os
import torch

model_path= os.path.join('results', 'results_EfficientNetV2', 'results_xp_1', 'models', 'best_EfficientNetV2_fold_0.pth')

st.set_page_config(
    page_title="Methane Sentinel - StarCop",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
    <style>
    .main { backgroundColor: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.image("app/assets/methane_logo.png", width=200) 
    st.header("⚙️ Paramètres")
    
    st.subheader("1. Sélection des données")

    df = load_test_metadata()
    test_ids = df['id'].tolist()
    selected_id = st.selectbox("Choisir une image de test", test_ids)
    
    rgb_sample, mag1c_sample = get_images_from_id_for_display(selected_id)

    st.subheader("2. Configuration Modèle")
    selected_fold = st.selectbox("Sélectionner le modèle", ['EfficientNetV2'])
    
    threshold = st.slider("Seuil de détection (Confidence)", 0.1, 0.9, 0.5, 0.05)
    
    st.divider()
    predict_btn = st.button("Lancer la détection", use_container_width=True, type="primary")

# Titre dynamique
st.title(f"Analyse de l'événement : {selected_id}")

# ---  INPUTS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Image RGB (Sentinel-2)")

    st.image(rgb_sample, use_column_width=True, caption="RGB 640 nm - 550 nm - 460 nm")

with col2:
    st.subheader("🔬 Filtre MAG1C")

    st.image(mag1c_sample, use_column_width=True, caption="Filtre MAG1C - Intensité du panache de méthane")

st.divider()

# ---  PRÉDICTION ---
if predict_btn:
    
    input, gt = get_images_from_id_for_inference(selected_id)
    model = load_methane_model(model_path, device="cpu")
    
    with st.spinner("🧠 Le modèle analyse les données Sentinel-2..."):
        with torch.no_grad():
            pred = model(input.unsqueeze(0))
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()
        gt = gt.squeeze().cpu().numpy()

    tab1, tab2 = st.tabs(["📊 Analyse Superposée", "🔍 Comparaison Côte-à-Côte"])

    with tab1:
        st.subheader("Superposition des performances sur le terrain")
        
        empty_l, col_img, empty_r = st.columns([1, 2, 1])

        with col_img:
            fig = plot_overlay_analysis(rgb_sample, pred, gt)
            st.pyplot(fig, use_container_width=True)
            
            st.markdown("""
                <div style="display: flex; justify-content: center; gap: 20px;">
                    <span style="color: #2ecc71;">● True Positive</span>
                    <span style="color: #e74c3c;">● False Positive</span>
                    <span style="color: #f1c40f;">● False Negative</span>
                </div>
            """, unsafe_allow_html=True)

    with tab2:
        empty_l, col_a, col_b, empty_r = st.columns([1, 2, 2, 1])

        with col_a:
            st.image(pred, caption="Prédiction du modèle", use_container_width=True)

        with col_b:
            st.image(gt, caption="Ground Truth (Réalité)", use_container_width=True)
    with st.expander("ℹ️ Informations techniques sur le modèle"):
        st.write(f"Modèle : EfficientNetV2 | Nombre de folds: 5 ")
else:
    st.info("Sélectionnez un événement et cliquez sur 'Lancer la détection' pour voir le résultat.")