import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ui_utils import plot_overlay_analysis
from model_loader import load_methane_model, get_prediction
from data_utils import load_test_metadata, preprocess_for_inference, get_images_from_id_for_display, get_images_from_id_for_inference, get_rgb_stacked, count_subdirectories
import os
import torch
import segmentation_models_pytorch as smp
import json

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
    

    st.subheader("2. Configuration Modèle")
    selected_modele_name = st.selectbox("Sélectionner le modèle", ['EfficientNetV2', 'MiT'])
    
    path_to_experiments = os.path.join('results', f'results_{selected_modele_name}')
    num_experiments = count_subdirectories(path_to_experiments)

    selected_experiment = st.selectbox("Sélectionner la version du modèle", [f"xp_{i}" for i in range(1, int(num_experiments) + 1)])

    threshold = st.slider("Seuil de détection (Confidence)", 0.1, 0.99, 0.5, 0.05)

    backup_config_path = os.path.join(path_to_experiments, f'results_{selected_experiment}', 'logs', 'config_backup.json')
    with open(backup_config_path, 'r') as f:
        backup_config = json.load(f) 

    rgb_sample, mag1c_sample = get_images_from_id_for_display(selected_id, n_swir = int(backup_config['n_swir']))

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
    
    # --- Configuration ---
    num_folds = 5
    accumulated_preds = None

    input_img, gt = get_images_from_id_for_inference(selected_id, n_swir = int(backup_config['n_swir'] ))
    input_tensor = input_img.unsqueeze(0).to("cpu") 

    with st.spinner(f"🧠 Analyse en cours par l'ensemble des {num_folds} modèles..."):
        with torch.no_grad():
            for fold in range(num_folds):
                current_model_path = os.path.join(
                    'results', 
                    f'results_{selected_modele_name}', 
                    f'results_{selected_experiment}', 
                    'models', 
                    f'best_{selected_modele_name}_fold_{fold}.pth'
                )

                model = load_methane_model(selected_modele_name, current_model_path, in_channels = backup_config['n_swir'] + 4, device="cpu")
                model.eval()
                
                output = model(input_tensor)
                prob = torch.sigmoid(output).cpu().numpy().squeeze()
                

                if accumulated_preds is None:
                    accumulated_preds = prob
                else:
                    accumulated_preds += prob
                
                del model 
        
        avg_pred = accumulated_preds / num_folds

        pred_binary = (avg_pred > threshold).astype(np.uint8)

        tp, fp, fn, tn = smp.metrics.get_stats(torch.tensor(pred_binary).int(), gt.int(), mode='binary', threshold=0.5)

        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        gt = gt.cpu().numpy().squeeze()


    tab1, tab2 = st.tabs(["📊 Analyse Superposée", "🔍 Comparaison Côte-à-Côte"])

    with tab1:
        st.subheader(f"Superposition des performances sur le terrain, IOU SCORE: {iou_score}")
        
        empty_l, col_img, empty_r = st.columns([1, 2, 1])

        with col_img:
            fig = plot_overlay_analysis(rgb_sample, pred_binary, gt)
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
            st.image(pred_binary * 255, caption="Prédiction du modèle", use_container_width=True)

        with col_b:
            st.image(gt, caption="Ground Truth (Réalité)", use_container_width=True)
    with st.expander("ℹ️ Informations techniques sur le modèle"):
        st.write(f"Modèle : {selected_modele_name} | Nombre de folds: 5 ")
else:
    st.info("Sélectionnez un événement et cliquez sur 'Lancer la détection' pour voir le résultat.")