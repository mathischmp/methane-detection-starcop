import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import rasterio
from methan_detection import visionDataTransformer
@st.cache_data
def load_test_metadata():
    test_data = os.path.join('data', 'STARCOP_test', 'test.csv')
    return pd.read_csv(test_data)

def preprocess_for_inference(rgb, mag1c):
    """Prépare les données pour le modèle (Concaténation + Tenseur)."""
    rgb = rgb.astype(np.float32) / 255.0
    mag1c = np.expand_dims(mag1c, axis=-1).astype(np.float32)
    
    combined = np.concatenate([rgb, mag1c], axis=-1)
    tensor = torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0)
    return tensor

def get_rasterio_image(selected_id):
    ft = os.path.join('data', 'STARCOP_test', selected_id)

    aviris_r = os.path.join(ft, "TOA_AVIRIS_640nm.tif")
    aviris_g = os.path.join(ft, "TOA_AVIRIS_550nm.tif")
    aviris_b = os.path.join(ft, "TOA_AVIRIS_460nm.tif")
    magic_path = os.path.join(ft, "mag1c.tif")
    gt_path = os.path.join(ft, "labelbinary.tif")

    with rasterio.open(gt_path) as src:
        size_read = 300
        # Compute shape to read to from pyramids and speed up plotting
        shape = src.shape
        if (size_read >= shape[0]) and (size_read >= shape[1]):
            out_shape = shape
        elif shape[0] > shape[1]:
            out_shape = (size_read, int(round(shape[1]/shape[0] * size_read)))
        else:
            out_shape = (int(round(shape[0] / shape[1] * size_read)), size_read)
        gt = src.read(1, out_shape=out_shape)
    
    with rasterio.open(magic_path) as src:
        mag1c = src.read(1, out_shape=out_shape)
    with rasterio.open(aviris_r) as src:
        r = src.read(1, out_shape=out_shape)
    with rasterio.open(aviris_g) as src:
        g = src.read(1, out_shape=out_shape)
    with rasterio.open(aviris_b) as src:
        b = src.read(1, out_shape=out_shape)

    return r,g,b, mag1c, gt


def get_images_from_id_for_display(selected_id):
    """Charge les images RGB et MAG1C à partir de l'ID sélectionné."""
    
    r, g, b, mag1c, gt = get_rasterio_image(selected_id)
    rgb = np.asarray([r,g,b])
    rgb = np.clip(rgb/60., 0, 1) # Limite les valeurs extrêmes
    rgb = np.transpose(np.asanyarray(rgb),(1,2,0)) # [H, W, 3]

    mag1c_display = (mag1c - mag1c.min()) / (mag1c.max() - mag1c.min() + 1e-8)
    
    return rgb, mag1c_display

def get_images_from_id_for_inference(selected_id):

    r, g, b, mag1c, gt = get_rasterio_image(selected_id)
    vtransformer = visionDataTransformer.VisionDataTransformer()

    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    mag1c = np.asarray(mag1c).astype(np.float32)
    gt = np.asarray(gt).astype(np.float32)
    
    rgb, mag1c, gt = vtransformer.transform_for_validation(rgb, mag1c, gt)
    input = torch.cat([rgb, mag1c], dim=0)

    print(f"Input shape for inference: {input.shape}, GT shape: {gt.shape}")

    return input, gt

def get_rgb_stacked(selected_id):
    r, g, b, _, _ = get_rasterio_image(selected_id)
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    return rgb