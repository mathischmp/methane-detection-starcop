import streamlit as st
import torch
import segmentation_models_pytorch as smp
from methan_detection.models import EfficientNetV2, MiT, ConvNext # Importe ta classe de modèle

@st.cache_resource
def load_methane_model(name, checkpoint_path, device="cpu"):
    """Charge un modèle depuis un checkpoint et le garde en cache."""
    
    if name == "EfficientNetV2":
        model = EfficientNetV2(num_classes=1, pretrained=True, in_channels=4)
    elif name == "MiT":
        model = MiT(num_classes=1, pretrained=True, in_channels=4)
    elif name == "ConvNext":
        model = ConvNext(num_classes=1, pretrained=True, in_channels=4)
    else:
        raise ValueError(f"Model name {name} is not supported.")

    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()
    return model

def get_prediction(model, inputs, device="cpu"):
    """Effectue l'inférence sur un tenseur [1, 4, H, W]"""
    with torch.no_grad():
        logits = model(inputs.to(device))
        probs = torch.sigmoid(logits)
    return probs.cpu().numpy().squeeze() # Retourne une carte [H, W]