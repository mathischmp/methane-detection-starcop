import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def plot_overlay_analysis(rgb, pred, gt, alpha=0.5):
    """
    rgb: image [H, W, 3] normalisée
    pred, gt: masques binaires [H, W]
    """
    # 0: TN, 1: FP, 2: FN, 3: TP
    error_map = np.zeros_like(pred, dtype=int)
    error_map[(pred == 1) & (gt == 0)] = 1  # Rouge
    error_map[(pred == 0) & (gt == 1)] = 2  # Jaune
    error_map[(pred == 1) & (gt == 1)] = 3  # Vert

    colors = [
        (0, 0, 0, 0.0),    # TN : Totalement Transparent
        (1, 0, 0, alpha),  # FP : Rouge semi-transparent
        (1, 0.8, 0, alpha),# FN : Jaune semi-transparent
        (0, 1, 0, alpha)   # TP : Vert semi-transparent
    ]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(2, 2))
    
    ax.imshow(rgb)
    
    ax.imshow(error_map, cmap=cmap, interpolation='nearest')
    
    ax.axis('off')
    return fig