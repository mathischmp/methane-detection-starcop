from .dataset import Dataset
import os
from methan_detection.utils import setup_model, load_config
import pandas as pd
from torch.utils.data import DataLoader
import torch
import json

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import segmentation_models_pytorch as smp

class ModelTester:
    
    def __init__(self, model_name, num_xp):
        self.model_name = model_name
        self.num_xp = num_xp
        self.config = load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.test_data_folder = os.path.join('..', self.config['storage']['local_raw_path'], 'STARCOP_test')
        self.model_folder = os.path.join('..', self.config['storage']['local_results_path'], f'results_{self.model_name}', f'results_xp_{self.num_xp}', 'models')
        

        json_path = os.path.join('..', self.config['storage']['local_results_path'], f'results_{self.model_name}', f'results_xp_{self.num_xp}', 'logs', 'config_backup.json')

        with open(json_path, 'r') as f:
            self.backup_config = json.load(f)   


    def evaluate(self, n_visualize):
        # Placeholder for evaluation logic
        # This should include code to run the model on the test data and calculate metrics

        df_test = self.load_test_csv()
        test_dataset = Dataset(config=self.config, labels=df_test, training_type=None, transform=False, test=True)
        test_loader = DataLoader(test_dataset, batch_size=self.backup_config['batch_size'], shuffle=False, num_workers=self.backup_config['num_workers'])
        accumulated_probs = [0] * len(test_loader)

        for n_fold in range(self.backup_config['n_folds']):
            model = self.load_model_from_fold_number(n_fold)
            with torch.no_grad():
                for i, (rgb, mag1c, gt, qplume) in enumerate(test_loader):

                    match self.backup_config['method']:
                        case "concat": 
                            if mag1c.dim() == 3: 
                                mag1c = mag1c.unsqueeze(1)
                            inputs = torch.cat([rgb, mag1c], dim=1).to(self.device)
    
                    pred = model(inputs)
                    probs = torch.sigmoid(pred).cpu()
                    accumulated_probs[i] += probs
            del model
            torch.cuda.empty_cache()

        final_masks = [(p / self.backup_config['n_folds'] > 0.5).int() for p in accumulated_probs]
        gt_masks = self.get_all_ground_truths(test_loader)
        gt_masks = gt_masks.int()

        preds_final = torch.cat(final_masks, dim=0).squeeze(1)
        
        mean_iou_score = self.compute_final_iou_score(preds_final, gt_masks)
        
        print(f"Final Mean IoU Score: {mean_iou_score:.4f}")

        self.visualize_methane_errors(preds_final, gt_masks, n_visualize)

        pass

    def load_test_csv(self):
        csv_path = os.path.join(self.test_data_folder, 'test.csv')
        print(f"Loading test CSV file from {csv_path}...")
        df = pd.read_csv(csv_path)
        return df
    
    def load_model_from_fold_number(self, n_fold):

        model = setup_model(model_type=self.model_name)
        path = os.path.join(self.model_folder, f'best_{self.model_name}_fold_{n_fold}.pth')
        model.load_state_dict(torch.load(path, weights_only=True))
        model.to(self.device)
        model.eval()
        
        return model
    
    def get_all_ground_truths(self, test_loader):
        all_gts = []
        for rgb, mag1c, gt, qplume in test_loader:
            all_gts.append(gt.cpu())
        
        return torch.cat(all_gts, dim=0)

    def compute_final_iou_score(self, preds, gts):
        """
        preds: Tenseur binaire [N, H, W] ou [N, 1, H, W]
        gts: Tenseur binaire [N, H, W] ou [N, 1, H, W]
        """

        #preds = preds.float().view(preds.size(0), -1)
        #gts = gts.float().view(gts.size(0), -1)

        tp, fp, fn, tn = smp.metrics.get_stats(preds, gts, mode='binary', threshold=0.5)

        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        
        return iou_score
    

    def visualize_methane_errors(self, preds, gts, n):
        """
        Affiche n échantillons aléatoires avec les TP, TN, FP, et FN colorés.
        preds, gts : Tenseurs ou arrays binaires [N, H, W]
        """
        # 1. Sélection aléatoire des indices
        indices = random.sample(range(len(preds)), n)
        
        colors = ["#4b4b4b", '#e74c3c', '#f1c40f', '#2ecc71']
        cmap = ListedColormap(colors)
        
        fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))
        if n == 1: axes = [axes]

        for i, idx in enumerate(indices):
            p = preds[idx].squeeze().numpy() if torch.is_tensor(preds) else preds[idx]
            g = gts[idx].squeeze().numpy() if torch.is_tensor(gts) else gts[idx]
            
            error_map = np.zeros_like(p, dtype=int)
            
            error_map[(p == 1) & (g == 0)] = 1  # False Positive 
            error_map[(p == 0) & (g == 1)] = 2  # False Negative 
            error_map[(p == 1) & (g == 1)] = 3  # True Positive 
            

            iou = self.compute_final_iou_score(preds[idx], gts[idx])

            # 5. Affichage
            im = axes[i].imshow(error_map, cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
            axes[i].set_title(f"Sample {idx} | IoU: {iou:.3f}", fontsize=12, pad=10)
            axes[i].axis('off')

        # 6. Légende commune
        legend_labels = {
            'True Negative (Fond)': colors[0],
            'False Positive (Alarme inutile)': colors[1],
            'False Negative (Manqué)': colors[2],
            'True Positive (Détecté)': colors[3]
        }
        patches = [mpatches.Patch(color=v, label=k) for k, v in legend_labels.items()]
        fig.legend(handles=patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), frameon=True)
        
        plt.tight_layout()
        plt.show()