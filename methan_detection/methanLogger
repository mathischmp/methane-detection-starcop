import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import shutil

class MethaneLogger:
    def __init__(self, config, base_path="outputs/runs"):
        # Création d'un ID unique pour l'expérience
        timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%M")
        exp_name = config.get("project", {}).get("experiment_name", "exp")
        self.run_dir = os.path.join(base_path, f"{exp_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Sauvegarde de la config pour la postérité
        with open(os.path.join(self.run_dir, "config_backup.json"), "w") as f:
            json.dump(config, f, indent=4)
            
        self.history = [] # Liste de dictionnaires pour le résumé global
        self.current_fold = None
        self.fold_dir = None

    def set_fold(self, fold_idx):
        """Prépare le logger pour un nouveau pli de la Cross-Val."""
        self.current_fold = fold_idx
        self.fold_dir = os.path.join(self.run_dir, f"fold_{fold_idx}")
        os.makedirs(os.path.join(self.fold_dir, "samples"), exist_ok=True)
        self.fold_metrics = []

    def log_metrics(self, epoch, train_loss, val_loss, val_iou):
        """Enregistre les métriques d'une époque."""
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou
        }
        self.fold_metrics.append(metrics)
        print(f"[Fold {self.current_fold} | Epoch {epoch}] Loss: {val_loss:.4f} | IoU: {val_iou:.4f}")

    def save_checkpoint(self, model, is_best=False):
        """Sauvegarde les poids du modèle."""
        name = "best_model.pth" if is_best else "last_model.pth"
        path = os.path.join(self.fold_dir, name)
        torch.save(model.state_dict(), path)

    def save_sample_plot(self, epoch, image, mask, prediction):
        """
        Enregistre une image comparative (RGB, Ground Truth, Prediction).
        Utile pour voir si le modèle détecte bien la forme du panache.
        """
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(image.permute(1, 2, 0).cpu().numpy()[:,:,:3]) # RGB
        ax[0].set_title("Input RGB")
        ax[1].imshow(mask.squeeze().cpu().numpy(), cmap='gray')
        ax[1].set_title("Ground Truth")
        ax[2].imshow(prediction.squeeze().cpu().numpy(), cmap='magma')
        ax[2].set_title("Prediction")
        
        plt.savefig(os.path.join(self.fold_dir, "samples", f"epoch_{epoch}.png"))
        plt.close()

    def finish_fold(self):
        """Génère les graphes et sauvegarde les stats du pli à la fin de l'entraînement."""
        df = pd.DataFrame(self.fold_metrics)
        df.to_csv(os.path.join(self.fold_dir, "metrics.csv"), index=False)
        
        # Plotting Loss
        plt.figure(figsize=(10, 5))
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
        plt.title(f"Loss Curve - Fold {self.current_fold}")
        plt.legend()
        plt.savefig(os.path.join(self.fold_dir, "loss_curve.png"))
        plt.close()

        # On garde la meilleure métrique du pli pour le résumé global
        best_val_iou = df['val_iou'].max()
        self.history.append({"fold": self.current_fold, "best_val_iou": best_val_iou})

    def finalize_global_report(self):
        """Calcule la performance moyenne sur tous les plis."""
        df_global = pd.DataFrame(self.history)
        mean_iou = df_global['best_val_iou'].mean()
        std_iou = df_global['best_val_iou'].std()
        
        summary = f"CV Performance: {mean_iou:.4f} (+/- {std_iou:.4f})"
        with open(os.path.join(self.run_dir, "global_summary.txt"), "w") as f:
            f.write(summary)
            f.write("\n\nFull History:\n")
            f.write(df_global.to_string())
            
        print("\n" + "="*30)
        print(f"EXPERIMENT FINISHED: {summary}")
        print("="*30)