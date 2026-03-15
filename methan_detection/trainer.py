from methan_detection import models
from methan_detection import visionDataTransformer
import torch
import yaml
import torch.optim as optim
from methan_detection import dice_loss
import os
from .dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import math
from tqdm.auto import tqdm
import numpy as np
from .methaneLogger import MethaneLogger
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss, FocalLoss, SoftBCEWithLogitsLoss
import segmentation_models_pytorch as smp

class Trainer:
    
    def __init__(self, model, df, num_xp = 1):
        self.config = self.load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.data_transformer = visionDataTransformer.VisionDataTransformer()   
        self.df = df
        self.method = self.config['training']['method']
        self.logger = MethaneLogger(self.config, self.model.get_name(), num_xp)
        
        self.result_folder = os.path.join('..', self.config['storage']['local_results_path'], f'results_{self.model.get_name()}')
        os.makedirs(self.result_folder, exist_ok=True)

        self.result_folder = os.path.join(self.result_folder, f'results_xp_{num_xp}')
        os.makedirs(self.result_folder, exist_ok=True)

        self.result_folder = os.path.join(self.result_folder, 'models')
        os.makedirs(self.result_folder, exist_ok=True)

        if self.config['training']['loss'] == "dice":
            self.criterion = DiceLoss(mode='binary')
        elif self.config['training']['loss'] == "combined":
            self.criterion = dice_loss.MethaneCombinedLoss()
        
    def load_config(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "config.yaml"
            )
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def dice_coef_torch(self, pred, gt, smooth=1e-6):
        #On GPU
        pred = (torch.sigmoid(pred) > 0.5).float()
        gt = gt.float()
        
        intersection = (pred * gt).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)
        
        return dice.item()
    
    def compute_iou_score(self, pred, gt):
        pred = (torch.sigmoid(pred) > 0.5).int()
        gt = gt.int()
        tp, fp, fn, tn = smp.metrics.get_stats(pred, gt, mode='binary', threshold=0.5)

        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        
        return iou_score
    
    def get_train_valid_from_fold(self, df, n_fold : int):
        assert n_fold <= self.config['training']['n_folds']

        train = df[df['fold'] != n_fold].reset_index(drop = True)
        valid = df[df['fold'] == n_fold].reset_index(drop = True)

        return Dataset(self.config, train, "easy", transform = True), Dataset(self.config, valid, "easy", transform = False)


    def train_one_epoch(self, train_loader, optimizer):
        total_loss = 0
        optimizer.zero_grad()
        self.model.smp_model.train()

        for rgb,mag1c,gt,qplume in train_loader:

            match self.method:
                case "concat": 
                    if mag1c.dim() == 3: 
                         mag1c = mag1c.unsqueeze(1)
                    inputs = torch.cat([rgb, mag1c], dim=1).to(self.device)
            
            gt = gt.to(self.device)

            pred = self.model(inputs).squeeze(1)
            loss = self.criterion(pred, gt)
            
            loss.backward()
            total_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

        return total_loss / len(train_loader)


    def validate_one_epoch(self, valid_loader):
        total_loss = 0
        self.model.smp_model.eval()
        iou_score = 0

        with torch.no_grad():
            for rgb,mag1c,gt,qplume in valid_loader:
                match self.method:
                    case "concat": 
                        if mag1c.dim() == 3: 
                            mag1c = mag1c.unsqueeze(1)
                        inputs = torch.cat([rgb, mag1c], dim=1).to(self.device)
                gt = gt.to(self.device)

                pred = self.model(inputs).squeeze(1)
                loss = self.criterion(pred, gt)
                total_loss += loss.item()
                iou_score += self.compute_iou_score(gt, pred)
        
        return total_loss / len(valid_loader), iou_score / len(valid_loader)



    def validate_one_fold(self, df, n_fold : int): 
        best_validation_loss = float('inf')
        self.logger.set_fold(n_fold)

        model_dict_path = os.path.join(self.result_folder, f"best_{self.model.get_name()}_fold_{n_fold}.pth")

        train_dataset, valid_dataset = self.get_train_valid_from_fold(df, n_fold)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['training']['batch_size'], 
            shuffle=True, num_workers=self.config['training']['num_workers'],
            pin_memory=False)
        
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.
            config['training']['batch_size'], 
            shuffle=False, num_workers=self.config['training']['num_workers'],
            pin_memory=False)
        
        self.model.smp_model.train()
        
        decoder_lr = float(self.config['training']['unfreeze_lr'])
        encoder_lr = decoder_lr * 0.1
        

        head_params = list(self.model.smp_model.decoder.parameters()) + list(self.model.smp_model.segmentation_head.parameters())
        wd = float(self.config['training']['weight_decay'])
        optimizer = optim.Adam([
            {'params': self.model.smp_model.encoder.parameters(), 'lr': encoder_lr, 'weight_decay': wd}, 
            {'params': head_params, 'lr': decoder_lr, 'weight_decay': wd}
        ]
        )
        
        for param in self.model.smp_model.encoder.parameters():
            param.requires_grad = False


        scheduler = self.methan_scheduler(optimizer, int(self.config['training']['n_total_epochs']))

        print(f'\n=== Fold {n_fold} | Phase 1: Encoder Frozen ===')

        n_epochs = self.config['training']['n_epochs_before_unfreeze']
        pbar = tqdm(range(n_epochs), desc=f"Fold {n_fold}", unit="epoch")

        for n in pbar:
            train_loss = self.train_one_epoch(train_loader, optimizer)
            val_loss, iou_score = self.validate_one_epoch(valid_loader)
            scheduler.step()
            pbar.set_postfix({
                "T-Loss": f"{train_loss:.4f}",
                "V-Loss": f"{val_loss:.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.1e}"
            })
            pbar.update(1)

            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                torch.save(self.model.state_dict(), model_dict_path)
            
            self.logger.log_metrics(epoch=n, train_loss=train_loss, val_loss=val_loss, iou_score=iou_score)

        for param in self.model.smp_model.encoder.parameters():
            param.requires_grad = True

        print(f'\n=== Fold {n_fold} | Phase 2: Encoder Unfrozen ===')

        n_epochs = self.config['training']['n_total_epochs'] - self.config['training']['n_epochs_before_unfreeze']
        pbar = tqdm(range(n_epochs), desc=f"Fold {n_fold}", unit="epoch")

        for n in pbar:
            train_loss = self.train_one_epoch(train_loader, optimizer)
            val_loss, dice_score = self.validate_one_epoch(valid_loader)
            scheduler.step()
            pbar.set_postfix({
                "T-Loss": f"{train_loss:.4f}",
                "V-Loss": f"{val_loss:.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.1e}"
            })
            pbar.update(1)

            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                torch.save(self.model.state_dict(), model_dict_path)
            
            self.logger.log_metrics(epoch=n + self.config['training']['n_epochs_before_unfreeze'], train_loss=train_loss, val_loss=val_loss, iou_score=dice_score)
        
        self.logger.finish_fold()
        return None


    def methan_scheduler(self, optimizer, max_epochs):
        def lr_lambda(epoch):
            e = max(0, epoch)
            warmup_epochs = self.config['training']['warmup_epochs']
            if e < warmup_epochs:
                return float(e + 1) / float(max(1, warmup_epochs))
            progress = (e - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
            progress = min(1.0,progress)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def run(self):
        print('Starting training...')
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
            self.model = nn.DataParallel(self.model)
        
        for n_fold in range(self.config['training']['n_folds']):
            self.validate_one_fold(self.df, n_fold)
        
        self.logger.finalize_global_report()
        return None
