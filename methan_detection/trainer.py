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

class Trainer:
    
    def __init__(self, model, df):
        self.config = self.load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.data_transformer = visionDataTransformer.VisionDataTransformer()   
        self.df = df

        if self.config['training']['loss'] == "dice":
            self.criterion = dice_loss.MethaneDiceLoss()
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
    
    
    def get_train_valid_from_fold(self, df, n_fold : int):
        assert n_fold <= self.config['training']['n_folds']

        train = df[df['fold'] != n_fold].reset_index(drop = True)
        valid = df[df['fold'] == n_fold].reset_index(drop = True)

        return Dataset(self.config, train, "easy", transform = True), Dataset(self.config, valid, "easy", transform = False)


    def train_one_epoch(self, train_loader, optimizer):
        total_loss = 0
        optimizer.zero_grad()

        method = self.config['training']['method']


        for r,g,b,mag1c,gt,qplume in train_loader:

            match method:
                case "stacking": 
                    inputs = torch.stack([r,g,b,mag1c], dim=1).to(self.device)
            
            gt = gt.to(self.device)

            pred = self.model(inputs).squeeze(1)
            loss = self.criterion(pred, gt)
            loss.backward()
            total_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

        return total_loss / len(train_loader)


    def validate_one_fold(self, df, n_fold : int): 

        train_dataset, valid_dataset = self.get_train_valid_from_fold(df, n_fold)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['training']['batch_size'], 
            shuffle=True, num_workers=self.config['training']['num_workers'],
            pin_memory=True)
        
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.
            config['training']['batch_size'], 
            shuffle=False, num_workers=self.config['training']['num_workers'],
            pin_memory=True)
        
        self.model.smp_model.train()
        for param in self.model.smp_model.encoder.parameters():
            param.requires_grad = False
        
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(self.config['training']['head_lr']),
            weight_decay = float(self.config['training']['weight_decay'])
        )

        scheduler = self.methan_scheduler(optimizer, int(self.config['training']['n_total_epochs']))

        print(f'\n=== Fold {n_fold} | Phase 1: Backbone Frozen ===')

        for n in range(self.config['training']['n_epochs_before_unfreeze']):
            train_loss = self.train_one_epoch(train_loader, optimizer)
            
            #val_loss, val_iou = self.validate(valid_loader)
            scheduler.step()
            print(f"Epoch {n} | Train Loss: {train_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")


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
        
        return None
