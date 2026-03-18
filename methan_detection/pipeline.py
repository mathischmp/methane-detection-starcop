import pandas as pd
import yaml
import gdown
import os
import zipfile
from tqdm import tqdm
import rasterio
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from .utils import setup_model, load_config

from methan_detection import models
from .trainer import Trainer

class Pipeline:
    
    def __init__(self, training_type):
        
        assert training_type in ["easy", "hard"]

        self.config = load_config()
        self.training_type = training_type
        self.n_xp = self.config['training']['num_xp']


    def run(self):
        #self.donwload_data_from_drive()
        #if self.config['training']['load_zip_data']:
            #self.load_data()
        df = self.load_csv()
        df = self.create_folds(df)
        trainer = Trainer(df=df, num_xp=self.n_xp)
        trainer.run()
        return None
   

    def donwload_data_from_drive(self):
        if self.training_type == "easy":
            url = self.config["storage"]["drive_data_easy_train"]
        else:
            None #later
 
        output = os.path.join('..', self.config['storage']['local_raw_path'],  f'starcop_train_{self.training_type}.zip')

        if not os.path.exists(output):
                gdown.download(url, output, quiet=False)
                print("Téléchargement terminé !")
        else:
                print("Les données sont déjà présentes.")
        return None
    
    def load_data(self):
        dataset_folder = self.config['storage']['local_raw_path']
        zip_file = os.path.join('..', dataset_folder, f'STARCOP_train_{self.training_type}.zip')
        
        if os.path.exists(zip_file):
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(os.path.join('..', dataset_folder))
                zip_ref.close()
            #os.remove(zip_file)
        else:
            raise FileNotFoundError(f"Zip file not found at {zip_file}")
        
        return dataset_folder
    
    def load_csv(self):
        csv_path = os.path.join('..', self.config['storage']['local_raw_path'], f'STARCOP_train_{self.training_type}', f'train_{self.training_type}.csv')
        print(f"Loading CSV file from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        return df
    

    def create_folds(self, df: pd.DataFrame):
        n_folds = self.config['training']['n_folds']
        df = df.copy()
        
        seuil_95 = df['qplume'].quantile(0.95)
        df = df[df['qplume'] <= seuil_95].copy()
        df = df.reset_index(drop=True)
        
        is_positive = df['qplume'] > 0
        
        num_bins =int(np.floor(1 + np.log2(len(is_positive))))
        print(f"Using {num_bins} bins based on Sturges' formula")

        df['total_bin'] = 0

        if is_positive.any():
            df.loc[is_positive, 'total_bin'] = pd.cut(
                df.loc[is_positive, 'qplume'], 
                bins=num_bins, 
                labels=False, 
                duplicates='drop'
            ) + 1

        df['date'] = pd.to_datetime(df['date'])
        df["date"] = df["date"].dt.day
        
        X = df['id']
        Y = df['total_bin']
        groups = df['date']
        
        df['fold'] = -1
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y, groups)):
            df.loc[val_idx, 'fold'] = fold
        
        print("\nFold distribution:")
        print(df['fold'].value_counts().sort_index())

        return df
         
    