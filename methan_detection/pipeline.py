import pandas
import yaml
import gdown
import os
import zipfile
from tqdm import tqdm
    
class Pipeline:
    
    def __init__(self, training_type):
        
        assert training_type in ["easy", "hard"]

        self.config = self.load_config()
        self.training_type = training_type
   

    def donwload_data_from_drive(self):
        if self.training_type == "easy":
            url = self.config["storage"]["drive_data_easy_train"]
        else:
            None #later
 
        output = os.path.join(self.config['storage']['local_raw_path'],  f'starcop_train_{self.training_type}.zip')

        if not os.path.exists(output):
                gdown.download(url, output, quiet=False)
                print("Téléchargement terminé !")
        else:
                print("Les données sont déjà présentes.")
        return None

    
    def load_config(self, config_path="methan_detection/config.yaml"):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def load_data(self):
         dataset_folder = self.config['storage']['local_raw_path']
         files_to_extract = os.path.join(self.config['storage']['local_raw_path'], f'starcop_train_{self.training_type}.zip')
         for zip_files in tqdm(files_to_extract):
            with zipfile.ZipFile(zip_files, "r") as zip_ref:
                zip_ref.extractall(dataset_folder)
                zip_ref.close()
         

    