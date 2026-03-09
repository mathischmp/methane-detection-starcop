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
    

    def run(self):
        self.donwload_data_from_drive()
        self.load_data()
        df = self.load_csv()
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

    
    def load_config(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "config.yaml"
            )
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def load_data(self):
        dataset_folder = self.config['storage']['local_raw_path']
        zip_file = os.path.join('..', dataset_folder, f'STARCOP_train_{self.training_type}.zip')
        
        if os.path.exists(zip_file):
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(os.path.join('..', dataset_folder))
                zip_ref.close()
            os.remove(zip_file)
        else:
            raise FileNotFoundError(f"Zip file not found at {zip_file}")
        
        return dataset_folder
    
    def load_csv(self):
        csv_path = os.path.join('..', self.config['storage']['local_raw_path'], f'train_{self.training_type}.csv')
        
        df = pandas.read_csv(csv_path)
        
        return df
         

    