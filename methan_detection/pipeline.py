import pandas
import yaml
import gdown
import os
import zipfile
from tqdm import tqdm
#from dataset import Dataset
import rasterio
import numpy as np
    
class Pipeline:
    
    def __init__(self, training_type):
        
        assert training_type in ["easy", "hard"]

        self.config = self.load_config()
        self.training_type = training_type
    

    def run(self):
        #self.donwload_data_from_drive()
        #self.load_data()
        df = self.load_csv()
        self.create_dataset(df)
        input_data = self.data_preprocessing(df)
        return input_data
   

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
        csv_path = os.path.join('..', self.config['storage']['local_raw_path'], f'STARCOP_train_{self.training_type}', f'train_{self.training_type}.csv')
        
        df = pandas.read_csv(csv_path)
        
        return df
    

    def data_preprocessing(self, df): 
        folder = os.path.join('..', self.config['storage']['local_raw_path'], f'STARCOP_train_{self.training_type}')
        size_read = 300
        
        input_data = []
        for idx, event_id in enumerate(list(df["id"])):
            ft = os.path.join(folder, event_id)
            aviris_r = os.path.join(ft, "TOA_AVIRIS_640nm.tif")
            aviris_g = os.path.join(ft, "TOA_AVIRIS_550nm.tif")
            aviris_b = os.path.join(ft, "TOA_AVIRIS_460nm.tif")
            magic_path = os.path.join(ft, "mag1c.tif")
            # Ground truth:
            gt_path = os.path.join(ft, "labelbinary.tif")

            with rasterio.open(gt_path) as src:
                width = src.width
                height = src.height

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
                magic = src.read(1, out_shape=out_shape)
            with rasterio.open(aviris_r) as src:
                r = src.read(1, out_shape=out_shape)
            with rasterio.open(aviris_g) as src:
                g = src.read(1, out_shape=out_shape)
            with rasterio.open(aviris_b) as src:
                b = src.read(1, out_shape=out_shape)

            row = [r, g, b, magic, gt]
            input_data.append(row)
                            
        return np.array(input_data)


    def create_dataset(self, df):

       
        None #later
         

    