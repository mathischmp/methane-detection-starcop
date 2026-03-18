from methan_detection import models
import os
import zipfile
import yaml

def setup_model(model_type="EfficientNetV2", in_channels =4):
    
    match model_type:
        case "EfficientNetV2":
            model = models.EfficientNetV2(num_classes=1, pretrained=True, in_channels=in_channels)
        case "MiT":
            model = models.MiT(num_classes=1, pretrained=True, in_channels=in_channels)
        case "ConvNext":
            model = models.ConvNext(num_classes=1, pretrained=True, in_channels=in_channels)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")
    return model

def load_test_data(config):
    dataset_folder = config['storage']['local_raw_path']
    zip_file = os.path.join('..', dataset_folder, f'STARCOP_test.zip')
    
    if os.path.exists(zip_file):
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(os.path.join('..', dataset_folder))
            zip_ref.close()
        os.remove(zip_file)
    else:
        raise FileNotFoundError(f"Zip file not found at {zip_file}")
    
    return dataset_folder

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "config.yaml"
        )
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config