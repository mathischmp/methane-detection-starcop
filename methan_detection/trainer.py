import models
import visionDataTransformer
import torch
import yaml


class Trainer:
    
    def __init__(self, train_loader, val_loader):
        with open("config.yaml", 'r') as file:
            self.config = yaml.safe_load(file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.EfficientNetV2(num_classes=1, pretrained=True, in_channels=4).to(self.device)
        self.data_transformer = visionDataTransformer.VisionDataTransformer()   
        self.train_loader = train_loader
        self.val_loader = val_loader