import models
import visionDataTransformer
import torch
import yaml
import torch.optim as optim
import dice_loss

class Trainer:
    
    def __init__(self, model, dataset):
        with open("config.yaml", 'r') as file:
            self.config = yaml.safe_load(file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.data_transformer = visionDataTransformer.VisionDataTransformer()   
        self.dataset = dataset

        if self.config['training']['loss'] == "dice":
            self.criterion = dice_loss.MethaneDiceLoss()
        elif self.config['training']['loss'] == "combined":
            self.criterion = dice_loss.MethaneCombinedLoss()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['training']['learning_rate'])
    
    def train_one_epoch(self, train_loader):
        self.model.train()
        self.model.encoder.eval()
        total_loss = 0
        self.optimizer.zero_grad()



        return None


    def run(self):
        print('Starting training...')
        return None
