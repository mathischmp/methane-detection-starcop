import torch
import torch.nn as nn
import torch.nn.functional as F

class MethaneDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        smooth: petite valeur pour éviter la division par zéro 
        si le masque et la prédiction sont vides.
        """
        super(MethaneDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)
        
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        cardinal = probs.sum(dim=1) + targets.sum(dim=1)

        dice_coeff = (2. * intersection + self.smooth) / (cardinal + self.smooth)
        
        return 1 - dice_coeff.mean()

class MethaneCombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.5):
        super().__init__()
        self.dice = MethaneDiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_dice = weight_dice

    def forward(self, logits, targets):
        # targets doit être de type float pour BCE
        targets = targets.float() 
        
        loss_dice = self.dice(logits, targets)
        loss_bce = self.bce(logits, targets)
        
        return (self.weight_dice * loss_dice) + ((1 - self.weight_dice) * loss_bce)