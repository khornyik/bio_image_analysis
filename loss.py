import torch 
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, weight, size_average):
        super(DiceBCELoss, self).__init__()

        self.weight = weight
        self.size_average = size_average

    def forward(self, pred, target, smooth):

        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()

        diceloss = 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

        bce = F.binary_cross_entropy(
            pred, target, weight = self.weight,
              size_average = self.size_average, reduction = 'mean')
        
        dice_bce = diceloss + bce
        
        return dice_bce






class TotalLoss(nn.Module):
    def __init__(self, coef_nuclei, coef_membrane):
        super().__init__()

        self.ceof_nucelei = coef_nuclei
        self.coef_membrane = coef_membrane

    def forward(self, z, pred_nuclei, pred_membrane, target, mask_nuclei, mask_membrane):

        z_membrane = z[mask_membrane == 1]
        z_background = z[mask_membrane == 0]


        return z_membrane, z_background