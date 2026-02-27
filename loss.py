import torch 
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    '''
    A class combining Dice loss and BCE loss into one call.

    Inherits from torch.nn.Module.

    Attributes:
    ------------
    weight: weights to be used for the BCE loss.
    size_average: size average to be used for the BCE loss.

    '''
    def __init__(self, weight, size_average):
        '''
        Initialises the DiceBCE loss.

        Parameters:
            weight (torch.tensor): manual overwrite to weights. None keeps default weights. 
            size_average (bool): take size average.
        '''
        super(DiceBCELoss, self).__init__()

        self.weight = weight
        self.size_average = size_average

    def forward(self, pred, target, smooth):
        '''
        Computation at call.

        Parameters:
            pred (torch.tensor): predicted tensor from model.
            target (torch.tensor): the tensor being predicted.
            smooth (float): constant for Dice loss.
        
        Returns:
            (float): loss of the summation of the Dice loss and BCE loss.
        '''

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

        self.coef_nuclei = coef_nuclei
        self.coef_membrane = coef_membrane

    def latent_smoothness(self, z):
        return  ((z[...,1:] - z[...,:-1])**2).mean()


    def forward(self, z, pred_nuclei, pred_membrane, target, mask_nuclei, mask_membrane, loss_func):

        z_membrane = z[mask_membrane == 1]
        z_background = z[mask_membrane == 0]

        l_smoothness = self.latent_smoothness(z)

        loss_nuclei = loss_func(pred_nuclei, mask_nuclei) + self.coef_nuclei * l_smoothness

        loss_membrance = loss_func(pred_membrane, mask_membrane) # add more




        return z_membrane, z_background