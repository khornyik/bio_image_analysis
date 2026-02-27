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
    def __init__(self, weight: torch.tensor = None, size_average: bool = None):
        '''
        Initialises the DiceBCE loss.

        Parameters:
            weight (torch.tensor): manual overwrite to weights. None keeps default weights. 
            size_average (bool): take size average.
        '''
        super(DiceBCELoss, self).__init__()

        self.weight = weight
        self.size_average = size_average

    def forward(self, pred: torch.tensor = None, target: torch.tensor = None, smooth: int = 1):
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
    '''
    Calculates the total loss by separately finding a loss for both the nuclei and cell membranes.
    Both use DiceBCE loss as a base, and then adds small differences for each: taking into
    consideration latent smoothness for the nuclei, and a topology of the cell membranes.

    Attributes:
    ------------
    coef_nuclei: scaling coefficient for the latent smoothness loss for nuclei.
    coef_membrane: scaling coefficient for the topological loss for cell membranes.

    '''
    def __init__(self, coef_nuclei: float = 0.2, coef_membrane: float = 0.2):
        '''
        Initialises the total loss.

        Parameters:
            coef_nuclei (float): scaling coefficient for the latent smoothness loss for nuclei.
            coef_membrane (float): scaling coefficient for the topological loss for cell membranes.
        '''
        super().__init__()

        self.coef_nuclei = coef_nuclei
        self.coef_membrane = coef_membrane

    def latent_smoothness(self, z: torch.tensor = None):
        '''
        Gives the latent smoothness for the nuclei.

        Parameters:
            z (torch.tensor): tensor to calculate its latent smoothness.

        Returns:
            (float): latent smoothness of the input tensor.
        '''
        return  ((z[...,1:] - z[...,:-1])**2).mean()


    def forward(self, z: torch.tensor = None, pred_nuclei: torch.tensor = None, pred_membrane: torch.tensor = None, target: torch.tensor = None, mask_nuclei: torch.tensor = None, mask_membrane: torch.tensor = None, loss_func = None):
        '''
        Computation at call.

        Parameters:
            z (torch.tensor): tensor from Unet model before decodings.
            pred_nuclei (torch.tensor): prediction of nuclei.
            pred_membrane (torch.tensor): prediction of cell membranes.
            target (torch.tensor): tensor being predicted by the model.
            mask_nuclei (torch.tensor): tensor of the mask for nuclei.
            mask_membrane (torch.tensor): tensor of the mask for the cell membranes.
            loss_func (Callable[float]): loss function which returns a float value as loss. 
        
        Returns:
            (float): total loss value.
        '''

        z_membrane = z[mask_membrane == 1]
        z_background = z[mask_membrane == 0]

        l_smoothness = self.latent_smoothness(z)

        loss_nuclei = loss_func(pred_nuclei, mask_nuclei) + self.coef_nuclei * l_smoothness

        loss_membrance = loss_func(pred_membrane, mask_membrane) # add more




        return 1.0