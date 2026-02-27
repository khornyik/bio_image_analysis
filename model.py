import torch 
import torch.nn as nn
import torch.nn.functional as F



class Conv(nn.Module):
    '''
    A double convolution layer to improve accuracy of model. 

    Inherits from torch.nn.Module.

    Attributes:
    ------------
    layer: double Conv2d layers, both activated by batchNorm2d and finally a ReLU. 

    '''
    def __init__(self, in_size: int = 0, out_size: int = 0):
        '''
        Initialises the double convolution layer.
        
        Parameters:
            in_size (int): number of channels of input. 
            out_size (int): number of channels after convolution.  
        '''
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace = True)
        )

    def forward(self, x: torch.tensor = None):
        '''
        Computation at call. 
        
        Parameters:
            x (torch.tensor): tensor to be processed. 

        Returns:
            (torch.tensor): processed tensor.
        '''
        return self.layer(x)
    


class Encoder(nn.Module):
    '''
    Encoder for images. 

    Inherits from torch.nn.Module. 

    Attributes:
    ------------

    layer1: first double convolution layer of encoder. 
    layer2: second double convolution layer of encoder.
    layer3: third double convolution layer of encoder.
    layer4: fourth double convolution layer of encoder.
    layer5: fifth double convolution layer of encoder.
    layer6: sixth double convolution layer of encoder.

    pool1: 2d max pooling over 3 planes.
    pool2: 2d max pooling over 3 planes.
    pool3: 2d max pooling over 3 planes.
    pool4: 2d max pooling over 3 planes.
    pool5: 2d max pooling over 3 planes.

    '''
    def __init__(self, in_size: int = 0):
        '''
        Initialises the encoder.

        Parameters:
            in_size (int): number of input channels.
        '''
        super().__init__()

        self.layer1 = Conv(in_size, 32)
        self.pool1 = nn.MaxPool2d(3)

        self.layer2 = Conv(32, 64)
        self.pool2 = nn.MaxPool2d(3)

        self.layer3 = Conv(64, 128)
        self.pool3 = nn.MaxPool2d(3)

        self.layer4 = Conv(128, 256)
        self.pool4 = nn.MaxPool2d(3)

        self.layer5 = Conv(256, 512)
        self.pool5 = nn.MaxPool2d(3)

        self.layer6 = Conv(512, 1024)

    def forward(self, x: torch.tensor = None):
        '''
        Applies double convolution with max poolings which are saved as skips for later use.
        
        Parameters:
            x (torch.tensor): tensor to be processed.

        Returns:
            z (torch.tensor): the fully processed tensor.
            (tuple): a tuple of the max poolings at each stage between the double convolutions.
        '''

        skip_1 = self.layer1(x)
        x = self.pool1(skip_1)

        skip_2 = self.layer2(x)
        x = self.pool2(skip_2)

        skip_3 = self.layer3(x)
        x = self.pool3(skip_3)

        skip_4 = self.layer4(x)
        x = self.pool4(skip_4)

        skip_5 = self.layer5(x)
        x = self.pool5(skip_5)

        z = self.layer6(x)

        return z, (skip_1, skip_2, skip_3, skip_4, skip_5)



class AttentionLayer(nn.Module):
    '''
    The attention layer of the UNet model. 

    Inherits from torch.nn.Module.

    Attributes:
    ------------
    w_g: gated short convolution block.
    w_x: liquid foundation model.
    psi: convolution down to 1 output channel. 

    '''
    def __init__(self, gating_channels: int = 0, lfm_channels: int = 0, int_channels: int = 0):
        '''
        Initialises the attention layer.
        
        Parameters:
            gating_channels (int): number of channels of the input of the gated short convolution layer.
            lfm_channels (int): number of channels of the input of the liquid foundation model.
            int_channels (int): output channels of the gated and lfm layers. 
        '''
        super().__init__()

        self.w_g = nn.Conv2d(gating_channels, int_channels, 1)
        self.w_x = nn.Conv2d(lfm_channels, int_channels, 1)
        self.psi = nn.Conv2d(int_channels, 1, 1)

    def forward(self, x: torch.tensor = None, g: torch.tensor = None):
        '''
        Computation at call.
        
        Parameters:
            x (torch.tensor): tensor to be processed.
            g (torch.tensor): tensor from a max pooling from encoder layer. 

        Returns:
            (torch.tensor): the input tensor scaled by psi - a tensor of zeroes and ones. 
        '''

        psi = F.relu(self.w_g(g) + self.w_x(x))
        psi = torch.sigmoid(self.psi(psi))

        return x * psi


class NonLocalLayer(nn.Module):
    '''
    The non-local layer of the UNet model. 

    Inherits from torch.nn.Module.

    Attributes:
    ------------
    theta: single convolution layer.
    phi: single convolution layer.
    g: single convolution layer.
    out: single convolution layer.

    '''
    def __init__(self, in_size: int = 0):
        '''
        Initialises the non-local layer.
        
        Inherits from torch.nn.Module.

        Parameters:
            in_size (int): size of the input. 
        '''
        super().__init__()

        self.theta = nn.Conv2d(in_size, in_size // 2, 1)
        self.phi = nn.Conv2d(in_size, in_size // 2, 1)
        self.g = nn.Conv2d(in_size, in_size // 2, 1)
        self.out = nn.Conv2d(in_size // 2, in_size, 1)

    def forward(self, x: torch.tensor = None):
        '''
        Computation at call. 
        
        Parameters:
            x (torch.tensor): tensor to be prcoessed. 

        Returns:
            (torch.tensor): initial tensor x translated by a tensor of constants. 
        '''

        b, c, h, w = x.shape 

        theta = self.theta(x).view(b, c // 2, -1)
        phi = self.phi(x).view(b, c // 2, -1)
        g = self.g(x).view(b, c // 2, -1)

        attention = torch.softmax(
            torch.bmm(g, theta.transpose(1, 2), phi), dim = -1
        )

        output = torch.bmm(g, attention.transpose(1, 2))
        output = output.view(b, c // 2, h, w)

        return x + self.out(output)


class NucleiDecoder(nn.Module):
    '''
    Decoder for the nuclei in images. 

    Inherits from torch.nn.Module.

    Attributes:
    ------------
    up1: fractionally-strided convolution layer.
    up2: fractionally-strided convolution layer.
    up3: fractionally-strided convolution layer.
    up4: fractionally-strided convolution layer.
    up5: fractionally-strided convolution layer.

    layer1: double convolution layer down from 1024 to 512 channels.
    layer2: double convolution layer down from 512 to 256 channels.
    layer3: double convolution layer down from 256 to 128 channels.
    layer4: double convolution layer down from 128 to 64 channels.
    layer5: double convolution layer down from 64 to 32 channels.

    output: single convolution layer down from 32 to 1 channels.
    
    '''
    def __init__(self):
        '''
        Initialises the decoder for nuclei.
        '''
        super().__init__()

        self.up5 = nn.ConvTranspose2d(1024, 512, 2, stride = 2)
        self.layer1 = Conv(1024, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride = 2)
        self.layer2 = Conv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride = 2)
        self.layer3 = Conv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride = 2)
        self.layer4 = Conv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride = 2)
        self.layer5 = Conv(64, 32)

        self.output = nn.Conv2d(32, 1, 1)


    def forward(self, z: torch.tensor = None, skips: tuple[torch.tensor] = None):
        '''
        Computation at call.

        Parameters:
            z (torch.tensor): the tensor to decode.
            skips (tuple[torch.tensor]): tuple of intermediate max poolings of different number 
                of channels from the encoder layer.
        
        Returns:
            (torch.tensor): estimation of nuclei in the original image. 
        '''

        skip_1, skip_2, skip_3, skip_4, skip_5 = skips

        x = self.up5(z)
        x = self.layer1(torch.cat([x, skip_5], dim = 1))

        x = self.up4(x)
        x = self.layer2(torch.cat([x, skip_4], dim = 1))

        x = self.up3(x)
        x = self.layer3(torch.cat([x, skip_3], dim = 1))

        x = self.up2(x)
        x = self.layer4(torch.cat([x, skip_2], dim = 1))

        x = self.up1(x)
        x = self.layer5(torch.cat([x, skip_1], dim = 1))

        return torch.sigmoid(self.output(x))



class MembraneDecoder(nn.Module):
    '''
    Decoder for cell membranes in the image.

    Inherits from torch.nn.Module.

    Attributes:
    ------------
    up1: fractionally-strided convolution layer.
    up2: fractionally-strided convolution layer.
    up3: fractionally-strided convolution layer.
    up4: fractionally-strided convolution layer.
    up5: fractionally-strided convolution layer.

    layer1: double convolution layer down from 1024 to 512 channels.
    layer2: double convolution layer down from 512 to 256 channels.
    layer3: double convolution layer down from 256 to 128 channels.
    layer4: double convolution layer down from 128 to 64 channels.
    layer5: double convolution layer down from 64 to 32 channels.

    att1: atention layer layer at 512 channels.
    att2: atention layer layer at 256 channels.
    att3: atention layer layer at 128 channels.
    att4: atention layer layer at 64 channels.
    att5:  atention layer layer at 32 channels.

    output: single convolution layer down from 32 to 1 channels.
    '''
    def __init__(self):
        '''
        Initialises the decoder for cell membranes.
        '''
        super().__init__()

        self.up5 = nn.ConvTranspose2d(1024, 512, 2, stride = 2)
        self.layer1 = Conv(1024, 512)
        self.att1 = AttentionLayer(512, 512, 256)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride = 2)
        self.layer2 = Conv(512, 256)
        self.att2 = AttentionLayer(256, 256, 128)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride = 2)
        self.layer3 = Conv(256, 128)
        self.att3 = AttentionLayer(128, 128, 64)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride = 2)
        self.layer4 = Conv(128, 64)
        self.att4 = AttentionLayer(64, 64, 32)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride = 2)
        self.layer5 = Conv(64, 32)
        self.att5 = AttentionLayer(32, 32, 16)

        self.output = nn.Conv2d(32, 1, 1)


    def forward(self, z: torch.tensor = None, skips: tuple[torch.tensor] = None):
        '''
        Computation at call.

        Parameters:
            z (torch.tensor): the tensor to decode.
            skips (tuple[torch.tensor]): tuple of intermediate max poolings of different number 
                of channels from the encoder layer.

        Returns:
            (torch.tensor): estimation of cell membranes in the original image.
        '''

        skip_1, skip_2, skip_3, skip_4, skip_5 = skips

        x = self.up5(z)
        skip_5 = self.att1(x, skip_5)
        x = self.layer1(torch.cat([x, skip_5], dim = 1))

        x = self.up4(x)
        skip_4 = self.att2(x, skip_4)
        x = self.layer2(torch.cat([x, skip_4], dim = 1))

        x = self.up3(x)
        skip_3 = self.att3(x, skip_3)
        x = self.layer3(torch.cat([x, skip_3], dim = 1))

        x = self.up2(x)
        skip_2 = self.att4(x, skip_2)
        x = self.layer4(torch.cat([x, skip_2], dim = 1))

        x = self.up1(x)
        skip_1 = self.att5(x, skip_1)
        x = self.layer5(torch.cat([x, skip_1], dim = 1))

        return torch.sigmoid(self.output(x))


class UNet(nn.Module):
    '''
    UNet model consisting of an encoder, bottleneck, non-local layer, and a decoder for the 
    nuclei and cell membranes.

    Inherits from torch.nn.Module.

    Attributes:
    ------------
    encoder: encoder for the model.
    bn: bottleneck for the model (single convolution from 512 to 1024 channels).
    non_local: non-local layer for model.
    d_membrane: decoder for the cell membranes.
    d_nuclei: decoder for the nuclei.

    '''
    def __init__(self):
        '''
        Initialises the UNet model.
        '''
        super().__init__()

        self.encoder = Encoder()
        self.bn = Conv(512, 1024)
        self.non_local = NonLocalLayer(1024)
        self.d_membrane = MembraneDecoder()
        self.d_nuclei = NucleiDecoder()

    def forward(self, x: torch.tensor = None):
        '''
        Computation at call.

        Parameters:
            x (torch.tensor): the image to segment.

        Returns:
            pred_nuclei (torch.tensor): prediciton of the nuclei in the original image.
            pred_membrane (torch.tensor): prediction of the cell memranes in the original image.
            z (torch.tensor): intermediate tensor before decoding.
        '''
        z, skips = self.encoder(x)
        z = self.bn(z)
        z = self.non_local(z)

        pred_membrane = self.d_membrane(z, skips)
        pred_nuclei = self.d_nuclei(z, skips)

        return pred_nuclei, pred_membrane, z
