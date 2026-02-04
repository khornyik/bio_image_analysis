import torch 
import torch.nn as nn
import torch.nn.functional as F



class Conv(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.layer(x)
    


class Encoder(nn.Module):
    def __init__(self, in_size):
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

    def forward(self, x):

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
    def __init__(self, gating_channels, lfm_channels, int_channels):
        super().__init__()

        self.w_g = nn.Conv2d(gating_channels, int_channels, 1)
        self.w_x = nn.Conv2d(lfm_channels, int_channels, 1)
        self.psi = nn.Conv2d(int_channels, 1, 1)

    def forward(self, x, g):

        psi = F.relu(self.w_g(g) + self.w_x(x))
        psi = torch.sigmoid(self.psi(psi))

        return x * psi


class NonLocalLayer(nn.Module):
    def __init__(self, in_size):
        super().__init__()

        self.theta = nn.Conv2d(in_size, in_size // 2, 1)
        self.phi = nn.Conv2d(in_size, in_size // 2, 1)
        self.g = nn.Conv2d(in_size, in_size // 2, 1)
        self.out = nn.Conv2d(in_size // 2, in_size, 1)

    def forward(self, x):

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
    def __init__(self):
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


    def forward(self, z, skips):

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
    def __init__(self):
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


    def forward(self, z, skips):

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
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.bn = Conv(512, 1024)
        self.non_local = NonLocalLayer(1024)
        self.d_membrane = MembraneDecoder()
        self.d_nuclei = NucleiDecoder()

    def forward(self, x):
        z, skips = self.encoder(x)
        z = self.bn(z)
        z = self.non_local(z)

        pred_membrane = self.d_membrane(z, skips)
        pred_nuclei = self.d_nuclei(z, skips)

        return pred_nuclei, pred_membrane, z
