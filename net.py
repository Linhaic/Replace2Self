from layer import *

import torch
import torch.nn as nn
from torch.nn import init

class UNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(UNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        """
        Encoder part
        """

        self.enc1_1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc1_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.pool1 = Pooling2d(pool=2, type='avg')

        self.enc2_1 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc2_2 = CNR2d(2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.pool2 = Pooling2d(pool=2, type='avg')

        self.enc3_1 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc3_2 = CNR2d(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.pool3 = Pooling2d(pool=2, type='avg')

        self.enc4_1 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc4_2 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.pool4 = Pooling2d(pool=2, type='avg')

        self.enc5_1 = CNR2d(8 * self.nch_ker, 2 * 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        """
        Decoder part
        """

        self.dec5_1 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.unpool4 = UnPooling2d(pool=2, type='nearest')

        self.dec4_2 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec4_1 = DECNR2d(8 * self.nch_ker,     4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.unpool3 = UnPooling2d(pool=2, type='nearest')

        self.dec3_2 = DECNR2d(2 * 4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec3_1 = DECNR2d(4 * self.nch_ker,     2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.unpool2 = UnPooling2d(pool=2, type='nearest')

        self.dec2_2 = DECNR2d(2 * 2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec2_1 = DECNR2d(2 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.unpool1 = UnPooling2d(pool=2, type='nearest')

        self.dec1_2 = DECNR2d(2 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec1_1 = DECNR2d(1 * self.nch_ker,     1 * self.nch_out, kernel_size=3, stride=1, norm=[],        relu=[],  drop=[], bias=False)

    def forward(self, x):

        """
        Encoder part
        """

        enc1 = self.enc1_2(self.enc1_1(x))
        pool1 = self.pool1(enc1)

        enc2 = self.enc2_2(self.enc2_1(pool1))
        pool2 = self.pool2(enc2)

        enc3 = self.enc3_2(self.enc3_1(pool2))
        pool3 = self.pool3(enc3)

        enc4 = self.enc4_2(self.enc4_1(pool3))
        pool4 = self.pool4(enc4)

        enc5 = self.enc5_1(pool4)

        """
        Encoder part
        """
        dec5 = self.dec5_1(enc5)

        unpool4 = self.unpool4(dec5)
        cat4 = torch.cat([enc4, unpool4], dim=1)
        dec4 = self.dec4_1(self.dec4_2(cat4))

        unpool3 = self.unpool3(dec4)
        cat3 = torch.cat([enc3, unpool3], dim=1)
        dec3 = self.dec3_1(self.dec3_2(cat3))

        unpool2 = self.unpool2(dec3)
        cat2 = torch.cat([enc2, unpool2], dim=1)
        dec2 = self.dec2_1(self.dec2_2(cat2))

        unpool1 = self.unpool1(dec2)
        cat1 = torch.cat([enc1, unpool1], dim=1)
        dec1 = self.dec1_1(self.dec1_2(cat1))

        x = dec1

        return x


# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
# https://arxiv.org/abs/1609.04802
class ResNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm', nblk=16):
        super(ResNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        self.nblk = nblk

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.enc1 = CNR2d(self.nch_in, self.nch_ker, kernel_size=3, stride=1, padding=1, norm=[], relu=0.0)

        res = []
        for i in range(self.nblk):
            res += [ResBlock(self.nch_ker, self.nch_ker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0, padding_mode='reflection')]
        self.res = nn.Sequential(*res)

        self.dec1 = CNR2d(self.nch_ker, self.nch_ker, kernel_size=3, stride=1, padding=1, norm=norm, relu=[])

        self.conv1 = Conv2d(self.nch_ker, self.nch_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.enc1(x)
        x0 = x

        x = self.res(x)

        x = self.dec1(x)
        x = x + x0

        x = self.conv1(x)

        return x

