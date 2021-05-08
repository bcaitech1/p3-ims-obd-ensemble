from .base import *
from torch import nn
from fastai.vision.all import *

import torch
import timm

class EffUnet(nn.Module):
    def __init__(self, stride=1, **kwargs):
        super().__init__()
        #encoder
        m = timm.create_model('tf_efficientnet_b4_ns', pretrained=True)
        self.enc0 = nn.Sequential(m.conv_stem, m.bn1, nn.SiLU(inplace=True))
        self.enc1 = m.blocks[0:2] # [24, 24, 24, 32, 32]
        self.enc2 = m.blocks[2] # [40, 40, 48, 48, 56]
        self.enc3 = m.blocks[3:5] # [112, 112, 120, 136, 160]
        self.enc4 = m.blocks[5:7] # [320, 320, 352, 384, 448]
        #aspp with customized dilatations
        self.aspp = ASPP(448,112,dilations=[stride*1,stride*2,stride*3,stride*4],out_c=256)
        self.drop_aspp = nn.Dropout2d(0.5)
        #decoder
        self.dec4 = UnetBlock(256,160,128)
        self.dec3 = UnetBlock(128,56, 64)
        self.dec2 = UnetBlock(64,32,32)
        self.dec1 = UnetBlock(32,48,16)
        self.fpn = FPN([256,128,64,32],[16]*4)
        self.drop = nn.Dropout2d(0.3)
        self.final_conv = ConvLayer(16+16*4, 12, ks=1, norm_type=None, act_cls=None)
    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.aspp(enc4)
        dec3 = self.dec4(self.drop_aspp(enc5),enc3)
        dec2 = self.dec3(dec3,enc2)
        dec1 = self.dec2(dec2,enc1)
        dec0 = self.dec1(dec1,enc0)
        x = self.fpn([enc5, dec3, dec2, dec1], dec0)
        x = self.final_conv(self.drop(x))
        x = F.interpolate(x,scale_factor=2,mode='bilinear')
        return x