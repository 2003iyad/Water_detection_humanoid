"""
model.py — Water segmentation architecture.
Author: Iyad Laphir

WaterSegNet: ResNet18 encoder (pretrained) + U-Net style decoder.
Outputs:
    mask_logits : [B, 1, H, W]  — pixel-level water region
    cls_logits  : [B, 5]        — water level 0-4
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

NUM_CLASSES = 5
CLASS_NAMES = ["level-0", "level-1", "level-2", "level-3", "level-4"]


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsample 2x then two ConvBlocks. Concatenates skip connection if provided."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=False)
        self.conv = nn.Sequential(
            ConvBlock(in_ch + skip_ch, out_ch),
            ConvBlock(out_ch, out_ch),
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class WaterSegNet(nn.Module):
    """
    ResNet18 encoder (pretrained) + U-Net decoder + level classifier.

    Encoder output sizes (input 416x416):
        enc_stem   : [B,  64, 104, 104]
        enc_layer1 : [B,  64, 104, 104]
        enc_layer2 : [B, 128,  52,  52]
        enc_layer3 : [B, 256,  26,  26]
        enc_layer4 : [B, 512,  13,  13]  <- bottleneck

    Decoder upsamples: 13 -> 26 -> 52 -> 104 -> 208 -> 416

    Returns:
        mask_logits : [B, 1, 416, 416]
        cls_logits  : [B, NUM_CLASSES]
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.enc_stem   = nn.Sequential(resnet.conv1, resnet.bn1,
                                        resnet.relu, resnet.maxpool)
        self.enc_layer1 = resnet.layer1
        self.enc_layer2 = resnet.layer2
        self.enc_layer3 = resnet.layer3
        self.enc_layer4 = resnet.layer4

        for module in [self.enc_stem, self.enc_layer1]:
            for param in module.parameters():
                param.requires_grad = False

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128,  64,  64)
        self.dec1 = DecoderBlock( 64,   0,  32)
        self.dec0 = DecoderBlock( 32,   0,  16)

        self.seg_head = nn.Conv2d(16, 1, kernel_size=1)

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        s0 = self.enc_stem(x)
        s1 = self.enc_layer1(s0)
        s2 = self.enc_layer2(s1)
        s3 = self.enc_layer3(s2)
        s4 = self.enc_layer4(s3)

        d4 = self.dec4(s4, s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2)
        d0 = self.dec0(d1)

        return self.seg_head(d0), self.cls_head(s4)
