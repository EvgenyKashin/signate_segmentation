from torch import nn
import torch
from torchvision import models
from torch.nn import functional as F
from modules import ABN
from modules import WiderResNet


def get_channels(architecture):
    if architecture in ['resnet18', 'resnet34']:
        return [512, 256, 128, 64]
    elif architecture in ['resnet50', 'resnet101', 'resnet152']:
        return [2048, 1024, 512, 256]
    else:
        raise Exception(f'{architecture} is not supported as backbone')


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, kernel_size=3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super().__init__()
        self.in_channels = in_channels
        self.is_deconv = is_deconv

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels)
            )

    def forward(self, x):
        if self.is_deconv:
            return self.block(x)
        else:
            return self.block(F.interpolate(x, scale_factor=2, mode='nearest'))


class ResNetUnet(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, backbone='resnet34', pretrained=True,
                 is_deconv=False):
        super().__init__()
        self.resnet = models.__dict__[backbone](pretrained=pretrained)
        encoder_channels = get_channels(backbone)

        self.center = DecoderBlock(encoder_channels[0], num_filters * 8 * 2,
                                   num_filters * 8, is_deconv)
        self.dec5 = DecoderBlock(encoder_channels[0] + num_filters * 8, num_filters * 8 * 2,
                                 num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(encoder_channels[1] + num_filters * 8, num_filters * 8 * 2,
                                 num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(encoder_channels[2] + num_filters * 8, num_filters * 4 * 2,
                                 num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(encoder_channels[3] + num_filters * 2, num_filters * 2 * 2,
                                 num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2,
                                 num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        size = x.size()
        assert size[-1] % 64 == 0 and size[-2] % 64 == 0, \
            'image resolution has to be divisible by 64 for resnet'

        enc0 = self.resnet.conv1(x)
        enc0 = self.resnet.bn1(enc0)
        enc0 = self.resnet.relu(enc0)
        enc0 = self.resnet.maxpool(enc0)

        enc1 = self.resnet.layer1(enc0)
        enc2 = self.resnet.layer2(enc1)
        enc3 = self.resnet.layer3(enc2)
        enc4 = self.resnet.layer4(enc3)

        center = self.center(F.max_pool2d(enc4, kernel_size=2, stride=2))

        dec5 = self.dec5(torch.cat([center, enc4], 1))
        dec4 = self.dec4(torch.cat([dec5, enc3], 1))
        dec3 = self.dec3(torch.cat([dec4, enc2], 1))
        dec2 = self.dec2(torch.cat([dec3, enc1], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        return self.final(dec0)


class TernausNetV2(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, is_deconv=False, **kwargs):
        super().__init__()
        if 'norm_act' not in kwargs:
            norm_act = ABN
        else:
            norm_act = kwargs['norm_act']

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = WiderResNet(structure=[1, 1, 1, 3, 1, 1], classes=1, norm_act=norm_act)

        self.center = DecoderBlock(512, num_filters * 8, num_filters * 8,
                                   is_deconv=is_deconv)
        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8, num_filters * 8,
                                 is_deconv=is_deconv)
        self.dec4 = DecoderBlock(384 + num_filters * 8, num_filters * 8, num_filters * 8,
                                 is_deconv=is_deconv)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 6, num_filters * 6,
                                 is_deconv=is_deconv)
        self.dec2 = DecoderBlock(128 + num_filters * 6, num_filters * 4, num_filters * 2,
                                 is_deconv=is_deconv)
        self.dec1 = ConvRelu(64 + num_filters * 2, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        mod1 = self.encoder.mod1(x)
        mod2 = self.encoder.mod2(self.pool(mod1))
        mod3 = self.encoder.mod3(self.pool(mod2))
        mod4 = self.encoder.mod4(self.pool(mod3))
        mod5 = self.encoder.mod5(self.pool(mod4))

        center = self.center(self.pool(mod5))

        dec5 = self.dec5(torch.cat([mod5, center], 1))
        dec4 = self.dec4(torch.cat([mod4, dec5], 1))
        dec3 = self.dec3(torch.cat([mod3, dec4], 1))
        dec2 = self.dec2(torch.cat([mod2, dec3], 1))
        dec1 = self.dec1(torch.cat([mod1, dec2], 1))
        return self.final(dec1)