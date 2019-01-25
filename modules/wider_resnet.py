from collections import OrderedDict
import torch.nn as nn
from .bn import ABN, InPlaceABN
from .misc import GlobalAvgPool2d
from .residual import IdentityResidualBlock
from functools import partial


class WiderResNet(nn.Module):
    def __init__(self, structure, norm_act=ABN, classes=0):
        super().__init__()
        self.structure = structure

        if len(structure) != 6:
            raise ValueError('Expected a structure with six values')

        self.mod1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        ]))

        in_channels = 64
        channels = [(128, 128), (256, 256), (384, 384), (512, 512), (512, 1024, 2048),
                    (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            blocks = []
            for block_id in range(num):
                if mod_id == 2 or mod_id == 3:
                    drop = partial(nn.Dropout2d, p=0.3)
                else:
                    drop = None

                blocks.append((f'block{block_id+1}',
                               IdentityResidualBlock(in_channels, channels[mod_id],
                                                     norm_act=norm_act, dropout=drop)))
                in_channels = channels[mod_id][-1]

            if mod_id <= 4:
                self.add_module(f'pool{mod_id+2}', nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module(f'mod{mod_id+2}', nn.Sequential(OrderedDict(blocks)))

        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([
                ('avg_pool', GlobalAvgPool2d()),
                ('fc', nn.Linear(in_channels, classes))
            ]))

    def forward(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(self.pool4(out))
        out = self.mod5(self.pool5(out))
        out = self.mod6(self.pool6(out))
        out = self.mod7(out)
        out = self.bn_out(out)

        if hasattr(self, 'classes'):
            out = self.classifier(out)

        return out