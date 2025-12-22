import torch.nn as nn

import os
os.environ['NEURITE_BACKEND'] = 'pytorch'
import neurite_sandbox as nes  # noqa: E402


class MultiMorph(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 features=[64, 64, 64, 64],
                 conv_kernel_size=3,
                 centralize_warpes=True):
        super(MultiMorph, self).__init__()
        self.gnet = nes.models.GroupNet(in_channels=in_channels,
                                        out_channels=out_channels,
                                        features=features,
                                        conv_kernel_size=conv_kernel_size)
        self.subt = nes.layers.SubtractMean(dim=1)
        self.centralize_warpes = centralize_warpes

    def forward(self, x):
        predw = self.gnet(x)
        if self.centralize_warpes:
            predw = self.subt(predw)
        return predw
