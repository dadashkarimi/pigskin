"""
Group layers
"""

import voxelmorph as vxm
import torch
import torch.nn as nn
import einops
from .. import layers
import pylot

import os
os.environ['VXM_BACKEND'] = 'pytorch'


class MeanConv2d(nn.Module):
    """ Perform a group mean convolution (see UniverSeg paper).

    inputs are [b, n, c, h, w] where 
        b is the batch size
        n is the number of group entries
        c is the number of channels 
        h is the height
        w is the width

    operation is:
        mean representation along group
        concat the mean representation with each group entry representation
        perform a convolution for each concated representation

    The idea is that this allows the entries to interact through the mean 
    representation, while still performing individual convolutions.

    """

    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 do_activation=True, do_batchnorm=True):
        super(MeanConv2d, self).__init__()

        conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=kernel_size, padding=padding)
        lst = [conv]
        if do_batchnorm:
            lst.append(nn.BatchNorm2d(out_channels))
        if do_activation:
            lst.append(nn.PReLU())

        self.conv = nn.Sequential(*lst)

    def forward(self, x):
        # TODO, use pylot.util.shapechecker
        n = x.shape[1]

        # mean represetation along group
        meanx = torch.mean(x, dim=1, keepdim=False)  # [B, C, H, W]
        meanx = einops.repeat(meanx, 'b c h w -> b n c h w', n=n)  # too memory intense?

        # concat the mean representation with each group entry representation
        x = torch.cat([x, meanx], dim=2)  # [b n 2c h w]

        # move group to batch dimension and do convolution
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.conv(x)
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=n)

        return x


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MaxPool2d, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        n = x.shape[1]
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.pool(x)
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=n)
        return x


class UpsamplingBilinear2d(nn.Module):
    def __init__(self, scale_factor):
        super(UpsamplingBilinear2d, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x):
        n = x.shape[1]
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.upsample(x)
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=n)
        return x


class Warp2d(nn.Module):
    def __init__(self, vol_shape, mode='bilinear'):
        super(Warp2d, self).__init__()
        self.st = layers.SpatialTransformer(vol_shape, mode=mode)

    def forward(self, x, w):
        n = x.shape[1]
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        w = einops.rearrange(w, 'b n c h w -> (b n) c h w')
        x = self.st(x, w)
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=n)
        return x
