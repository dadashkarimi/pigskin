import neurite_sandbox as nes
import torch
import torch.nn as nn

import os
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # noqa: E402


class Grad2d:
    """
    2D gradient loss for group warps
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred):
        dy = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class MinVar2d(nn.Module):
    """ assumes data is of the shape [b n c h w]

    Args:
        nn (_type_): _description_
    """

    def __init__(self, volshape):
        super(MinVar2d, self).__init__()
        self.size = volshape

    def forward(self, images, warps):
        # warp inputs
        warp_layer = nes.layers.group.Warp2d(self.size)
        warped = warp_layer(images, warps)
        m = torch.mean(torch.var(warped, dim=1))
        return m


class MinAtlCC2d(nn.Module):
    """ assumes data is of the shape [b n c h w]

    Args:
        nn (_type_): _description_
    """

    def __init__(self, volshape):
        super(MinAtlCC2d, self).__init__()
        self.size = volshape

    def forward(self, images, warps):
        # warp inputs
        warp_layer = nes.layers.group.Warp2d(self.size)
        warped = warp_layer(images, warps)
        atl = torch.mean(warped, dim=1)
        q = 0.
        # should figure out how to do with einops repeat and such.
        for i in range(warped.shape[1]):
            q += vxm.losses.NCC().loss(atl, warped[:, i, ...])
        return q / warped.shape[1]


class MinAtlCCAndGrad2d(nn.Module):
    def __init__(self, volshape, lbd):
        super(MinAtlCCAndGrad2d, self).__init__()
        self.size = volshape
        self.lbd = lbd
        self.minvar = MinAtlCC2d(volshape)

    def forward(self, images, warps):
        # warp inputs
        m = self.minvar(images, warps)
        g2 = Grad2d('l2').loss(warps)

        self.m = m
        self.g2 = g2

        return m + self.lbd * g2


class MinVarAndGrad2d(nn.Module):
    def __init__(self, volshape, lbd):
        super(MinVarAndGrad2d, self).__init__()
        self.size = volshape
        self.lbd = lbd
        self.minvar = MinVar2d(volshape)

    def forward(self, images, warps):
        # warp inputs
        m = self.minvar(images, warps)
        g2 = Grad2d('l2').loss(warps)

        self.m = m
        self.g2 = g2

        return m + self.lbd * g2
