import numpy as np
import torch

from . import data
from .. import utils


class MNISTStochasticThickness():

    def __init__(self, thr_range=[0.1, 0.9], img_noise_wts=[0.75, 0.25]):
        self.thr_range = thr_range
        self.img_noise_wts = img_noise_wts

        self.data = data.MNIST().data

    def generator(self, num_entries=5, split='train'):
        data = self.data[split]
        data_blur = self.data[split + '_blur']
        assert len(data.shape) == 4, 'data should be a 4D tensor of size'

        while True:
            # get a data point
            self.idx = np.random.randint(0, data.shape[0], 1)
            x_ = data[self.idx, ...]

            # add noise to it.
            wt = self.img_noise_wts
            x = wt[0] * x_ + wt[1] * torch.rand(x_.shape, device=x_.device)
            x = x.repeat(num_entries, 1, 1, 1)
            x = torch.concat([x, torch.rand(x.shape, device=x.device)], 1)

            # choose a threshold to simulate randomness of this point
            thr = utils.rand_uniform(self.thr_range, 1, device=x.device)
            y = data_blur[self.idx, ...] > thr
            y = y.float().repeat(num_entries, 1, 1, 1)

            # yield data
            yield x, y

    def __call__(self, *args, **kwds):
        self.generator(*args, **kwds)


class MNISTStochasticPair():
    """
    This class generates a pair of images from MNIST dataset and segments one of it.
    """

    def __init__(self):
        self.data = data.MNIST().data

    def generator(self, num_entries=5, split='train'):
        data = self.data[split]
        assert len(data.shape) == 4, 'data should be a 4D tensor of size'

        while True:
            # get a data point
            self.idx = np.random.randint(0, data.shape[0], 2)
            x_ = data[self.idx, ...]

            # mmix the two images
            x1 = torch.roll(x_[0:1, ...], 5, 3)
            x2 = torch.roll(x_[1:2, ...], -5, 3)
            x = 0.5 * x1 + 0.5 * x2
            x = x.repeat(num_entries, 1, 1, 1)
            x = torch.concat([x, torch.rand(x.shape, device=x.device)], 1)

            # choose one of the two images to segment
            thr = 0.5
            idx = np.random.choice([0, 1])
            y = [x1, x2][idx] > thr
            y = y.float().repeat(num_entries, 1, 1, 1)

            # yield data
            yield x, y


class OASISStochasticLabel():
    """
    This class generates a pair of images from MNIST dataset and segments one of it.
    """

    def __init__(self, resize=None):
        self.data = data.OASIS2D(resize=resize).data

    def generator(self, num_entries=5, num_labels=5, seg_type='seg4'):
        while True:
            # get a data point
            self.idx = np.random.randint(0, self.data['norm'].shape[0], 1)
            x = self.data['norm'][self.idx, ...]

            self.lidx = np.random.randint(0, num_labels, 1)
            y = torch.eq(self.data[seg_type][self.idx, ...], self.lidx[0])
            y = y.float().repeat(num_entries, 1, 1, 1)

            x = x.repeat(num_entries, 1, 1, 1)
            x = torch.concat([x, torch.rand(x.shape, device=x.device)], 1)

            # yield data
            yield x, y
