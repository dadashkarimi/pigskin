import torch
import torch.nn as nn


class Logistic(nn.Module):
    def __init__(self, slope=1.0, midpoint=0.0, supremum=1.0):
        """
        slope (alpha)
        midpoint (x0)
        supremum (L)
        """
        super(Logistic, self).__init__()
        self.midpoint = midpoint
        self.slope = slope
        self.supremum = supremum

    def forward(self, x):
        x = self.slope * (x - self.midpoint)
        x = self.supremum * torch.sigmoid(x)
        return x


def assert_in_range(tensor, range, name='tensor'):
    assert len(range) == 2, 'range should be in form [min, max]'
    assert tensor.min() >= range[0], f'{name} should be in {range}, found: {tensor.min()}'
    assert tensor.max() <= range[1], f'{name} should be in {range}, found: {tensor.max()}'


def rand_uniform(rng, *args, **kwargs):
    """
    random uniform tensor float
    """
    assert len(rng) == 2, 'range should be a list with two entries'
    return torch.rand(*args, **kwargs) * (rng[1] - rng[0]) + rng[0]
