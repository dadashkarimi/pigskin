import torch.nn as nn
import torch.nn.functional as F
import neurite_sandbox as nes


class MinDice(nn.Module):
    def __init__(self, dim=0, eps=1e-3, from_logits=False):
        super(MinDice, self).__init__()
        self.dim = dim
        self.eps = eps
        self.from_logits = from_logits

    def dice(self, y_pred, y_true):
        """
        soft dice (higher is better)

        output is of shape y_pred.shape[0:dim+1]. 
        If using loss() function below, this will take the amin over dimension dim
        """

        # if expecting logits, sigmoid
        if self.from_logits:
            y_pred = F.sigmoid(y_pred.float())

        # check that all data is in [0, 1]
        nes.utils.assert_in_range(y_pred, [0, 1], 'y_pred')
        nes.utils.assert_in_range(y_true, [0, 1], 'y_true')

        # decide which dimensions to sum over.
        # in the minimum-dice loss, we sum over all dimensions
        # *after* the one we are taking the min over
        sum_dims = tuple(range(self.dim + 1, len(y_pred.shape)))

        # compute dice
        num = 2 * (y_pred * y_true).sum(dim=sum_dims) + self.eps
        denom = (y_pred ** 2).sum(dim=sum_dims) + \
            (y_true ** 2).sum(dim=sum_dims) + self.eps
        score = num / denom
        return score

    def loss(self, y_pred, y_true):
        """
        return the minimum (1 - dice_score) along dimension dim

        output is of shape y_pred.shape[0:dim]
        """
        score = 1 - self.dice(y_pred, y_true)
        return score.amin(dim=self.dim)

    def __call__(self, y_pred, y_true):
        return self.loss(y_pred, y_true)


class MinCE(nn.Module):

    def __init__(self, dim=0, from_logits=True):
        super(MinCE, self).__init__()
        self.dim = dim
        self.from_logits = from_logits

    def loss(self, y_pred, y_true):
        mean_dims = tuple(range(self.dim + 1, len(y_pred.shape)))

        # check that all data is in [0, 1]
        nes.utils.assert_in_range(y_true, [0, 1], 'y_true')

        if self.from_logits:
            score = nn.BCEWithLogitsLoss(reduction='none')(y_pred, y_true)
        else:
            nes.utils.assert_in_range(y_pred, [0, 1], 'y_pred')
            score = nn.BCELoss(reduction='none')(y_pred, y_true)

        score = score.mean(dim=mean_dims)
        return score.amin(dim=self.dim)

    def __call__(self, y_pred, y_true):
        return self.loss(y_pred, y_true)
