import torch
from torch import nn
import torch.nn.functional as F


class LossMulti(nn.Module):
    def __init__(self, num_classes, jaccard_weight=0, weight=None):
        super().__init__()
        self.nll_loss = nn.NLLLoss(weight)
        self.num_classes = num_classes
        self.jaccard_weight = jaccard_weight

    def forward(self, inputs, targets):
        ls_inputs = F.log_softmax(inputs)
        loss = (1 - self.jaccard_weight) * self.nll_loss(ls_inputs, targets)
        if self.jaccard_weight:
            eps = 1e-15
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = ls_inputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= torch.log((intersection + eps) / (union - intersection + eps)) \
                    * self.jaccard_weight / self.num_classes
        return loss
