import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


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


class FocalLoss(nn.Module):
    # Adopted from: https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1),-1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
        targets = targets.view(-1, 1)

        logpt = F.log_softmax(inputs)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
