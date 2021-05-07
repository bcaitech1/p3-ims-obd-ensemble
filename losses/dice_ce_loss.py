import torch
from torch import nn
from torch.nn import functional as F

class DiceCELoss(nn.Module):
    def __init__(self):
        super(DiceCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-7):

        ce_loss = F.cross_entropy(inputs, targets)

        inputs = inputs.log_softmax(dim=1).exp()

        bs = targets.size(0)
        num_classes = inputs.size(1)
        dims = (0,2)

        targets = targets.view(bs, -1)
        inputs = inputs.view(bs, num_classes, -1)

        targets = F.one_hot(targets, num_classes)
        targets = targets.permute(0, 2, 1)

        intersection = torch.sum(inputs * targets, dim=dims)
        cardinality = torch.sum(inputs + targets, dim=dims)

        dice = (2.0 * intersection + smooth) / (cardinality + smooth)

        loss = 1 - dice

        mask = targets.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        return (loss.mean()) + ce_loss