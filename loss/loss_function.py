import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import sys
from loss.soft_skeleton import soft_skel

class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean', ignore_index=-1):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "predict & target batch size don't match"
        CE = nn.CrossEntropyLoss(reduction=self.reduction, ignore_index=self.ignore_index)
        loss = CE(output, target)
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean', ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, output, target, weights):
        assert output.shape[0] == target.shape[0], "predict & target batch size don't match"
        CE = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)
        loss = CE(output, target)
        loss = (loss * weights).view(output.shape[0], -1).sum(1) / weights.view(output.shape[0], -1).sum(1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class soft_cldice(nn.Module):
    def __init__(self, iter_=50, smooth=1., aux=False, aux_weight=0.4):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.aux = aux
        self.aux_weight = aux_weight

    def _base_forward(self, output, target, **kwargs):

        output = torch.max(output, 1)[1]
        output_one_hot = F.one_hot(torch.clamp_min(output, 0))
        target_one_hot = F.one_hot(torch.clamp_min(target, 0))

        output_one_hot = output_one_hot.permute(0, 3, 1, 2)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2)

        output_one_hot = output_one_hot.float()
        target_one_hot = target_one_hot.float()

        skel_pred = soft_skel(output_one_hot, self.iter)
        skel_true = soft_skel(target_one_hot, self.iter)

        tprec = (torch.sum(torch.mul(skel_pred, target_one_hot)[:, 1:, ...]) + self.smooth) / (torch.sum(skel_pred[:, 1:, ...]) + self.smooth)
        tsens = (torch.sum(torch.mul(skel_true, output_one_hot)[:, 1:, ...]) + self.smooth) / (torch.sum(skel_true[:, 1:, ...]) + self.smooth)
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice


    def _aux_forward(self, output, target, **kwargs):

        loss = self._base_forward(output[0], target)
        for i in range(1, len(output)):
            aux_loss = self._base_forward(output[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, output, target):

        if self.aux:
            return self._aux_forward(output, target)
        else:
            return self._base_forward(output, target)

def segmentation_loss(loss='CE'):
    if loss == 'crossentropy' or loss == 'CE':
        seg_loss = CrossEntropyLoss(reduction='mean')
    elif loss == 'weighted_crossentropy' or loss == 'WCE':
        seg_loss = WeightedCrossEntropyLoss(reduction='mean')
    elif loss == 'cldice' or loss == 'CLDICE':
        seg_loss = soft_cldice()
    else:
        print('sorry, the loss you input is not supported yet')
        sys.exit()

    return seg_loss


# if __name__ == '__main__':
#     from models import *
#     # criterion = segmentation_loss(loss='dice')
#     criterion = nn.CrossEntropyLoss(reduction='none')
#     criterion1 = segmentation_loss(loss='CE')
#     model = unet(1, 2)
#     model.eval()
#     input = torch.rand(3, 1, 128, 128)
#     mask = torch.ones(3, 128, 128).long()
#
#     mask[:, 40:100, 30:60] = 1
#     output = model(input)
#
#     weights = torch.rand(3, 128, 128)
#     loss = criterion(output, mask)
#     loss = torch.mean((loss * weights).view(input.shape[0], -1).sum(1) / weights.view(input.shape[0], -1).sum(1))
#     print(loss)
#
#     loss_ = criterion1(output, mask, weights)
#     print(loss)
#
#     # weights = (weights.view(3, -1) / weights.view(3, -1).sum(1).view(3, -1)).view(3, 128, 128)
#     #
#     # loss = torch.sum(loss * weights) / 3
#     # print(loss)
#     # loss.requires_grad_(True)
#     # loss.backward()

