# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from functools import partial

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class SRN_loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(SRN_loss, self).__init__()
        # self.eps = e
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True
        self.resize = partial(F.interpolate, mode='area', recompute_scale_factor=True)
        # self.scale = 1e2 / np.log(1e2)

    def forward(self, batch_p, batch_l):
        assert batch_p[0].shape[0] == batch_l.shape[0]
        device = batch_p[0].device
        b, c, h, w = batch_p[0].shape
        # self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        loss = self.loss_weight * self.scale * torch.log(
            ((batch_p[0] - batch_l) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        ####################################################### multi-scale
        loss += 0.5 * self.loss_weight * self.scale * torch.log(
            ((batch_p[1] - self.resize(input=batch_l, scale_factor=0.5)) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        loss += 0.25 * self.loss_weight * self.scale * torch.log(
            ((batch_p[2] - self.resize(input=batch_l, scale_factor=0.25)) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        loss += 0.125 * self.loss_weight * self.scale * torch.log(
            ((batch_p[3] - self.resize(input=batch_l, scale_factor=0.125)) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        #################################################################
        # TODO scale100
        # js_loss = torch.tensor([0.7, 0.5, 0.3]).to(device).dot(
        #     torch.stack([js_div(batch_p[1], F.interpolate(batch_p[0], size=(h // 2, w // 2), mode="nearest")),
        #                  js_div(batch_p[2], F.interpolate(batch_p[0], size=(h // 4, w // 4), mode="nearest")),
        #                  js_div(batch_p[3], F.interpolate(batch_p[0], size=(h // 8, w // 8), mode="nearest"))]))

        # loss += js_loss
        return loss, 0.0

class FFTLoss(nn.Module):
    """L1 loss in frequency domain with FFT.

    Args:
        loss_weight (float): Loss weight for FFT loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (..., C, H, W). Predicted tensor.
            target (Tensor): of shape (..., C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (..., C, H, W). Element-wise
                weights. Default: None.
        """

        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        return self.loss_weight * l1_loss(pred_fft, target_fft, weight, reduction=self.reduction)


def GW_loss(x1, x2):
    sobel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sobel_y = [[-1,-2,-1],[0,0,0],[1,2,1]]
    b, c, w, h = x1.shape
    sobel_x = torch.FloatTensor(sobel_x).expand(c, 1, 3, 3)
    sobel_y = torch.FloatTensor(sobel_y).expand(c, 1, 3, 3)
    sobel_x = sobel_x.type_as(x1)
    sobel_y = sobel_y.type_as(x1)
    weight_x = nn.Parameter(data=sobel_x, requires_grad=False)
    weight_y = nn.Parameter(data=sobel_y, requires_grad=False)
    Ix1 = F.conv2d(x1, weight_x, stride=1, padding=1, groups=c)
    Ix2 = F.conv2d(x2, weight_x, stride=1, padding=1, groups=c)
    Iy1 = F.conv2d(x1, weight_y, stride=1, padding=1, groups=c)
    Iy2 = F.conv2d(x2, weight_y, stride=1, padding=1, groups=c)
    dx = torch.abs(Ix1 - Ix2)
    dy = torch.abs(Iy1 - Iy2)
#     loss = torch.exp(2*(dx + dy)) * torch.abs(x1 - x2)
    loss = (1 + 4*dx) * (1 + 4*dy) * torch.abs(x1 - x2)
    return torch.mean(loss)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3, reduction="mean"):
        super().__init__()
        if reduction != "mean" and reduction != "sum" and reduction != "None":
            raise ValueError("Reduction type not supported")
        else:
            self.reduction = reduction
        self.eps = eps

    def forward(self, output, target):
        diff = output - target

        out = torch.sqrt((diff * diff) + (self.eps * self.eps))
        if self.reduction == "mean":
            out = torch.mean(out)
        elif self.reduction == "sum":
            out = torch.sum(out)

        return out


class GradientWeightedLoss(nn.Module):
    def __init__(self, depth=4):
        """
        Taken from:
        """
        super(GradientWeightedLoss, self).__init__()
        self.depth = depth
        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        self.sobel_x = torch.tensor(sobel_x).float().expand(3, 1, 3, 3)
        self.sobel_y = torch.tensor(sobel_y).float().expand(3, 1, 3, 3)
        if torch.cuda.is_available():
            self.sobel_x = self.sobel_x.cuda()
            self.sobel_y = self.sobel_y.cuda()
        self.resize = partial(F.interpolate, mode='area', recompute_scale_factor=True)
        for i in range(self.depth):
            setattr(self, f'loss_{i}', CharbonnierLoss(reduction='None'))

    def _get_gradient(self, n_depth, p, l):
        l = self.resize(input=l, scale_factor=0.5 ** n_depth)
        Ix1 = F.conv2d(p, self.sobel_x, stride=1, padding=1, groups=3)
        Ix2 = F.conv2d(l, self.sobel_x, stride=1, padding=1, groups=3)
        Iy1 = F.conv2d(p, self.sobel_y, stride=1, padding=1, groups=3)
        Iy2 = F.conv2d(l, self.sobel_y, stride=1, padding=1, groups=3)
        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)
        return dx, dy

    def _get_pix_loss(self, n_depth, p, l, weight):
        level_loss = getattr(self, f'loss_{n_depth}')
        return weight * level_loss(p, self.resize(input=l, scale_factor=0.5 ** n_depth))

    def forward(self, batch_p, batch_l):
        assert batch_p[0].shape[0] == batch_l.shape[0]
        loss = 0.
        # self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        for i in range(self.depth):
            dx, dy = self._get_gradient(i, batch_p[i], batch_l)
            loss += torch.mean((1 + 4*dx) * (1 + 4*dy) * self._get_pix_loss(i, batch_p[i], batch_l, 1-0.2*i))

        # loss += js_loss
        return loss, 0.0



class MultiCharbonnierLoss(nn.Module):
    def __init__(self, depth=4):
        super(MultiCharbonnierLoss, self).__init__()
        self.depth = depth
        self.resize = partial(F.interpolate, mode='area', recompute_scale_factor=True)
        for i in range(self.depth):
            setattr(self, f'loss_{i}', CharbonnierLoss(reduction='mean'))

    def _get_pix_loss(self, n_depth, p, l, weight):
        level_loss = getattr(self, f'loss_{n_depth}')
        return weight * level_loss(p, self.resize(input=l, scale_factor=0.5 ** n_depth))

    def forward(self, batch_p, batch_l):
        assert batch_p[0].shape[0] == batch_l.shape[0]
        loss = 0.
        # self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        for i in range(self.depth):
            loss += self._get_pix_loss(i, batch_p[i], batch_l, 1-0.2*i)

        # loss += js_loss
        return loss, 0.0

class EdgeLoss(nn.Module):
    def __init__(self, weight=0.05):
        """
        Taken from:
        https://github.com/swz30/MPRNet/blob/main/Deblurring/losses.py
        """
        super(EdgeLoss, self).__init__()
        self.weight = weight
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down*4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return self.weight * loss