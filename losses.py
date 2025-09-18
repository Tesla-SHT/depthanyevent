# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Collection of Losses.
"""

from enum import Enum
from typing import Dict, Literal, Optional, Tuple, cast

import torch
from torch import Tensor, nn

from kornia.filters.sobel import spatial_gradient, sobel
import torch.nn.functional as F


def masked_reduction(
    input_tensor,
    mask,
    reduction_type: Literal["image", "batch"],
) -> Tensor:
    """
    Whether to consolidate the input_tensor across the batch or across the image
    Args:
        input_tensor: input tensor
        mask: mask tensor
        reduction_type: either "batch" or "image"
    Returns:
        input_tensor: reduced input_tensor
    """
    if reduction_type == "batch":
        # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
        divisor = torch.sum(mask)
        if divisor == 0:
            return torch.tensor(0, device=input_tensor.device)
        input_tensor = torch.sum(input_tensor) / divisor
    elif reduction_type == "image":
        # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
        valid = mask.nonzero()

        input_tensor[valid] = input_tensor[valid] / mask[valid]
        input_tensor = torch.mean(input_tensor)
    return input_tensor

def normalized_depth_scale_and_shift(
    prediction, target, mask, min_quantile = 0.0, max_quantile = 1.0
):
    """
    More info here: https://arxiv.org/pdf/2206.00665.pdf supplementary section A2 Depth Consistency Loss
    This function computes scale/shift required to normalizes predicted depth map,
    to allow for using normalized depth maps as input from monocular depth estimation networks.
    These networks are trained such that they predict normalized depth maps.

    Solves for scale/shift using a least squares approach with a closed form solution:
    Based on:
    https://github.com/autonomousvision/monosdf/blob/d9619e948bf3d85c6adec1a643f679e2e8e84d4b/code/model/loss.py#L7
    Args:
        prediction: predicted depth map
        target: ground truth depth map
        mask: mask of valid pixels
    Returns:
        scale and shift for depth prediction
    """
    if min_quantile > 0.0 or max_quantile < 1.0:
        # compute quantiles
        min_quantile = torch.quantile(target[mask], min_quantile)
        max_quantile = torch.quantile(target[mask], max_quantile)
        mask = (target >= min_quantile) & (target <= max_quantile) & mask

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    scale = torch.zeros_like(b_0)
    shift = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = (det != 0)

    scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return scale, shift

# losses based on https://github.com/autonomousvision/monosdf/blob/main/code/model/loss.py
class GradientLoss(nn.Module):
    """
    multiscale, scale-invariant gradient matching term to the disparity space.
    This term biases discontinuities to be sharp and to coincide with discontinuities in the ground truth
    More info here https://arxiv.org/pdf/1907.01341.pdf Equation 11
    """

    def __init__(self, scales: int = 4, reduction_type: Literal["image", "batch"] = "batch"):
        """
        Args:
            scales: number of scales to use
            reduction_type: either "batch" or "image"
        """
        super().__init__()
        self.reduction_type: Literal["image", "batch"] = reduction_type
        self.__scales = scales

    def forward(
        self,
        prediction,
        target,
        mask,
    ):
        """
        Args:
            prediction: predicted depth map
            target: ground truth depth map
            mask: mask of valid pixels
        Returns:
            gradient loss based on reduction function
        """
        assert self.__scales >= 1
        total = 0.0

        for scale in range(self.__scales):
            step = pow(2, scale)

            grad_loss = self.gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
            )
            total += grad_loss

        assert isinstance(total, Tensor)
        return total
    
    def gradient_loss(
        self,
        prediction,
        target,
        mask,
    ):
        """
        multiscale, scale-invariant gradient matching term to the disparity space.
        This term biases discontinuities to be sharp and to coincide with discontinuities in the ground truth
        More info here https://arxiv.org/pdf/1907.01341.pdf Equation 11
        Args:
            prediction: predicted depth map
            target: ground truth depth map
            reduction: reduction function, either reduction_batch_based or reduction_image_based
        Returns:
            gradient loss based on reduction function
        """
        summed_mask = torch.sum(mask, (1, 2))
        diff = prediction - target
        diff = torch.mul(mask, diff)

        grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
        mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
        grad_x = torch.mul(mask_x, grad_x)

        grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
        mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
        grad_y = torch.mul(mask_y, grad_y)

        image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))
        image_loss = masked_reduction(image_loss, summed_mask, self.reduction_type)

        return image_loss

class MiDaSMSELoss(nn.Module):
    """
    data term from MiDaS paper
    """

    def __init__(self, reduction_type: Literal["image", "batch"] = "batch"):
        super().__init__()

        self.reduction_type: Literal["image", "batch"] = reduction_type
        # reduction here is different from the image/batch-based reduction. This is either "mean" or "sum"
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(
        self,
        prediction,
        target,
        mask,
    ):
        """
        Args:
            prediction: predicted depth map
            target: ground truth depth map
            mask: mask of valid pixels
        Returns:
            mse loss based on reduction function
        """
        summed_mask = torch.sum(mask, (1, 2))
        image_loss = torch.sum(self.mse_loss(prediction, target) * mask, (1, 2))
        # multiply by 2 magic number?
        image_loss = masked_reduction(image_loss, 2 * summed_mask, self.reduction_type)

        return image_loss


class ScaleAndShiftInvariantLoss(nn.Module):
    """
    Scale and shift invariant loss as described in
    "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer"
    https://arxiv.org/pdf/1907.01341.pdf
    """

    def __init__(self, alpha: float = 0.5, scales: int = 4, reduction_type: Literal["image", "batch"] = "batch", weight = 1.0):
        """
        Args:
            alpha: weight of the regularization term
            scales: number of scales to use
            reduction_type: either "batch" or "image"
        """
        super().__init__()
        self.__data_loss = MiDaSMSELoss(reduction_type=reduction_type)
        self.__regularization_loss = GradientLoss(scales=scales, reduction_type=reduction_type)
        self.__alpha = alpha
        self.weight = weight

        self.__prediction_ssi = None

    def forward(
        self,
        prediction,
        target,
        mask,
    ):
        """
        Args:
            prediction: predicted depth map (unnormalized)
            target: ground truth depth map (normalized)
            mask: mask of valid pixels
        Returns:
            scale and shift invariant loss
        """

        #Squeeze inputs
        if len(prediction.shape) == 4:
            prediction = prediction.squeeze(1)
        if len(target.shape) == 4:
            target = target.squeeze(1)
        if len(mask.shape) == 4:
            mask = mask.squeeze(1)

        scale, shift = normalized_depth_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total*self.weight


    def __get_prediction_ssi(self):
        """
        scale and shift invariant prediction
        from https://arxiv.org/pdf/1907.01341.pdf equation 1
        """
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5, weight=1.0):
        super().__init__()
        self.lambd = lambd
        self.weight = weight

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        _a = torch.log(target[valid_mask] + 1e-6)
        _b = torch.log(pred[valid_mask] + 1e-6)
        _mask = torch.isinf(_a) | torch.isinf(_b)
        _mask = (~_mask)
        diff_log = _a[_mask] - _b[_mask]

        loss = torch.var(diff_log) + self.lambd * torch.pow(torch.mean(diff_log), 2)
        loss = torch.sqrt(loss)

        return loss*self.weight

class SiLoss(nn.Module):
    def __init__(self, lambd=0.5, weight=1.0):
        super().__init__()
        self.lambd = lambd
        self.weight = weight

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff = target[valid_mask] - pred[valid_mask]

        loss = torch.var(diff) + self.lambd * torch.pow(torch.mean(diff), 2)
        loss = torch.sqrt(loss)

        return loss*self.weight
    
class MultiScaleGradient(torch.nn.Module):
    def __init__(self, start_scale = 1, num_scales = 4, weight = 1.0):
        super(MultiScaleGradient,self).__init__()
        print('Setting up Multi Scale Gradient loss...')

        self.start_scale = start_scale
        self.num_scales = num_scales
        self.weight = weight

        self.multi_scales = [torch.nn.AvgPool2d(self.start_scale * (2**scale), self.start_scale * (2**scale)) for scale in range(self.num_scales)]

        print('Done')

    def forward(self, prediction, target, valid_mask):        
        loss_value = 0
        
        diff = prediction - target
                
        for m in self.multi_scales:
            # input and type are of the type [B x C x H x W]
            # Use kornia spatial gradient computation
            delta_diff = spatial_gradient(m(diff))
            is_nan = torch.isnan(delta_diff)
            _downsampled_valid_mask = F.interpolate(valid_mask.float(), (delta_diff.shape[-2], delta_diff.shape[-1]), mode='nearest').bool().unsqueeze(1)
            mask_sum = ((~is_nan) & _downsampled_valid_mask).sum()
            # output of kornia spatial gradient is [B x C x 2 x H x W]
            loss_value += torch.abs(delta_diff[(~is_nan) & _downsampled_valid_mask]).sum()/mask_sum*target.shape[0]*2

        return (loss_value/self.num_scales)*self.weight

def fetch_losses(config_losses):
    losses = []

    for loss in config_losses:
        loss_type = loss["loss_type"]
        if loss_type == "ScaleAndShiftInvariantLoss":
            losses.append(ScaleAndShiftInvariantLoss(alpha=loss["alpha"], scales=loss["scales"], reduction_type=loss["reduction_type"], weight=loss["weight"]))
        elif loss_type == "SiLogLoss":
            losses.append(SiLogLoss(lambd=loss["lambd"], weight=loss["weight"]))
        elif loss_type == "SiLoss":
            losses.append(SiLoss(lambd=loss["lambd"], weight=loss["weight"]))
        elif loss_type == "MultiScaleGradient":
            losses.append(MultiScaleGradient(start_scale=loss["start_scale"], num_scales=loss["num_scales"], weight=loss["weight"]))
        else:
            raise ValueError(f"Loss type {loss_type} not recognized")
    
    return losses