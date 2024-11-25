# https://github.com/mathiaszinnen/focal_loss_torch

from focal_loss.focal_loss import FocalLoss as FocalLossBase
from torch import Tensor
from typing import Union
import torch.nn as nn


class FocalLoss(FocalLossBase):
    def __init__(
            self,
            gamma,
            weight: Union[None, Tensor] = None,
            reduction: str = 'mean',
            ignore_index=-100,
            eps=1e-16
    ) -> None:
        super().__init__(gamma, weight, reduction, ignore_index, eps)
        # register self.weights to make "to" work
        if self.weights is not None:
            self.weights = nn.Parameter(self.weights)
