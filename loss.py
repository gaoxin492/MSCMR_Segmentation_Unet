import torch
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from metrics import *

class SoftDiceLoss(_Loss):
    __name__ = 'dice_loss'

    def __init__(self, num_classes, activation=None, weight=[0.375,0.425,0.2]):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation
        self.num_classes = num_classes
        self.weight = weight

    def forward(self, y_pred, y_true):
        class_dice = []

        for i in range(1, self.num_classes):
            class_dice.append(diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], activation=self.activation))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice