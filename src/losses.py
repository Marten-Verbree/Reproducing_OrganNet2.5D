import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary

def dice_coef(y_true, y_pred, epsilon=1e-6):
    """ Computes the Sørensen-dice score coefficien(DSC).
        DSC = (2*(|X&Y|)\(|X| + |Y|)
            = 2*sum(|A*B|)/(sum(A^2 + sum(B^2)
        ref: https://github.com/shalabh147/Brain-Tumor-Segmentation-and-Survival-Prediction-using-Deep-Neural-Networks/blob/master/utils.py
        ref: https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08#file-soft_dice_loss-py

    Args:
        :param y_true: is a tensor [H, W, D, L] with the ground truth of the OAR
        :param y_pred: is a tensor [H, W, D, L] with the predicted area of the OAR
        :param epsilon: Used for numerical stability to avoid divide by zeros.
    """
    dice_scores = []

    for i in range(y_pred.shape[-1]):

        y_pred_label = y_pred([:, :, :, i])
        y_true_labe =  y_true([:, :, :, i])

        if torch.sum(y_true_i) > 0:
            dice_numerator = 2 * torch.sum(y_true_label * y_pred_label)
            dice_denominator = torch.sum(y_true_label * torch(y_pred_label)) + epsilon
            dice_score = dice_numerator/dice_denominator
        else:
            dice_score = 0

        dice_scores.append(dice_score)

    dice_avg = torch.mean(dice_scores)
    return dice_avg, dice_scores

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def focal_loss(y_true, y_pred,  epsilon =1e-6):
    """ Computes the focal loss.
            FL(p_t) = mean(-alpha(1-p_t^gamma)* y *ln(p_t)
            Notice: y_pred is probability after clamping between 0 and 1
            ref: https://arxiv.org/pdf/2109.12634.pdf

        Args:
            :param y_true: is a tensor [H, W, D, L] with the ground truth of the OAR
            :param y_pred: is a tensor [H, W, D, L] with the predicted area of the OAR
            :param epsilon: Used for numerical stability to avoid divide by zeros
            :param gamma: Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
            :param ALPHA: assigned weights according to Chen et al. (2021)
        """

    ALPHA = torch.tensor([0.5, 1.0, 4.0, 1.0, 4.0, 4.0, 1.0, 1.0, 3.0, 3.0]) 
    GAMMA = 2

    loss_labels = []

    for i in range(y_pred.shape[-1]):

        y_pred_label = y_pred([:, :, :, i])
        y_true_label =  y_true([:, :, :, i])

        y_pred_clamp = torch.clamp(y_pred_label, epsilon, 1 - epsilon)
        cross_entropy = -y_true_label * torch.log(y_pred_label)

        back_ce = torch.pow(1 - y_pred_clamp, GAMMA) * cross_entropy[]

        focal_loss_label = torch.mul(ALPHA([i]), back_ce)

        loss_labels.append(focal_loss_label)

    loss =  torch.mean(loss_labels)

    return loss


def final_loss(y_true, y_pred):
    return focal_loss(y_true, y_pred) + dice_coef_loss(y_true, y_pred)