import numpy as np
from sklearn.metrics import confusion_matrix, jaccard_score, f1_score
import torch
from ripser import lower_star_img

def betti_number(img_true, pred):
    diags_pred = lower_star_img(pred)[:-1]
    diags = lower_star_img(img_true)[:-1]
    return abs(len(diags_pred) - len(diags))

def evaluate(y_pred, y_true):

    y_pred_ = y_pred.flatten()
    y_true_ = y_true.flatten().astype(np.uint8)

    # pixel eval
    m_jaccard = jaccard_score(y_true_, y_pred_)
    m_dice = f1_score(y_true_, y_pred_)

    # topology eval
    betti_error = 0
    for j in range(y_true.shape[0]):
        mask_j = y_true[j, :, :]
        pred_j = y_pred[j, :, :]

        betti_error += betti_number(mask_j, pred_j)

    betti_error = betti_error / y_true.shape[0]

    return m_jaccard, m_dice, betti_error