import shutil

import numpy as np
from sklearn.metrics import accuracy_score

import torch


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def cal_acc(true_y, pred_y, NUM_CLASSES):
    gt_list = true_y
    predict_list = pred_y
    num = NUM_CLASSES
    acc_sum = 0
    cal = 0
    sum_cal = 0
    for n in range(num):
        y = []
        predict_y = []
        for i in range(len(gt_list)):
            gt = gt_list[i]
            predict = predict_list[i]
            if gt == n:
                y.append(gt)
                predict_y.append(predict)
        if n < num-1:
            cal += accuracy_score(y, predict_y)*len(y)
    #         print(n)
        else:
            sum_cal = cal + accuracy_score(y, predict_y)*len(y)
    #     print(cal)

    Known_Acc = (cal / (len(gt_list)-len(y)))
    All_Acc = sum_cal / len(gt_list)

    return All_Acc,Known_Acc
def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))