import os
import pandas as pd
import numpy as np
import math
import time
import torch


def metric(logit, truth, threshold=0.5):
    batch_size, num_class = logit.shape
    # print(batch_size, num_class)

    with torch.no_grad():
        logit = logit.view(batch_size, num_class)
        # truth = truth.view(batch_size, num_class)

        # probability = torch.sigmoid(logit)
        # print(probability)
        p = (logit > threshold).float()
        t = (truth > 0.5).float()

        tp = ((p + t) == 2).float()  # True positives
        tn = ((p + t) == 0).float()  # True negatives
        # 各个类别预测正确的正样本、负样本数目
        # print(tp)
        # print(p)
        # print(t)
        tp = tp.sum(0)
        tn = tn.sum(0)
        num_pos = t.sum(0)
        num_neg = batch_size - num_pos
        # 预测正确的正样本和负样本的数目
        tp = tp.data.cpu().numpy()
        tn = tn.data.cpu().numpy()
        # 正样本、负样本的数目
        num_pos = num_pos.data.cpu().numpy()
        num_neg = num_neg.data.cpu().numpy()

        # tp = np.nan_to_num(tp / (num_pos + 1e-12), 0)
        # tn = np.nan_to_num(tn / (num_neg + 1e-12), 0)

        # tp = list(tp)
        # num_pos = list(num_pos)

    return tn, tp, num_neg, num_pos


class Meter():
    '''
    A meter to keep track of iou and dice scores throughout an epoch
    '''

    def __init__(self):
        self.base_threshold = 0.5
        self.true_negative = []
        self.true_poisitive = []
        self.number_negative = []
        self.number_positive = []

    def update(self, targets, outputs):
        tn, tp, num_neg, num_pos = metric(outputs, targets, self.base_threshold)
        self.true_negative.append(tn)
        self.true_poisitive.append(tp)
        self.number_negative.append(num_neg)
        self.number_positive.append(num_pos)

    def get_metrics(self):
        # 各类预测正确的样本数目，样本总数目
        class_tn = np.sum(np.array(self.true_negative), axis=0)
        class_tp = np.sum(np.array(self.true_poisitive), axis=0)
        class_num_neg = np.sum(np.array(self.number_negative), axis=0)
        class_num_pos = np.sum(np.array(self.number_positive), axis=0)
        # 预测正确的样本的总数目，样本总数目
        tn = np.sum(self.true_negative)
        tp = np.sum(self.true_poisitive)
        num_neg = np.sum(self.number_negative)
        num_pos = np.sum(self.number_positive)
        # 各类的正负样本的准确率和总的准确率
        class_neg_accuracy = class_tn / class_num_neg
        class_pos_accuracy = class_tp / class_num_pos
        class_accuracy = (class_tn + class_tp) / (class_num_neg + class_num_pos)
        # 正负样本各自的准确率和总的准确率
        neg_accuracy = tn / (num_neg + 1e-12)
        pos_accuracy = tp / (num_pos + 1e-12)
        accuracy = (tn + tp) / (num_neg + num_pos)

        return class_neg_accuracy, class_pos_accuracy, class_accuracy, neg_accuracy, pos_accuracy, accuracy
