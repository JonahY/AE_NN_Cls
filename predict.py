import os
import numpy as np
import math
import time
import argparse
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch import optim
import tqdm
from multiprocessing import cpu_count
from solver import Solver
from meter import Meter
from network import Classify_model
from dataset import classify_provider


class Predict():
    def __init__(self, config):
        self.layer = config.layer
        self.method = config.method
        self.model = Classify_model(self.layer, self.method, training=True)
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.cuda()

        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.epoch = config.epoch
        self.max_accuracy_valid = 0
        self.solver = Solver(self.model, self.method)
        self.criterion = torch.nn.MSELoss()
        self.weight_path = config.weight_path

    def validation(self, test_loader):
        self.model.eval()
        self.model.train(False)
        checkpoint = torch.load(self.weight_path)
        self.model.module.load_state_dict(checkpoint['state_dict'])
        meter = Meter()
        tbar = tqdm.tqdm(test_loader, ncols=80)
        loss_sum = 0

        with torch.no_grad():
            for i, (x, labels) in enumerate(tbar):
                labels_predict = self.solver.forward(x)
                labels_predict = torch.sigmoid(labels_predict)
                loss = self.solver.cal_loss(labels, labels_predict, self.criterion)
                loss_sum += loss.item()

                meter.update(labels, labels_predict.cpu())

                descript = "Val Loss: {:.7f}".format(loss.item())
                tbar.set_description(desc=descript)
        loss_mean = loss_sum / len(tbar)

        class_neg_accuracy, class_pos_accuracy, class_accuracy, neg_accuracy, pos_accuracy, accuracy = meter.get_metrics()
        print(
            "Class_0_accuracy: %0.4f | Class_1_accuracy: %0.4f | Negative accuracy: %0.4f | positive accuracy: %0.4f | accuracy: %0.4f" %
            (class_accuracy[0], class_accuracy[1], neg_accuracy, pos_accuracy, accuracy))
        return class_neg_accuracy, class_pos_accuracy, class_accuracy, neg_accuracy, pos_accuracy, accuracy, loss_mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', type=str, default=r'./pri_database.txt')
    parser.add_argument('--label_path', type=str, default=r'./label.txt')
    parser.add_argument('--weight_path', type=str, default='./checkpoints/origin/2020-11-16T10-33-35-16-classify/classify_fold0_origin_0.959231.pth')
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    parser.add_argument('--lr', type=float, default=0.001, help='init lr')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay in optimizer')
    parser.add_argument('--n_splits', type=int, default=1, help='n_splits_fold')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument("--layer", type=list, default=[10, 15, 15])
    parser.add_argument("--method", type=str, default='origin', help='origin, 10_select or 6_select')
    config = parser.parse_args()

    dataloaders = classify_provider(config.features_path, config.label_path, config.n_splits,
                                    config.batch_size, config.num_workers)
    for fold_index, [train_loader, valid_loader, all_dataloader] in enumerate(dataloaders):
        train_val = Predict(config)
        train_val.validation(all_dataloader)
