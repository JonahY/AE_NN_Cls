import os
import pandas as pd
import numpy as np
import math
import time
import argparse
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.nn import Module
from torch import optim
import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
from multiprocessing import cpu_count
from solver import Solver
from meter import Meter
from network import Classify_model
from dataset import classify_provider


class TrainVal():
    def __init__(self, config, fold):
        self.layer = config.layer
        self.method = config.method
        self.model = Classify_model(self.layer, self.method, training=True)
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.cuda()

        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.epoch = config.epoch
        self.fold = fold
        self.max_accuracy_valid = 0
        self.solver = Solver(self.model, self.method)
        self.criterion = torch.nn.MSELoss()
        # self.criterion = torch.nn.CrossEntropyLoss()

        # 初始化tensorboard
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S-%d}-classify".format(datetime.datetime.now(), fold)
        self.model_path = os.path.join(config.save_path, self.method, TIMESTAMP)
        self.writer = SummaryWriter(log_dir=os.path.join(self.model_path, TIMESTAMP))
        # if not os.path.exists(self.model_path):
        #     os.makedirs(self.model_path)

    def train(self, train_loader, test_loader):
        # optimizer = optim.SGD(self.model.module.parameters(), self.lr, weight_decay=self.weight_decay)
        optimizer = optim.Adam(self.model.module.parameters(), self.lr, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epoch + 10)
        global_step = 0

        for epoch in range(self.epoch):
            epoch += 1
            epoch_loss = 0
            self.model.train(True)

            tbar = tqdm.tqdm(train_loader, ncols=100)
            for i, (x, labels) in enumerate(tbar):
                labels_predict = self.solver.forward(x)
                labels_predict = torch.sigmoid(labels_predict)
                loss = self.solver.cal_loss(labels, labels_predict, self.criterion).float()
                epoch_loss += loss.item()
                self.solver.backword(optimizer, loss)

                params_groups_lr = str()
                for group_ind, param_group in enumerate(optimizer.param_groups):
                    params_groups_lr = params_groups_lr + 'params_group_%d' % (group_ind) + ': %.12f, ' % (
                        param_group['lr'])
                descript = "Fold: %d, Train Loss: %.7f, lr: %s" % (self.fold, loss.item(), params_groups_lr)
                tbar.set_description(desc=descript)

            lr_scheduler.step()
            global_step += len(train_loader)

            print('Finish Epoch [%d/%d], Average Loss: %.7f' % (epoch, self.epoch, epoch_loss / len(tbar)))

            class_neg_accuracy, class_pos_accuracy, class_accuracy, neg_accuracy, pos_accuracy, accuracy, loss_valid = \
                self.validation(test_loader)

            if accuracy > self.max_accuracy_valid:
                is_best = True
                self.max_accuracy_valid = accuracy
            else:
                is_best = False

            state = {
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'max_accuracy_valid': self.max_accuracy_valid,
            }

            self.solver.save_checkpoint(
                os.path.join(self.model_path, 'classify_fold%d_%s_%f.pth' % (
                self.fold, self.method, self.max_accuracy_valid)), state, is_best)
            self.writer.add_scalar('train_loss', epoch_loss / len(tbar), epoch)
            self.writer.add_scalar('valid_loss', loss_valid, epoch)
            self.writer.add_scalar('valid_accuracy', accuracy, epoch)
            self.writer.add_scalar('valid_class_0_accuracy', class_accuracy[0], epoch)
            self.writer.add_scalar('valid_class_1_accuracy', class_accuracy[1], epoch)

    def validation(self, test_loader):
        self.model.eval()
        self.model.train(False)
        meter = Meter()
        tbar = tqdm.tqdm(test_loader, ncols=100)
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
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    parser.add_argument('--lr', type=float, default=0.001, help='init lr')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay in optimizer')
    parser.add_argument('--n_splits', type=int, default=1, help='n_splits_fold')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epoch', type=int, default=150, help='epoch')
    parser.add_argument("--layer", default=[10, 10, 10])
    parser.add_argument("--method", type=str, default='6_select', help='origin, 10_select or 6_select')
    config = parser.parse_args()

    dataloaders = classify_provider(config.features_path, config.label_path, config.n_splits,
                                    config.batch_size, config.num_workers, config.method)
    for fold_index, [train_loader, valid_loader, test_loader] in enumerate(dataloaders):
        train_val = TrainVal(config, fold_index)
        train_val.train(train_loader, test_loader)
