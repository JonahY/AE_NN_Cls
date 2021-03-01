import os
import pandas as pd
import numpy as np
import math
import time
import argparse
import torch
import shutil


class Solver():
    def __init__(self, model, method):
        self.model = model
        self.method = method
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        # x = x.to(self.device)
        outputs = self.model(x, self.method)
        return outputs

    def cal_loss(self, targets, predicts, criterion):
        # targets = targets.to(self.device)
        return criterion(predicts, targets[:, 1].long())

    def backword(self, optimizer, loss):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def save_checkpoint(self, save_path, state, is_best):
        # torch.save(state, save_path)
        if is_best:
            torch.save(state, save_path)
            print('Saving Best Model.')
            # save_best_path = save_path.replace('.pth', '_best.pth')
            # shutil.copyfile(save_path, save_best_path)

    def load_checkpoint(self, load_path):
        if os.path.isfile(load_path):
            checkpoint = torch.load(load_path, map_location='cpu')
            # self.model.module.load_state_dict(checkpoint['state_dict'])
            print('Successfully Loaded from %s' % (load_path))
            return self.model
        else:
            raise FileNotFoundError(
                "Can not find weight file in {}".format(load_path))
