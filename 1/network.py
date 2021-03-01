import os
import pandas as pd
import numpy as np
import math
import time
import torch
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch.nn import Module


class Classify_model(Module):
    def __init__(self, layer, method='origin', training=True):
        super(Classify_model, self).__init__()
        if method in ['origin', '6_select']:
            self.linear1 = torch.nn.Linear(15, layer[0])
            self.linear2 = torch.nn.Linear(layer[0], layer[1])
            self.linear3 = torch.nn.Linear(layer[1], 2)
        elif method == '15_select':
            self.linear1 = torch.nn.Linear(15, layer[0])
            self.linear2 = torch.nn.Linear(layer[0], layer[1])
            self.linear3 = torch.nn.Linear(layer[1], layer[2])
            self.linear4 = torch.nn.Linear(layer[2], 2)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.training = training

    def forward(self, input, method):
        if method in ['origin', '6_select']:
            # print(input.size())
            y = self.linear1(input)
            y = self.relu(y)
            y = self.linear2(y)
            y = self.relu(y)
            y = self.linear3(y)
        elif method == '19_select':
            # print(input.size())
            y = self.linear1(input)
            y = self.sigmoid(y)
            y = F.dropout(y, 0.5, training=self.training)
            y = self.linear2(y)
            y = self.sigmoid(y)
            y = F.dropout(y, 0.5, training=self.training)
            y = self.linear3(y)
            y = self.sigmoid(y)
            y = F.dropout(y, 0.5, training=self.training)
            y = self.linear4(y)
        return y
