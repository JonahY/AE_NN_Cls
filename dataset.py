import os
import pandas as pd
import numpy as np
import math
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.nn import Module
from torch import optim
import datetime


def classify_provider(features_path, label_path, n_splits, batch_size, num_workers, method='origin'):
    # Amp,RiseT,Dur,Eny,RMS,Counts
    with open(features_path, 'r') as f:
        feature = np.array([i.split(',')[6:-4] for i in f.readlines()[1:]])
    feature = feature.astype(np.float32)

    with open(label_path, 'r') as f:
        label = np.array([i.strip() for i in f.readlines()[1:]])
    label = label.astype(np.float32).reshape(-1, 1)
    label[np.where(label == 2)] = 0
    ext = np.zeros([feature.shape[0], 1]).astype(np.float32)
    ext[np.where(label == 0)[0].tolist()] = 1
    label = np.concatenate((label, ext), axis=1)

    df = pd.DataFrame(feature)
    df.columns = ['Amp', 'RiseT', 'Dur', 'Eny', 'RMS', 'Counts']
    df['Counts/Dur'] = df['Counts'] / df['Dur']
    df['RiseT/Dur'] = df['RiseT'] / df['Dur']
    df['Eny/Dur'] = df['Eny'] / df['Dur']
    df['Amp*RiseT'] = df['Amp'] * df['RiseT']

    if method == '10_select':
        feature = df.values
    elif method == '6_select':
        feature = df[['Eny', 'Amp*RiseT', 'Dur', 'RMS', 'Counts/Dur', 'RiseT/Dur']].values

    train_dfs = list()
    val_dfs = list()
    all_dfs = list()
    if n_splits != 1:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=69)
        for train_df_index, val_df_index in skf.split(feature, label[:, 0]):
            train_dfs.append([feature[train_df_index], label[train_df_index, :]])
            val_dfs.append([feature[val_df_index], label[val_df_index, :]])
    else:
        df_temp = train_test_split(feature, label, test_size=0.2, stratify=label, random_state=69)
        train_dfs.append([df_temp[0], df_temp[2]])
        val_dfs.append([df_temp[1], df_temp[3]])
        all_dfs.append([np.concatenate((df_temp[0], df_temp[1]), axis=0),
                         np.concatenate((df_temp[2], df_temp[3]), axis=0)])
        # print(len(train_dfs), len(val_dfs), len(all_dfs))

    dataloaders = list()
    for df_index, (train_df, val_df, all_df) in enumerate(zip(train_dfs, val_dfs, all_dfs)):
        train_dataset = SteelClassDataset(train_df)
        val_dataset = SteelClassDataset(val_df)
        all_dataset = SteelClassDataset(all_df)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      shuffle=True)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    shuffle=False)
        all_dataloader = DataLoader(all_dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     pin_memory=True,
                                     shuffle=False)
        dataloaders.append([train_dataloader, val_dataloader, all_dataloader])
    return dataloaders


class SteelClassDataset(Dataset):
    def __init__(self, dataset):
        super(SteelClassDataset, self).__init__()
        self.feature = dataset[0]
        self.label = dataset[1]

    def __getitem__(self, idx):
        x = self.feature[idx]
        y = self.label[idx]
        return x, y

    def __len__(self):
        return len(self.label)
