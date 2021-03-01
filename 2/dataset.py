import os
import cv2
import warnings
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import sys
from tqdm import tqdm

sys.path.append('.')
from utils import data_augmentation, image_torch
warnings.filterwarnings("ignore")


def classify_provider(root, fold, n_splits, batch_size, num_workers, mean, std, crop, height, width):
    df = pd.read_csv(os.path.join(root, fold), header=None)
    labels_1dim = [np.argmax(np.array(df.iloc[i, :])) + 1 for i in range(df.count()[0])]

    train_dfs = list()
    val_dfs = list()
    if n_splits != 1:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=69)
        for train_df_index, val_df_index in skf.split(df, labels_1dim):
            train_dfs.append(df.loc[df.index[train_df_index]])
            val_dfs.append(df.loc[df.index[val_df_index]])
    else:
        df_temp = train_test_split(df, test_size=0.2, stratify=df, random_state=69)
        train_dfs.append(df_temp[0])
        val_dfs.append(df_temp[1])

    dataloaders = list()
    for train_df, val_df in zip(train_dfs, val_dfs):
        train_dataset = AEClassDataset(train_df, root, mean, std, 'train', crop=crop, height=height, width=width)
        val_dataset = AEClassDataset(val_df, root, mean, std, 'val')
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      num_workers=num_workers,
                                      pin_memory=False,
                                      shuffle=True)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    pin_memory=False,
                                    shuffle=False)
        dataloaders.append([train_dataloader, val_dataloader])
    return dataloaders


class AEClassDataset(Dataset):
    def __init__(self, df, data_folder, mean, std, phase, crop=False, height=None, width=None):
        super(AEClassDataset, self).__init__()
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms
        self.crop = crop
        self.height = height
        self.width = width
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        label = np.array(self.df.iloc[idx, :])
        image_path = os.path.join(self.root, "train dataset", str(idx + 1) + '.jpg')
        img = cv2.imread(image_path)
        img = self.transforms(self.phase, img, self.mean, self.std, crop=self.crop, height=self.height, width=self.width)
        return img, label

    def __len__(self):
        return len(self.fnames)


def augmentation(image, crop=False, height=None, width=None):
    image_aug = data_augmentation(image, crop=crop, height=height, width=width)
    image_aug = Image.fromarray(image_aug)

    return image_aug


def get_transforms(phase, image, mean, std, crop=False, height=None, width=None):
    if phase == 'train':
        image = augmentation(image, crop=crop, height=height, width=width)

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean, std)
    transform_compose = transforms.Compose([to_tensor, normalize])
    image = transform_compose(image)

    return image


if __name__ == "__main__":
    data_folder = r"H:\VALLEN\Ni-tension test-pure-1-0.01-AE-20201030"
    df_path = "train info.csv"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    batch_size = 12
    num_workers = 1
    n_splits = 5
    mask_only = False
    crop = False
    height = 256
    width = 512
    # 测试分割数据集
    class_dataloader = classify_provider(data_folder, df_path, n_splits, batch_size, num_workers, mean, std, crop, height, width)
    for fold_index, [classify_train_dataloader, classify_val_dataloader] in enumerate(class_dataloader):
        class_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        for [classify_images, classify_targets] in tqdm(classify_val_dataloader):
            image = classify_images[0]
            target = classify_targets[0]
            image = image_torch(image, mean, std)
            classify_target = classify_targets[0]
            position_x = 10
            for i in range(classify_target.size(0)):
                color = class_color[i]
                position_x += 50
                position = (position_x, 50)
                if classify_target[i] != 0:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    img_ = image.copy()
                    image = cv2.putText(img_, str(i+1), position, font, 1.2, color, 2)
            cv2.imshow('win', image)
            cv2.waitKey(0)
    pass
