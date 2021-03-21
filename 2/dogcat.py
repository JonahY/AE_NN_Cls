# coding=utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
import shutil


# 随机种子设置
random_state = 42
np.random.seed(random_state)

# kaggle原始数据集地址
original_dataset_dir = r'D:\data\archive\training_set\training_set'
total_num = 4000
random_idx = np.array(range(total_num))
np.random.shuffle(random_idx)

# 待处理的数据集地址
base_dir = r'D:\data\archive\data2'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# 训练集、测试集的划分
sub_dirs = ['train', 'test']
animals = ['cats', 'dogs']
train_idx = random_idx[:int(total_num * 0.9)]
test_idx = random_idx[int(total_num * 0.9):]
numbers = [train_idx, test_idx]
for idx, sub_dir in enumerate(sub_dirs):
    dir = os.path.join(base_dir, sub_dir)
    if not os.path.exists(dir):
        os.mkdir(dir)
    for animal in animals:
        animal_dir = os.path.join(dir, animal)  #
        if not os.path.exists(animal_dir):
            os.mkdir(animal_dir)
        fnames = [animal[:-1] + '.{}.jpg'.format(i+1) for i in numbers[idx]]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, animal, fname)
            dst = os.path.join(animal_dir, fname)
            shutil.copyfile(src, dst)

        # 验证训练集、验证集、测试集的划分的照片数目
        print(animal_dir + ' total images : %d' % (len(os.listdir(animal_dir))))

