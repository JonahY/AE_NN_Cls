import torch
from matplotlib import pyplot as plt
from torch import nn
import torchvision
from torchvision.models import shufflenet_v2_x1_0
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import shutil


def plot_result():
    x = [i + 1 for i in range(len(train_acc_list))]
    plt.title("Accuracy")
    plt.plot(x, train_acc_list, "-b")
    plt.plot(x, test_acc_list, "-r")
    plt.legend(["train accuracy", "test accuracy"], loc='upper left')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig('./acc.png')
    plt.clf()

    x = [i + 1 for i in range(len(train_loss_list))]
    plt.title("Loss")
    plt.plot(x, train_loss_list, "-b")
    plt.plot(x, test_loss_list, "-r")
    plt.legend(["train loss", "test loss"], loc='upper left')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig('./loss.png')
    plt.clf()


device = torch.device("cpu")

n_class = 3
MAX_EPOCH = 15
batch_size = 16
image_size = (224, 224)
lr, weight_decay = 1e-4, 1e-4
create_data = False
root = '/home/Yuanbincheng/data/Ni-tension test-pure-1-0.01-AE-20201030'
fold = 'train info_cwt.csv'
n_splits = 10
train_transform = transforms.Compose(
    [transforms.Resize([256, 256]),
     transforms.RandomCrop(224),
     transforms.RandomRotation(degrees=(-40, 40)),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])


df = pd.read_csv(os.path.join(root, fold), header=None)
labels_1dim = np.argmax(np.array(df), axis=1)

if create_data:
    print('<' * 20 + ' Start creating datasets ' + '>' * 20)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=55)
    for idx, [train_df_index, val_df_index] in tqdm(enumerate(skf.split(df, labels_1dim), 1)):
        for i in train_df_index:
            shutil.copy('/home/Yuanbincheng/data/Ni-tension test-pure-1-0.01-AE-20201030/train dataset_cwt/%s.jpg' % (i + 1),
                        '/home/Yuanbincheng/project/dislocation_cls/2/3cls/train_%d/%d/' % (idx, labels_1dim[i]))
        for i in val_df_index:
            shutil.copy('/home/Yuanbincheng/data/Ni-tension test-pure-1-0.01-AE-20201030/train dataset_cwt/%s.jpg' % (i + 1),
                        '/home/Yuanbincheng/project/dislocation_cls/2/3cls/test_%d/%d/' % (idx, labels_1dim[i]))
    print('<' * 20 + ' Finish creating datasets ' + '>' * 20)

model = shufflenet_v2_x1_0(pretrained=True)
model.fc = nn.Linear(1024, n_class)

train_acc_list = []
train_loss_list = []
test_loss_list = []
test_acc_list = []

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()

for fold in range(1, 11):
    train_dataset = torchvision.datasets.ImageFolder(root='3cls/train_%d' % fold, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataset = torchvision.datasets.ImageFolder(root='3cls/test_%d' % fold, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.train()
    for epoch in range(MAX_EPOCH):
        num_correct = 0
        num_pred = 0
        train_loss = 0
        for train_batch_x, train_batch_y in tqdm(train_loader):
            train_batch_x, train_batch_y = train_batch_x.to(device), train_batch_y.to(device)
            out = model(train_batch_x)

            loss = criterion(out, train_batch_y)
            p = (out > 0.1).float()
            pred = torch.max(p, 1)[1]

            train_loss += loss.item()
            num_correct += (pred == train_batch_y).sum().item()
            num_pred += pred.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)
        print('Train fold {}/{} | epoch {}/{} | Acc: {:.6f}({}/{}) | loss:{}'.format(fold, n_splits, epoch + 1,
                                                                                     MAX_EPOCH, (num_correct / num_pred),
                                                                                     num_correct, num_pred, train_loss))

        train_loss_list += [train_loss]
        train_acc_list += [num_correct / num_pred]

        model.eval()
        num_correct = 0
        num_pred = 0
        test_loss = 0
        y_true = []
        y_pre = []
        for test_batch_x, test_batch_y in tqdm(test_loader):
            test_batch_x, test_batch_y = test_batch_x.to(device), test_batch_y.to(device)

            out = model(test_batch_x)
            loss = criterion(out, test_batch_y)
            p = (out > 0.1).float()
            pred = torch.max(p, 1)[1]

            y_true.extend(test_batch_y.numpy().tolist())
            y_pre.extend(pred.numpy().tolist())

            test_loss += loss.item()
            num_correct += (pred == test_batch_y).sum().item()
            num_pred += pred.size(0)
        test_loss = test_loss / len(test_loader)

        print(classification_report(y_true, y_pre))
        print('Test |Acc: {:.6f}({}/{}) | loss:{}'.format((num_correct / num_pred), num_correct, num_pred, test_loss))

        test_loss_list += [test_loss]
        test_acc_list += [num_correct / num_pred]

        plot_result()
