import torch
from matplotlib import pyplot as plt
from torch import nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
from dataset import classify_provider
from sklearn.preprocessing import OneHotEncoder
from loss import ClassifyLoss
from meter import Meter


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
MAX_EPOCH = 50
batch_size = 64
image_size = (224, 224)
train_transform = transforms.Compose(
    [transforms.Resize([256, 256]),
     transforms.RandomCrop(224),
     transforms.RandomRotation(degrees=(-40, 40)),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

train_dataset = torchvision.datasets.ImageFolder(root='train', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_dataset = torchvision.datasets.ImageFolder(root='test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
dataloaders = classify_provider(
    '/home/Yuanbincheng/data/Ni-tension test-pure-1-0.01-AE-20201030',
    'train info_cwt.csv',
    10,
    64,
    8,
    mean,
    std,
    True,
    224,
    224
    )

ohe = OneHotEncoder()
ohe.fit([[0], [1], [2]])

model = models.shufflenet_v2_x1_0(pretrained=True)
# model = models.alexnet(pretrained=False)
model.fc = nn.Linear(1024, n_class)
# model.classifier[6] = nn.Sequential(nn.Linear(4096, n_class))

train_acc_list = []
train_loss_list = []
test_loss_list = []
test_acc_list = []
# train_loader = dataloaders[0][0]
# test_loader = dataloaders[0][1]

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
# criterion = torch.nn.CrossEntropyLoss()
criterion = ClassifyLoss()

for epoch in range(MAX_EPOCH):
    model.train()
    num_correct = 0
    num_pred = 0
    train_loss = 0
    for train_batch_x, train_batch_y in tqdm(train_loader):
        train_batch_y = torch.from_numpy(ohe.transform(train_batch_y.reshape(-1, 1)).toarray())
        train_batch_x, train_batch_y = train_batch_x.to(device), train_batch_y.to(device)
        # train_batch_y = torch.max(train_batch_y, 1)[1]

        out = model(train_batch_x)
        # print(out.shape, train_batch_y.shape)
        loss = criterion(out, train_batch_y)

        p = (out > 0.5).float()
        pred = torch.max(p, 1)[1]

        train_loss += loss.item()
        # num_correct += (pred == train_batch_y).sum().item()
        # num_pred += pred.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(train_loader)
    print("Fold: %d, Train Loss: %.7f, lr: %s" % (0, train_loss, 0.0001))
    # print('Train | epoch {}/{} | Acc: {:.6f}({}/{}) | loss:{}'.format(epoch + 1, MAX_EPOCH, (num_correct / num_pred),
    #                                                                   num_correct, num_pred, train_loss))

    # train_loss_list += [train_loss]
    # train_acc_list += [num_correct / num_pred]

    model.eval()
    num_correct = 0
    num_pred = 0
    test_loss = 0
    y_true = []
    y_pre = []
    meter = Meter()
    for test_batch_x, test_batch_y in tqdm(test_loader):
        test_batch_y = torch.from_numpy(ohe.transform(test_batch_y.reshape(-1, 1)).toarray())
        test_batch_x, test_batch_y = test_batch_x.to(device), test_batch_y.to(device)
        # test_batch_y = torch.max(test_batch_y, 1)[1]

        out = model(test_batch_x)
        loss = criterion(out, test_batch_y)

        # p = (out > 0.5).float()
        # pred = torch.max(p, 1)[1]

        meter.update(test_batch_y, out.cpu())

        # y_true.extend(test_batch_y.numpy().tolist())
        # y_pre.extend(pred.numpy().tolist())
        
        test_loss += loss.item()
        # num_correct += (pred == test_batch_y).sum().item()
        # num_pred += pred.size(0)
    test_loss = test_loss / len(test_loader)
    class_neg_accuracy, class_pos_accuracy, class_accuracy, neg_accuracy, pos_accuracy, accuracy = meter.get_metrics()
    print("Class_0_accuracy: %0.4f | Positive accuracy: %0.4f | Negative accuracy: %0.4f | \n"
          "Class_1_accuracy: %0.4f | Positive accuracy: %0.4f | Negative accuracy: %0.4f | \n"
          "Class_2_accuracy: %0.4f | Positive accuracy: %0.4f | Negative accuracy: %0.4f |" %
          (class_accuracy[0], class_pos_accuracy[0], class_neg_accuracy[0],
           class_accuracy[1], class_pos_accuracy[1], class_neg_accuracy[1],
           class_accuracy[2], class_pos_accuracy[2], class_neg_accuracy[2]))

    # print(classification_report(y_true, y_pre))
    # print('Test |Acc: {:.6f}({}/{}) | loss:{}'.format((num_correct / num_pred), num_correct, num_pred, test_loss))

    # test_loss_list += [test_loss]
    # test_acc_list += [num_correct / num_pred]

    # plot_result()
