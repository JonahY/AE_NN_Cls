from torch import optim
import torch
from tqdm import tqdm
from solver import Solver
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import datetime
import os
import codecs, json
import time
import argparse

from cal_classify_accuracy import Meter
from set_seed import seed_torch
import pickle
import random
from network import ClassifyResNet
from dataset import classify_provider
from loss import ClassifyLoss
from ResNeXt import ResNeXt
# from Alexnet import AlexNet
from alexnet_pytorch import AlexNet
from VGG import *
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class TrainVal():
    def __init__(self, config):
        # self.model = ClassifyResNet(config.model_name, config.class_num, training=True)
        # self.model = AlexNet(config.class_num)
        self.model = AlexNet.from_pretrained('alexnet', num_classes=config.class_num)
        # self.model = VGG_19(config.class_num)
        if torch.cuda.is_available():
            self.device = torch.device("cuda:%i" % config.device[0])
            self.model = torch.nn.DataParallel(self.model, device_ids=config.device)
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)

        # 加载超参数
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.epoch = config.epoch

        # 实例化实现各种子函数的 solver 类
        self.solver = Solver(self.model, self.device)

        # 加载损失函数
        self.criterion = ClassifyLoss(weight=[0.8, 0.2])

        # 创建保存权重的路径
        self.TIME = "{0:%Y-%m-%dT%H-%M-%S}-classify".format(datetime.datetime.now())
        self.model_path = os.path.join(config.root, config.save_path, config.model_name, self.TIME)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.max_accuracy_valid = 0
        # 设置随机种子，注意交叉验证部分划分训练集和验证集的时候，要保持种子固定
        self.seed = int(time.time())
        # self.seed = 1570421136
        seed_torch(self.seed)

    def train(self, dataloaders):
        # optimizer = optim.Adam(self.model.module.parameters(), self.lr, weight_decay=self.weight_decay)
        optimizer = optim.SGD(self.model.module.parameters(), self.lr, momentum=0.9,
                              weight_decay=self.weight_decay, nesterov=True)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epoch + 50)
        global_step = 0

        for fold_index, [train_loader, valid_loader] in enumerate(dataloaders):

            # 保存json文件和初始化tensorboard
            TIMESTAMP = '-fold'.join([self.TIME, str(fold_index)])
            self.writer = SummaryWriter(log_dir=os.path.join(self.model_path, TIMESTAMP))
            with codecs.open(os.path.join(self.model_path, TIMESTAMP, TIMESTAMP) + '.json', 'w', "utf-8") as json_file:
                json.dump({k: v for k, v in config._get_kwargs()}, json_file, ensure_ascii=False)

            with open(os.path.join(self.model_path, TIMESTAMP, TIMESTAMP) + '.pkl', 'wb') as f:
                pickle.dump({'seed': self.seed}, f, -1)

            for epoch in range(self.epoch):
                epoch += 1
                epoch_loss = 0
                self.model.train(True)

                tbar = tqdm(train_loader)
                for i, (images, labels) in enumerate(tbar):
                    # 网络的前向传播与反向传播
                    labels_predict = self.solver.forward(images)
                    labels_predict = labels_predict.unsqueeze(dim=2).unsqueeze(dim=3)
                    loss = self.solver.cal_loss(labels, labels_predict, self.criterion)
                    # loss = F.cross_entropy(labels_predict[0], labels)
                    epoch_loss += loss.item()
                    self.solver.backword(optimizer, loss)

                    # 保存到tensorboard，每一步存储一个
                    self.writer.add_scalar('train_loss', loss.item(), global_step + i)
                    # self.writer.add_images('my_image_batch', labels_predict[1].cpu().detach().numpy(), global_step + i)
                    params_groups_lr = str()
                    for group_ind, param_group in enumerate(optimizer.param_groups):
                        params_groups_lr = params_groups_lr + 'params_group_%d' % (group_ind) + ': %.12f, ' % (
                        param_group['lr'])
                    descript = "Fold: %d, Train Loss: %.7f, lr: %s" % (fold_index, loss.item(), params_groups_lr)
                    tbar.set_description(desc=descript)

                # 每一个epoch完毕之后，执行学习率衰减
                lr_scheduler.step()
                global_step += len(train_loader)

                # 验证模型
                class_neg_accuracy, class_pos_accuracy, class_accuracy, neg_accuracy, pos_accuracy, accuracy, loss_valid = \
                    self.validation(valid_loader)

                # Print the log info
                print('Finish Epoch [%d/%d] | Average training Loss: %.7f | Average validation Loss: %.7f | Total accuracy: %0.4f |' % (
                epoch, self.epoch, epoch_loss / len(tbar), loss_valid, accuracy))

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

                self.solver.save_checkpoint(os.path.join(self.model_path, TIMESTAMP, TIMESTAMP + '.pth'), state,
                                            is_best, self.max_accuracy_valid)
                self.writer.add_scalar('valid_loss', loss_valid, epoch)
                self.writer.add_scalar('valid_accuracy', accuracy, epoch)
                self.writer.add_scalar('valid_class_0_accuracy', class_accuracy[0], epoch)
                self.writer.add_scalar('valid_class_1_accuracy', class_accuracy[1], epoch)
                # self.writer.add_scalar('valid_class_2_accuracy', class_accuracy[2], epoch)

    def validation(self, valid_loader):
        self.model.eval()
        meter = Meter()
        tbar = tqdm(valid_loader)
        loss_sum = 0

        with torch.no_grad():
            for i, (images, labels) in enumerate(tbar):
                # 完成网络的前向传播
                labels_predict = self.solver.forward(images)
                labels_predict = labels_predict.unsqueeze(dim=2).unsqueeze(dim=3)
                loss = self.solver.cal_loss(labels, labels_predict, self.criterion)
                # loss = F.cross_entropy(labels_predict[0], labels)
                loss_sum += loss.item()

                meter.update(labels, labels_predict.cpu())

                descript = "Val Loss: {:.7f}".format(loss.item())
                tbar.set_description(desc=descript)
        loss_mean = loss_sum / len(tbar)

        class_neg_accuracy, class_pos_accuracy, class_accuracy, neg_accuracy, pos_accuracy, accuracy = meter.get_metrics()
        print("Class_0_accuracy: %0.4f | Positive accuracy: %0.4f | Negative accuracy: %0.4f | \n"
              "Class_1_accuracy: %0.4f | Positive accuracy: %0.4f | Negative accuracy: %0.4f |"%
              (class_accuracy[0], class_pos_accuracy[0], class_neg_accuracy[0],
               class_accuracy[1], class_pos_accuracy[1], class_neg_accuracy[1]))
        # print("Class_0_accuracy: %0.4f | Positive accuracy: %0.4f | Negative accuracy: %0.4f | \n"
        #       "Class_1_accuracy: %0.4f | Positive accuracy: %0.4f | Negative accuracy: %0.4f | \n"
        #       "Class_2_accuracy: %0.4f | Positive accuracy: %0.4f | Negative accuracy: %0.4f |" %
        #       (class_accuracy[0], class_pos_accuracy[0], class_neg_accuracy[0],
        #        class_accuracy[1], class_pos_accuracy[1], class_neg_accuracy[1],
        #        class_accuracy[2], class_pos_accuracy[2], class_neg_accuracy[2]))
        return class_neg_accuracy, class_pos_accuracy, class_accuracy, neg_accuracy, pos_accuracy, accuracy, loss_mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='epoch')
    parser.add_argument('--n_splits', type=int, default=10, help='n_splits_fold')
    parser.add_argument("--device", type=int, nargs='+', default=[i for i in range(torch.cuda.device_count())])
    parser.add_argument('--crop', type=bool, default=False, help='if true, crop image to [height, width].')
    parser.add_argument('--height', type=int, default=None, help='the height of cropped image')
    parser.add_argument('--width', type=int, default=None, help='the width of cropped image')
    # model set
    parser.add_argument('--model_name', type=str, default='unet_resnet34',
                        help='unet_resnet34/unet_se_resnext50_32x4d/unet_efficientnet_b4'
                             '/unet_resnet50/unet_efficientnet_b4')
    # model hyper-parameters
    parser.add_argument('--class_num', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5, help='init lr')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay in optimizer')
    # dataset
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--root', type=str, default='/home/Yuanbincheng/data/Ni-tension test-pure-1-0.01-AE-20201030')
    parser.add_argument('--fold', type=str, default='train info_cwt_-noise.csv')
    config = parser.parse_args()
    print(config)

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    dataloaders = classify_provider(
        config.root,
        config.fold,
        config.n_splits,
        config.batch_size,
        config.num_workers,
        mean,
        std,
        config.crop,
        config.height,
        config.width
        )

    train_val = TrainVal(config)
    train_val.train(dataloaders)