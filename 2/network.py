import segmentation_models_pytorch as smp
import torch
from torch.nn import Module
from torch import nn
import torch.nn.functional as F


class Model():
    def __init__(self, model_name, encoder_weights='imagenet', class_num=4):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_num = class_num
        self.encoder_weights = encoder_weights
        if encoder_weights is None:
            print('Random initialize weights...')

    def create_model_cpu(self):
        print("Using model: {}".format(self.model_name))
        model = None

        # Uent resnet系列
        if self.model_name == 'unet_resnet34':
            model = smp.Unet('resnet34', encoder_weights=self.encoder_weights, classes=self.class_num, activation=None)
        elif self.model_name == 'unet_resnet50':
            model = smp.Unet('resnet50', encoder_weights=self.encoder_weights, classes=self.class_num, activation=None)
        # Unet resnext系列
        elif self.model_name == 'unet_resnext50_32x4d':
            model = smp.Unet('resnext50_32x4d', encoder_weights=self.encoder_weights, classes=self.class_num,
                             activation=None)
        # Unet se_resnet系列
        elif self.model_name == 'unet_se_resnet50':
            model = smp.Unet('se_resnet50', encoder_weights=self.encoder_weights, classes=self.class_num,
                             activation=None)
        # Unet se_resnext 系列
        elif self.model_name == 'unet_se_resnext50_32x4d':
            model = smp.Unet('se_resnext50_32x4d', encoder_weights=self.encoder_weights, classes=self.class_num,
                             activation=None)
        # Unet dpn 系列
        elif self.model_name == 'unet_dpn68':
            model = smp.Unet('dpn68', encoder_weights=self.encoder_weights, classes=self.class_num, activation=None)
        # Unet Efficient 系列
        elif self.model_name == 'unet_efficientnet_b4':
            model = smp.Unet('efficientnet-b4', encoder_weights=self.encoder_weights, classes=self.class_num,
                             activation=None)
        elif self.model_name == 'unet_efficientnet_b3':
            model = smp.Unet('efficientnet-b3', encoder_weights=self.encoder_weights, classes=self.class_num,
                             activation=None)

        return model

    def create_model(self):
        model = self.create_model_cpu()

        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model)

        model.to(self.device)

        return model


class ClassifyResNet(Module):
    def __init__(self, model_name, class_num=4, training=True, encoder_weights='imagenet'):
        super(ClassifyResNet, self).__init__()
        self.class_num = class_num
        model = Model(model_name, encoder_weights=encoder_weights, class_num=class_num).create_model_cpu()
        # 注意模型里面必须包含 encoder 模块
        self.encoder = model.encoder
        self.module = Module
        if model_name == 'unet_resnet34':
            self.feature = nn.Conv2d(3, 32, kernel_size=1)
        elif model_name == 'unet_resnet50':
            self.feature = nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(512, 32, kernel_size=1)
            )
        elif model_name == 'unet_se_resnext50_32x4d':
            self.feature = nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(512, 32, kernel_size=1)
            )
        elif model_name == 'unet_efficientnet_b4':
            self.feature = nn.Sequential(
                nn.Conv2d(448, 160, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(160, 32, kernel_size=1)
            )
        elif model_name == 'unet_efficientnet_b3':
            self.feature = nn.Conv2d(384, 32, kernel_size=1)

        self.logit = nn.Conv2d(32, self.class_num, kernel_size=1)

        self.training = training

    def forward(self, x):
        x = self.encoder(x)[0]
        x = F.dropout(x, 0.5, training=self.training)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.feature(x)
        logit = self.logit(x)

        return logit
