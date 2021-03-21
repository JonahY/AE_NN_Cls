import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import os


def get_image_info(image_dir):
    # Pytorch DataLoader就是使用PIL所读取的图像格式
    # 建议就用这种方法读取图像，当读入灰度图像时convert('')
    image_info = Image.open(image_dir).convert('RGB')
    # 数据预处理方法
    image_transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_info = image_transform(image_info)
    image_info = image_info.unsqueeze(0)
    return image_info


def get_k_layer_feature_map(feature_extractor, k, x):
    with torch.no_grad():
        for index, layer in enumerate(feature_extractor):
            x = layer(x)
            if k == index:
                return x


def show_feature_map(feature_map, k):
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.cpu().numpy()
    print(feature_map.shape)
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    for index in range(1, feature_map_num + 1):
        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index - 1], cmap='gray')
        plt.axis('off')
        if not os.path.exists("./featureMap/Layer%d" % k):
            os.mkdir("./featureMap/Layer%d" % k)
        scipy.misc.imsave("./featureMap/Layer%d/%s.png" % (k, str(index)), feature_map[index - 1])
    plt.show()


if __name__ == '__main__':
    image_dir = "/home/Yuanbincheng/data/Ni-tension test-pure-1-0.01-AE-20201030/train dataset_cwt/1.jpg"
    model = models.alexnet(pretrained=True)
    use_gpu = torch.cuda.is_available()
    image_info = get_image_info(image_dir)
    feature_extractor = model.features
    for k in [0, 3, 6, 8, 10]:
        feature_map = get_k_layer_feature_map(feature_extractor, k, image_info)
        # 定义提取第几层的feature map
        show_feature_map(feature_map, k)
