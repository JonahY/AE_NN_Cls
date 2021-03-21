import cv2
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
from torchvision import transforms
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, CLAHE, RandomRotate90, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma, OneOf,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightnessContrast, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, GaussNoise, Cutout, Rotate, Normalize, Crop, RandomCrop, Resize
)
from PIL import Image

sys.path.append('.')


def data_augmentation(phase, image, resize=256, crop_height=224, crop_width=224):
    if phase == 'train':
        transform_compose = transforms.Compose(
            [transforms.Resize(resize),
             transforms.RandomCrop((crop_height, crop_width)),
             transforms.RandomRotation(degrees=(-40, 40)),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()])
    else:
        transform_compose = transforms.Compose([transforms.Resize(crop_height), transforms.ToTensor()])

    image = transform_compose(Image.fromarray(image))

    # augmentations = Compose([
    #     Resize(256, 256),
    #     Rotate(limit=40),
    #     HorizontalFlip(),
    # ])
    #
    # if crop:
    #     assert height and width
    #     crop_aug = RandomCrop(height=height, width=width, always_apply=True)
    #     crop_sample = crop_aug(image=original_image)
    #     original_image = crop_sample['image']
    #
    # augmented = augmentations(image=original_image)
    # image_aug = augmented['image']

    return image


def image_torch(image, mean=None, std=None):
    if mean and std:
        for i in range(3):
            image[i] = image[i] * std[i]
            image[i] = image[i] + mean[i]
    image = image.permute(1, 2, 0).numpy()

    return image


def conv_image_visual(conv_image, image_weight, image_height, cy, cx, channels):
    '''
    slice off one image ande remove the image dimension
    original image is a 4d tensor[batche_size,weight,height,channels]
    '''
    conv_image = tf.slice(conv_image, (0, 0, 0, 0), (1, -1, -1, -1))
    conv_image = tf.reshape(conv_image, (image_height, image_weight, channels))
    # add a couple of pixels of zero padding around the image
    image_weight += 4
    image_height += 4
    conv_image = tf.image.resize_image_with_crop_or_pad(conv_image, image_height, image_weight)
    conv_image = tf.reshape(conv_image, (image_height, image_weight, cy, cx))
    conv_image = tf.transpose(conv_image, (2, 0, 3, 1))
    conv_image = tf.reshape(conv_image, (1, cy * image_height, cx * image_weight, 1))
    return conv_image
