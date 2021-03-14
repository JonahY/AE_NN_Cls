import cv2
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, CLAHE, RandomRotate90, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma, OneOf,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightnessContrast, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, GaussNoise, Cutout, Rotate, Normalize, Crop, RandomCrop, Resize
)

sys.path.append('.')


def data_augmentation(original_image, crop=False, height=None, width=None):
    augmentations = Compose([
        Resize(224, 224),
        # 直方图均衡化
        CLAHE(p=0.3),

        # 亮度、对比度
        RandomGamma(gamma_limit=(80, 120), p=0.1),
        RandomBrightnessContrast(p=0.1),

        # 模糊
        OneOf([
            MotionBlur(p=0.1),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.3),

        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2)
    ])

    if crop:
        assert height and width
        crop_aug = RandomCrop(height=height, width=width, always_apply=True)
        crop_sample = crop_aug(image=original_image)
        original_image = crop_sample['image']

    augmented = augmentations(image=original_image)
    image_aug = augmented['image']

    return image_aug


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
