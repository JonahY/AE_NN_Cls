import cv2
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, CLAHE, RandomRotate90, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma, OneOf,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightnessContrast, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,Cutout,Rotate, Normalize, Crop, RandomCrop
)

sys.path.append('.')


def data_augmentation(original_image, crop=False, height=None, width=None):
    augmentations = Compose([
        HorizontalFlip(p=0.4),
        VerticalFlip(p=0.4),
        ShiftScaleRotate(shift_limit=0.07, rotate_limit=0, p=0.4),
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