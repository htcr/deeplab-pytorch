#!/usr/bin/env python
# coding: utf-8
#
# Author: Congrui Hetang
# URL:    https://github.com/htcr
# Date:   21 March 2019

from __future__ import absolute_import, print_function

import os
import os.path as osp

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data

from .base import _BaseDataset

import random


class Matting(_BaseDataset):
    """
    PASCAL VOC Segmentation dataset with extra annotations
    """

    def __init__(self, data_root, scales=(0.3, 0.5, 1.0), crop_size=321):
        # dir containing alpha and fg
        self.data_root = data_root
        self.scales = scales
        self.crop_size = crop_size

        self.fg_dir = osp.join(self.data_root, 'fg')
        self.alpha_dir = osp.join(self.data_root, 'alpha_filtered')
        
        self.img_list = os.listdir(self.alpha_dir)
        self.mean_bgr = np.array((128.0, 128.0, 128.0))


    def _augmentation(self, fg, alpha):

        # Scaling
        h, w = fg.shape[0:2]
        scale_factor = random.choice(self.scales)
        h, w = (int(h * scale_factor), int(w * scale_factor))
        fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

        # Padding to fit for crop_size
        h, w = fg.shape[0:2]
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            fg = cv2.copyMakeBorder(fg, value=self.mean_bgr, **pad_kwargs)
            alpha = cv2.copyMakeBorder(alpha, value=0, **pad_kwargs)

        # Cropping
        h, w = fg.shape[0:2]
        start_h = random.randint(0, h - self.crop_size)
        start_w = random.randint(0, w - self.crop_size)
        end_h = start_h + self.crop_size
        end_w = start_w + self.crop_size
        fg = fg[start_h:end_h, start_w:end_w]
        alpha = alpha[start_h:end_h, start_w:end_w]

        # Random flipping
        if random.random() < 0.5:
            fg = np.fliplr(fg).copy()
            alpha = np.fliplr(alpha).copy()
        return fg, alpha


    def __getitem__(self, index):
        img_name = self.img_list[index]
        fg_path = osp.join(self.fg_dir, img_name)
        alpha_path = osp.join(self.alpha_dir, img_name)
        
        fg = cv2.imread(fg_path, cv2.IMREAD_COLOR) # H, W, 3, [0-255]
        alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE) # H, W, [0-255]

        fg, alpha = self._augmentation(fg, alpha)

        # only for standalone visualization
        # fg = fg.astype(np.float32)
        # HWC -> CHW
        # fg = fg.transpose(2, 0, 1)

        # fg: H, W, 3, [0-255], uint8
        # alpha: H, W, [0-255], uint8
        return fg, alpha


    def __len__(self):
        return len(self.img_list)


    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Root: {}".format(self.data_root)
        return fmt_str


def test():
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import torchvision
    import yaml
    from torchvision.utils import make_grid
    from tqdm import tqdm

    kwargs = {"nrow": 10, "padding": 50}
    batch_size = 100

    dataset = Matting(
        data_root="/media/mscv1/14fecf86-bdfa-4ebd-8b47-eea4ddee198e/alphamatting_datasets/adobe_deep_matting/Combined_Dataset/Training_set/Adobe-licensed images",
        crop_size=321,
        scales=(0.5, 1.0,),
    )
    print(dataset)

    loader = data.DataLoader(dataset, batch_size=batch_size)

    for i, (images, labels) in tqdm(
        enumerate(loader), total=np.ceil(len(dataset) / batch_size), leave=False
    ):
        if i == 0:
            image = make_grid(images, pad_value=-1, **kwargs).numpy()
            image = np.transpose(image, (1, 2, 0))
            mask = np.zeros(image.shape[:2])
            mask[(image != -1)[..., 0]] = 255
            image = np.dstack((image, mask)).astype(np.uint8)

            labels = labels[:, np.newaxis, ...]
            label = make_grid(labels, pad_value=255, **kwargs).numpy()
            label_ = np.transpose(label, (1, 2, 0))[..., 0].astype(np.float32)
            label = cm.jet_r(label_ / 255.0) * 255
            mask = np.zeros(label.shape[:2])
            label[..., 3][(label_ == 255)] = 0
            label = label.astype(np.uint8)

            tiled_images = np.hstack((image, label))
            # cv2.imwrite("./docs/datasets/voc12.png", tiled_images)
            plt.imshow(np.dstack((tiled_images[..., 2::-1], tiled_images[..., 3])))
            plt.show()
            break
