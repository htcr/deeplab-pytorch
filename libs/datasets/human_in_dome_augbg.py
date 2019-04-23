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
import PIL
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import ConcatDataset
import random
import torchvision
from cv2.ximgproc import guidedFilter

from .voc import VOC, VOCAug
from .cocostuff import CocoStuff10k, CocoStuff164k
from .atr import ATR
from .lip import LIP
from .cihp import CIHP

from bg_utils.aug_bg import DomeBackgroundGeneratorAugmentAll

def get_dataset(name):
    return {
        "cocostuff10k": CocoStuff10k,
        "cocostuff164k": CocoStuff164k,
        "voc": VOC,
        "vocaug": VOCAug,
        "atr": ATR,
        "lip": LIP,
        "cihp": CIHP
    }[name]

def img_transform(image, mean_bgr):
    image = image.astype(np.float32)
    # Mean subtraction
    image -= mean_bgr
    # HWC -> CHW
    image = image.transpose(2, 0, 1)
    return image


class HumanInDome(Dataset):
    def __init__(self, CONFIG):
        super(HumanInDome, self).__init__()
        human_datasets = list()
        # Build human dataset
        for DATASET_CONFIG in CONFIG.TRAIN_DATASETS:
            dataset = get_dataset(DATASET_CONFIG.NAME)(
                root=DATASET_CONFIG.ROOT,
                split=DATASET_CONFIG.SPLIT,
                ignore_label=DATASET_CONFIG.IGNORE_LABEL,
                mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
                augment=True,
                base_size=CONFIG.IMAGE.SIZE.BASE,
                crop_size=CONFIG.IMAGE.SIZE.TRAIN,
                scales=DATASET_CONFIG.SCALES,
                flip=True,
                get_raw_item=CONFIG.GET_RAW_ITEM if CONFIG.GET_RAW_ITEM is not None else False,
                gray_aug=CONFIG.GRAY_AUG if CONFIG.GRAY_AUG is not None else True
            )
            print(dataset)
            human_datasets.append(dataset)

        self.mean_bgr = np.array((CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R))
        self.human_dataset = ConcatDataset(human_datasets)

        # Build bg dataset
        bg_dataset_root = CONFIG.BG_DATASET_ROOT
        self.bg_gen = DomeBackgroundGeneratorAugmentAll(
            bg_src_path=bg_dataset_root
        )

        """
        self.syn_bg_folder = osp.join(bg_dataset_root, 'syn_bgs')
        self.real_bg_folder = osp.join(bg_dataset_root, 'real_bgs')
        self.bg_list = os.listdir(self.real_bg_folder)
        """

        self.bg_scales = CONFIG.BG_SCALES
        self.brightness_range = CONFIG.BRIGHTNESS_RANGE

        # Synthetic image ratio
        self.syn_ratio = float(CONFIG.SYN_RATIO)

        self.enlarge_kernel = np.ones((60, 60), np.uint8)

    def __getitem__(self, index):
        image_id, image, label = self.human_dataset[index]
        crop_size = image.shape[0]

        label = np.where(label == 1, 1, 0).astype(np.uint8)

        enlarged_label = cv2.dilate(label, self.enlarge_kernel, iterations=1)

        # decide use real or syns image
        toss = random.random()
        if toss > self.syn_ratio:
            # just regular segmentation
            # random grayscale
            if random.random() < 0.5:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
                image = np.repeat(image, 3, axis=2)
            
            image = img_transform(image, self.mean_bgr)
            
            fake_bg = np.zeros(image.shape, dtype=np.float32)
            

            return image_id, image, image, fake_bg, label, True, enlarged_label
        else:
            # read a bg pair
            """
            picked_bg = self.bg_list[random.randint(0, len(self.bg_list))]
            real_bg_path = osp.join(self.real_bg_folder, picked_bg)
            syn_bg_path = osp.join(self.syn_bg_folder, picked_bg)
            real_bg = cv2.imread(real_bg_path, cv2.IMREAD_COLOR)
            syn_bg = cv2.imread(syn_bg_path, cv2.IMREAD_COLOR) 
            """
            real_bg, syn_bg, is_gray = self.bg_gen.get_patch_pair()
            bh, bw = real_bg.shape[:2]

            # is_gray = 'gray' in picked_bg
            if is_gray:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
                image = np.repeat(image, 3, axis=2)

            # augment bg
            
            # Scale
            scale_factor = random.choice(self.bg_scales)
            bh, bw = (int(bh * scale_factor), int(bw * scale_factor))
            real_bg = cv2.resize(real_bg, (bw, bh), interpolation=cv2.INTER_LINEAR)
            syn_bg = cv2.resize(syn_bg, (bw, bh), interpolation=cv2.INTER_LINEAR)
            
            # Crop
            start_h = random.randint(0, bh - crop_size)
            start_w = random.randint(0, bw - crop_size)
            end_h = start_h + crop_size
            end_w = start_w + crop_size
            real_bg = real_bg[start_h:end_h, start_w:end_w]
            syn_bg = syn_bg[start_h:end_h, start_w:end_w]

            # composite
            
            """
            # guided Filter
            mask = np.where(label == 1, 255, 0).astype(np.uint8) #.reshape(image.shape[0], image.shape[1], 1)
            guidedFilter(guide=color_image, src=mask, radius=10, eps=1e-8, dst=mask)
            mask = mask.astype(np.float32) / 255.0
            """

            mask = np.where(label == 1, 1, 0).astype(np.uint8)

            masked_fg = image * mask.reshape(image.shape[0], image.shape[1], 1)
            masked_bg = syn_bg * (1 - mask.reshape(image.shape[0], image.shape[1], 1))
            comp_image = masked_fg + masked_bg
            comp_image = comp_image.astype(np.uint8)
            
            # brightness jitter
            pil_image = PIL.Image.fromarray(comp_image)
            min_bri, max_bri = self.brightness_range
            new_brightness = random.random() * (max_bri - min_bri) + min_bri
            pil_image = torchvision.transforms.functional.adjust_brightness(pil_image, new_brightness)
            comp_image = np.array(pil_image)

            pil_image = PIL.Image.fromarray(real_bg)
            min_bri, max_bri = self.brightness_range
            new_brightness = random.random() * (max_bri - min_bri) + min_bri
            pil_image = torchvision.transforms.functional.adjust_brightness(pil_image, new_brightness)
            real_bg = np.array(pil_image)


            if not is_gray:
                if random.random() > 0.5:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
                    image = np.repeat(image, 3, axis=2)
                    
                    comp_image = cv2.cvtColor(comp_image, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
                    comp_image = np.repeat(comp_image, 3, axis=2)
                    
                if random.random() > 0.5:
                    real_bg = cv2.cvtColor(real_bg, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
                    real_bg = np.repeat(real_bg, 3, axis=2)


            # build sample
            image = img_transform(image, self.mean_bgr)
            comp_image = img_transform(comp_image, self.mean_bgr)
            real_bg = img_transform(real_bg, self.mean_bgr)
            
            return image_id, image, comp_image, real_bg, label, False, enlarged_label

    def __len__(self):
        return len(self.human_dataset)


def test():
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import torchvision
    import yaml
    from torchvision.utils import make_grid
    from tqdm import tqdm
    from addict import Dict

    kwargs = {"nrow": 10, "padding": 50}
    batch_size = 100

    CONFIG = Dict(yaml.load(open('/home/mscv1/Desktop/FRL/IntelliThresh/masker/deeplab_pytorch/configs/human_in_dome.yaml', 'r')))

    dataset = HumanInDome(CONFIG)

    loader = DataLoader(dataset, batch_size=batch_size)

    for i, (image_ids, true_fgs, images, bgs, labels, is_real_flags, enlarged_labels) in tqdm(
        enumerate(loader), total=np.ceil(len(dataset) / batch_size), leave=False
    ):
        # labels = enlarged_labels
        print(image_ids)
        print(is_real_flags)
        if i == 0:
            # images = true_fgs
            mean = torch.tensor((104.008, 116.669, 122.675))[None, :, None, None]
            images += mean.expand_as(images)
            image = make_grid(images, pad_value=-1, **kwargs).numpy()
            image = np.transpose(image, (1, 2, 0))
            mask = np.zeros(image.shape[:2])
            mask[(image != -1)[..., 0]] = 255
            image = np.dstack((image, mask)).astype(np.uint8)

            labels = labels[:, np.newaxis, ...]
            label = make_grid(labels, pad_value=255, **kwargs).numpy()
            label_ = np.transpose(label, (1, 2, 0))[..., 0].astype(np.float32)
            label = cm.jet_r(label_ / 1.0) * 255
            mask = np.zeros(label.shape[:2])
            label[..., 3][(label_ == 255)] = 0
            label = label.astype(np.uint8)

            tiled_images = np.hstack((image, label))
            # cv2.imwrite("./docs/datasets/voc12.png", tiled_images)
            plt.imshow(np.dstack((tiled_images[..., 2::-1], tiled_images[..., 3])))
            plt.show()
            break
