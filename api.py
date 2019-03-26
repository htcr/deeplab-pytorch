import numpy as np
import cv2

import os
import os.path as osp

import torch
import yaml
from addict import Dict
import matplotlib.pyplot as plt

from .libs.models import *
from .libs.utils import DenseCRF

from demo import preprocessing, inference

class DeepLabV2Masker(object):
    def __init__(self, crf=True):
        cur_dir = osp.dirname(osp.realpath(__file__))
        
        config_path = osp.join(
            cur_dir,
            'configs/human.yaml'
        )
        model_path = osp.join(
            cur_dir,
            'data/models/human/deeplabv2_resnet101_msc/all_human/checkpoint_final.pth'
        )
        
        device = torch.device('cuda')
        CONFIG = Dict(yaml.load(open(config_path, 'r')))

        torch.set_grad_enabled(False)
        # CRF post-processor
        self.crf = crf
        if crf:
            self.postprocessor = DenseCRF(
                iter_max=CONFIG.CRF.ITER_MAX,
                pos_xy_std=CONFIG.CRF.POS_XY_STD,
                pos_w=CONFIG.CRF.POS_W,
                bi_xy_std=CONFIG.CRF.BI_XY_STD,
                bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
                bi_w=CONFIG.CRF.BI_W,
            )
        else:
            self.postprocessor = None
        
        self.model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(device)
        print("Model:", CONFIG.MODEL.NAME)

        self.CONFIG = CONFIG
        self.device = device
    

    def get_mask(self, image, bk):
        ori_h, ori_w = image.shape[:2]
        image, raw_image = preprocessing(image, self.device, self.CONFIG)
        
        bk = cv2.resize(bk, raw_image.shape[:2][::-1])
        
        diff = np.maximum(raw_image, bk).astype(np.float32) / (np.minimum(raw_image, bk).astype(np.float32) + 0.1)
        
        diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff)) * 255

        diff = diff.astype(np.uint8)

        raw_image = diff

        #plt.imshow(raw_image)
        #plt.show()        

        labelmap = inference(self.model, image, raw_image, self.postprocessor)
        mask = labelmap == 1
        mask = mask.astype(np.uint8) * 255
        mask = cv2.resize(mask, (ori_w, ori_h))
        mask = np.where(mask > 128, 255, 0).astype(np.uint8)
        return mask