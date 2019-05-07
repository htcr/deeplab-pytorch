import numpy as np
import cv2

import os
import os.path as osp

import torch
import yaml
from addict import Dict
import matplotlib.pyplot as plt

from .libs.models import *
from libs.models.deeplabv2_joint_bks import DeepLabV2JointBKS, DeepLabV2JointBKSV2, DeepLabV2JointBKSV2GF, DeepLabV2JointBKSV2Legacy
from libs.models.deepdiff_e2e import DeepDiffE2E
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


def crop_out(img, mask):
    new_img = img.astype(np.float32)
    gray = np.ones(new_img.shape, dtype=np.float32) * 128.0
    mask = mask[:, :, np.newaxis]
    new_img = new_img * mask +  gray * (1.0-mask)
    new_img = new_img.astype(np.uint8)
    return new_img


def debug_cascade(raw_image, logits, cascade_masks):
    H, W, C = raw_image.shape
    seg_probs = F.softmax(logits, dim=1)[0][1, :, :].cpu().numpy()
    seg_probs = cv2.resize(seg_probs, (W, H))
    seg_img = crop_out(raw_image, seg_probs)
    
    cv2.imwrite('seg_img.png', seg_img)
    
    for mask_id, mask in enumerate(cascade_masks):
        cur_mask_prob = mask[0][0, :, :].cpu().numpy()
        cur_mask_prob = cv2.resize(cur_mask_prob, (W, H))
        cur_mask_img = crop_out(raw_image, cur_mask_prob)
        cv2.imwrite('mask_img_{}.png'.format(mask_id), cur_mask_img)
        

def inference_mask(model, image, bk, raw_image=None, postprocessor=None):
    # Image -> Probability map
    logits, cascade_masks = model(image, bk)

    # debug_cascade(raw_image, logits, cascade_masks)

    pred_mask = cascade_masks[-1]
    
    probs = torch.cat((1.0 - pred_mask, pred_mask), dim=1)[0]
    probs = probs.cpu().numpy()

    _, H, W = probs.shape

    raw_image = cv2.resize(raw_image, (W, H))
    # Refine the prob map with CRF
    if postprocessor and raw_image is not None:
        probs_prev = probs
        probs = postprocessor(raw_image, probs)

    labelmap = np.argmax(probs, axis=0)
    alpha = probs[1, :, :]

    return labelmap, alpha


class DeepLabV2JointBKSMasker(object):
    def __init__(self, crf=True, device_id=0):
        cur_dir = osp.dirname(osp.realpath(__file__))
        
        config_path = osp.join(
            cur_dir,
            'configs/human_in_dome.yaml'
        )
        model_path = osp.join(
            cur_dir,
            'data/models/jointbksv2/deeplabv2_resnet101_msc/ablation_baseline/checkpoint_4000.pth'
            # 'data/models/jointbks2/deeplabv2_resnet101_msc/joint_bks/checkpoint_4000.pth'
        )
        
        print(device_id)
        device = torch.device('cuda:{}'.format(device_id))
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
        
        self.model = DeepLabV2JointBKS(n_classes=CONFIG.MODEL.N_CLASSES, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(device)

        self.CONFIG = CONFIG
        self.device = device
    

    def get_mask(self, image, bk):
        ori_h, ori_w = image.shape[:2]
        image_tensor, raw_image = preprocessing(image, self.device, self.CONFIG)
        bk_tensor, raw_bk = preprocessing(bk, self.device, self.CONFIG)
        
        """
        bk = cv2.resize(bk, raw_image.shape[:2][::-1])
        
        diff = np.maximum(raw_image, bk).astype(np.float32) / (np.minimum(raw_image, bk).astype(np.float32) + 0.1)
        
        diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff)) * 255

        diff = diff.astype(np.uint8)

        raw_image = diff
        """

        #plt.imshow(raw_image)
        #plt.show()        

        labelmap, alpha = inference_mask(self.model, image_tensor, bk_tensor, raw_image, self.postprocessor)
        
        """
        mask = labelmap == 1
        mask = mask.astype(np.uint8) * 255
        mask = cv2.resize(mask, (ori_w, ori_h))
        mask = np.where(mask > 128, 255, 0).astype(np.uint8)
        """

        alpha = cv2.resize(alpha, (ori_w, ori_h))
        mask = np.where(alpha > 0.5, 255, 0).astype(np.uint8)

        return mask


def mask_iou(mask1, mask2):
    """
    inputs shall be bools
    """
    u = mask1 + mask2
    i = mask1 * mask2
    u = np.sum(u)
    i = np.sum(i)
    
    return float(i) / u


def filter_component_by_iou(mask1, mask2, min_area=10000):
    """
    returns the component in mask1 with maximum max-iou with components from mask2
    """

    output1 = cv2.connectedComponentsWithStats(mask1, connectivity=4, ltype=cv2.CV_16U)
    num_labels1 = output1[0]
    labels1 = output1[1]
    stats1 = output1[2]
    centroids1 = output1[3]
    
    output2 = cv2.connectedComponentsWithStats(mask2, connectivity=4, ltype=cv2.CV_16U)
    num_labels2 = output2[0]
    labels2 = output2[1]
    stats2 = output2[2]
    centroids2 = output2[3]

    id_area_pair1 = [(stats1[idx, cv2.CC_STAT_AREA], idx) for idx in range(1, num_labels1)]
    remain_ids1 = [pair[1] for pair in id_area_pair1 if pair[0] > min_area]

    id_area_pair2 = [(stats2[idx, cv2.CC_STAT_AREA], idx) for idx in range(1, num_labels2)]
    remain_ids2 = [pair[1] for pair in id_area_pair2 if pair[0] > min_area]
    
    remain_comp1_id = -1
    max_max_iou = -1.0

    for remain_id1 in remain_ids1:
        comp1 = labels1 == remain_id1
        max_iou = -1.0
        for remain_id2 in remain_ids2:
            comp2 = labels2 == remain_id2
            iou = mask_iou(comp1, comp2)
            max_iou = max(max_iou, iou)
        if max_iou > max_max_iou:
            remain_comp1_id = remain_id1
            max_max_iou = max_iou

    new_mask = np.zeros(mask1.shape, dtype=np.uint8)
    new_mask[labels1 == remain_comp1_id] = 255

    return new_mask


def filter_segment_hilo(alpha, thresh_hi=0.9, thresh_lo=0.3):
    mask_hi = np.where(alpha > thresh_hi, 255, 0).astype(np.uint8)
    mask_lo = np.where(alpha > thresh_lo, 255, 0).astype(np.uint8)
    return filter_component_by_iou(mask_lo, mask_hi)


def filter_segment_hilo2(alpha, thresh_hi=0.9, thresh_lo=0.3):
    mask_lo = np.where(alpha > thresh_lo, 255, 0).astype(np.uint8)
    
    output = cv2.connectedComponentsWithStats(mask_lo, connectivity=4, ltype=cv2.CV_16U)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]

    id_area_pair = [(stats[idx, cv2.CC_STAT_AREA], idx) for idx in range(1, num_labels)]
    remain_ids = [pair[1] for pair in id_area_pair if pair[0] > 10000]

    max_hi_area = -1
    max_hi_area_id = -1
    for comp_id in remain_ids:
        comp = labels == comp_id
        comp_hi_area = np.sum(comp * (alpha > thresh_hi))
        if comp_hi_area > max_hi_area:
            max_hi_area = comp_hi_area
            max_hi_area_id = comp_id

    new_mask = np.zeros(mask_lo.shape, dtype=np.uint8)
    new_mask[labels == max_hi_area_id] = 255

    return new_mask


def filter_segment_hilo3(alpha, thresh_hi=0.9, thresh_lo=0.3):
    mask_lo = np.where(alpha > thresh_lo, 255, 0).astype(np.uint8)
    
    output = cv2.connectedComponentsWithStats(mask_lo, connectivity=4, ltype=cv2.CV_16U)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]

    id_area_pair = [(stats[idx, cv2.CC_STAT_AREA], idx) for idx in range(1, num_labels)]
    remain_id_area_pairs = [pair for pair in id_area_pair if pair[0] > 10000]
    #print(id_area_pair)
    #print(remain_id_area_pairs)

    new_mask = np.zeros(mask_lo.shape, dtype=np.uint8)
    for comp_area, comp_id in remain_id_area_pairs:
        comp = labels == comp_id
        comp_hi_area = np.sum(comp * (alpha > thresh_hi))
        #print(comp_hi_area)
        if float(comp_hi_area) / comp_area > thresh_hi:
            new_mask[comp] = 255

    return new_mask


class DeepLabV2JointBKSV2Masker(object):
    def __init__(self, crf=True, device_id=3, all_gray=True):
        cur_dir = osp.dirname(osp.realpath(__file__))
        
        config_path = osp.join(
            cur_dir,
            'configs/jointbksv2.yaml'
        )
        model_path = osp.join(
            cur_dir,
            'data/models/jointbksv2/deeplabv2_resnet101_msc/dropbg/checkpoint_11500.pth'
            # 'data/models/jointbksv2/deeplabv2_resnet101_msc/serialbg/checkpoint_5000.pth'
            # 'data/models/jointbksv2/deeplabv2_resnet101_msc/jointbksv2_all_gray/checkpoint_6000.pth'
            # 'data/models/jointbksv2/deeplabv2_resnet101_msc/jointbksv2_ablation_no_human_prior/checkpoint_4000.pth'
            # 'data/models/jointbksv2/deeplabv2_resnet101_msc/jointbksv2_baseline/checkpoint_4000.pth'
            # 'data/models/jointbksv2/deeplabv2_resnet101_msc/jointbksv2/checkpoint_4000.pth' 
            # 'data/models/jointbksv2/deeplabv2_resnet101_msc/jointbksv2_augall_enlarge2/checkpoint_50000.pth'  #'data/models/jointbksv2/deeplabv2_resnet101_msc/jointbksv2/checkpoint_4000.pth' 
            # 'data/models/jointbksv2/deeplabv2_resnet101_msc/jointbksv2_mat/checkpoint_36000.pth' 
        )
        
        print(device_id)
        device = torch.device('cuda:{}'.format(device_id))
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
        
        self.model = DeepLabV2JointBKSV2GF(n_classes=CONFIG.MODEL.N_CLASSES, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])
        # self.model = DeepLabV2JointBKSV2Legacy(n_classes=CONFIG.MODEL.N_CLASSES, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(device)

        self.CONFIG = CONFIG
        self.device = device

        self.all_gray = all_gray
    

    def get_mask(self, image, bk):
        if self.all_gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            bk = cv2.cvtColor(bk, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            bk = cv2.cvtColor(bk, cv2.COLOR_GRAY2BGR)

        ori_h, ori_w = image.shape[:2]
        image_tensor, raw_image = preprocessing(image, self.device, self.CONFIG)
        bk_tensor, raw_bk = preprocessing(bk, self.device, self.CONFIG)
        
        """
        bk = cv2.resize(bk, raw_image.shape[:2][::-1])
        
        diff = np.maximum(raw_image, bk).astype(np.float32) / (np.minimum(raw_image, bk).astype(np.float32) + 0.1)
        
        diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff)) * 255

        diff = diff.astype(np.uint8)

        raw_image = diff
        """

        #plt.imshow(raw_image)
        #plt.show()        

        labelmap, alpha = inference_mask(self.model, image_tensor, bk_tensor, raw_image, self.postprocessor)
        
        """
        mask = labelmap == 1
        mask = mask.astype(np.uint8) * 255
        mask = cv2.resize(mask, (ori_w, ori_h))
        mask = np.where(mask > 128, 255, 0).astype(np.uint8)
        """

        alpha = cv2.resize(alpha, (ori_w, ori_h))
        # mask = np.where(alpha > 0.9, 255, 0).astype(np.uint8)

        mask = filter_segment_hilo3(alpha, thresh_hi=0.9, thresh_lo=0.3)

        return mask