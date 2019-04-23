import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import os


def cv2_to_pil(img_cv2):
    img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    return img_pil


def pil_to_cv2(img_pil):
    img = np.asarray(img_pil)
    img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_cv2


def demo_color_jitter():
    transform = T.ColorJitter(
        brightness=(0.2, 1.5),
        contrast=0.5,
        saturation=0.5, 
        hue=0.05
    )

    img_path = '/home/mscv1/Desktop/FRL/synthetic_bgs/bg_patches/patch_0_4.png'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    img = cv2_to_pil(img)

    sample_num = 10
    output_path = '/home/mscv1/Desktop/FRL/synthetic_bgs/aug_bg'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in range(sample_num):
        img_pil_trans = transform(img)
        img_cv2 = pil_to_cv2(img_pil_trans)
        save_name = 'aug_{}.png'.format(i)
        save_path = os.path.join(output_path, save_name)
        cv2.imwrite(save_path, img_cv2)


class DomeBackgroundGenerator(object):
    """
    Return augmented background patches as cv2 BGR format.
    """
    def __init__(self, 
        bg_src_path,
        init_rescale_factor=0.5,
        crop_size=642):
        
        self.bg_src_path = bg_src_path
        self.init_rescale_factor = init_rescale_factor
        self.crop_size = crop_size
        
        # load all bg views into memory
        self.bg_view_ids = os.listdir(self.bg_src_path)
        self.is_gray = [view_id[:2] == '41' for view_id in self.bg_view_ids]
        
        self.bg_imgs = list()
        for view_id in self.bg_view_ids:
            bg_path = os.path.join(self.bg_src_path, view_id)
            print(bg_path)
            bg_img = cv2.imread(bg_path, cv2.IMREAD_COLOR)
            bg_img = cv2_to_pil(bg_img)
            self.bg_imgs.append(bg_img)
            print('load {}'.format(view_id)) 

        self.transform_crop = T.RandomCrop(
            size=self.crop_size)
        self.transform_color = T.ColorJitter(
            brightness=0, #=(0.2, 1.5),
            contrast=0.5,
            saturation=0.5, 
            hue=0.05
        )

    def get_patch_pair(self):
        select_bg_id = np.random.randint(low=0, high=len(self.bg_view_ids))
        bg_img = self.bg_imgs[select_bg_id]
        bg_patch = self.transform_crop(bg_img)
        bg_patch_aug = self.transform_color(bg_patch)

        bg_patch_cv2 = pil_to_cv2(bg_patch)
        bg_patch_aug_cv2 = pil_to_cv2(bg_patch_aug)  
        is_gray = self.is_gray[select_bg_id]

        return bg_patch_cv2, bg_patch_aug_cv2, is_gray
        

class DomeBackgroundGeneratorAugmentAll(object):
    """
    Return augmented background patches as cv2 BGR format.
    """
    def __init__(self, 
        bg_src_path,
        init_rescale_factor=0.5,
        crop_size=642):
        
        self.bg_src_path = bg_src_path
        self.init_rescale_factor = init_rescale_factor
        self.crop_size = crop_size
        
        # load all bg views into memory
        self.bg_view_ids = os.listdir(self.bg_src_path)
        self.is_gray = [view_id[:2] == '41' for view_id in self.bg_view_ids]
        
        self.bg_imgs = list()
        for view_id in self.bg_view_ids:
            bg_path = os.path.join(self.bg_src_path, view_id)
            print(bg_path)
            bg_img = cv2.imread(bg_path, cv2.IMREAD_COLOR)
            H, W = bg_img.shape[0:2]
            H, W = int(H*init_rescale_factor), int(W*init_rescale_factor)
            
            bg_img = cv2.resize(bg_img, (W, H))
            
            bg_img = cv2_to_pil(bg_img)
            self.bg_imgs.append(bg_img)
            print('load {}'.format(view_id)) 

        self.transform_crop = T.RandomCrop(
            size=self.crop_size)
        self.transform_color = T.ColorJitter(
            brightness=0, #=(0.2, 1.5),
            contrast=0.5,
            saturation=0.5, 
            hue=0.05
        )

    def get_patch_pair(self):
        select_bg_id = np.random.randint(low=0, high=len(self.bg_view_ids))
        bg_img = self.bg_imgs[select_bg_id]
        bg_patch = self.transform_crop(bg_img)
        bg_patch_aug1 = self.transform_color(bg_patch)
        bg_patch_aug2 = self.transform_color(bg_patch)

        bg_patch_aug1_cv2 = pil_to_cv2(bg_patch_aug1)
        bg_patch_aug2_cv2 = pil_to_cv2(bg_patch_aug2)  
        is_gray = self.is_gray[select_bg_id]

        return bg_patch_aug1_cv2, bg_patch_aug2_cv2, is_gray


if __name__ == '__main__':
    bg_gen = DomeBackgroundGeneratorAugmentAll(
        bg_src_path='/home/mscv1/Desktop/FRL/dome_bg_imgs/real_bg_imgs/debug'
    )

    sample_num = 20
    output_path = '/home/mscv1/Desktop/FRL/synthetic_bgs/aug_pairs'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in range(sample_num):
        bg_patch, bg_patch_aug, is_gray = bg_gen.get_patch_pair()
        
        save_name = '{}_{}_ori.png'.format(i, ('gray' if is_gray else 'rgb'))
        save_path = os.path.join(output_path, save_name)
        cv2.imwrite(save_path, bg_patch)

        save_name = '{}_{}_aug.png'.format(i, ('gray' if is_gray else 'rgb'))
        save_path = os.path.join(output_path, save_name)
        cv2.imwrite(save_path, bg_patch_aug)
        