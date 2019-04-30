"""
This script generate backround pairs for 
deep background removal network. Given a sequence
of pure backgrounds, it randomly pickes two frames
of a view and crops corresponding patches.
"""

import os
import os.path as osp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

from gen_img import show_img

from multiprocessing import Pool

dome_frames_root = '/home/mscv1/Desktop/FRL/dome_bg_imgs/serial'
frame_folders = os.listdir(dome_frames_root)
frame_folders.sort()
frame_num = len(frame_folders)

first_frame_path = osp.join(dome_frames_root, frame_folders[0])
view_imgs = os.listdir(first_frame_path)
view_num = len(view_imgs)


def resize_img(img, new_size):
    new_h, new_w = new_size
    new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return new_img


def get_bg_pair():
    chose_frame_id0, chose_frame_id1 = np.random.choice(frame_num, size=2)
    chose_frame_folder0, chose_frame_folder1 = frame_folders[chose_frame_id0], frame_folders[chose_frame_id1]
    #print(chose_frame_folder0)
    #print(chose_frame_folder1)

    chose_view_id = np.random.choice(view_num)
    
    chose_view_img = view_imgs[chose_view_id]
    #print(chose_view_img)


    is_gray = chose_view_img[:2] == '41'
    
    img0_path = osp.join(
        dome_frames_root, 
        chose_frame_folder0,
        chose_view_img
    )

    img1_path = osp.join(
        dome_frames_root,
        chose_frame_folder1,
        chose_view_img
    )

    img0 = cv2.imread(img0_path, cv2.IMREAD_COLOR)
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)

    return img0, img1, is_gray


def gen_patches(param):
    pid, num = param
    np.random.seed(pid)
    print('process {} started'.format(pid))

    patch_count = 0
    crop_size = 1284
    save_size = 642
    crop_per_img = 5

    root = '/home/mscv1/Desktop/FRL/synthetic_bgs/serial'
    syn_path = osp.join(root, 'syn_bgs')
    real_path = osp.join(root, 'real_bgs')
    
    if not osp.exists(syn_path):
        try:
            os.makedirs(syn_path)
        except Exception:
            pass
    
    if not osp.exists(real_path):
        try:
            os.makedirs(real_path)
        except Exception:
            pass

    while patch_count < num:
        for k in range(crop_per_img):
            syn_bg, real_bg, is_gray = get_bg_pair()
            # Cropping
            h, w = syn_bg.shape[:2]
            start_h = np.random.randint(0, h - crop_size + 1)
            start_w = np.random.randint(0, w - crop_size + 1)
            end_h = start_h + crop_size
            end_w = start_w + crop_size
            syn_patch = syn_bg[start_h:end_h, start_w:end_w, :]
            real_patch = real_bg[start_h:end_h, start_w:end_w, :]

            syn_patch = cv2.resize(syn_patch, (save_size, save_size))
            real_patch = cv2.resize(real_patch, (save_size, save_size))

            if is_gray:
                save_id = '{}_{}_gray.jpg'.format(pid, patch_count)
            else:
                save_id = '{}_{}_rgb.jpg'.format(pid, patch_count)

            save_path_syn = osp.join(syn_path, save_id)
            save_path_real = osp.join(real_path, save_id)

            cv2.imwrite(save_path_syn, syn_patch)
            cv2.imwrite(save_path_real, real_patch)
            if pid == 0:
                print('process {} saved to {}'.format(pid, save_path_syn))
                print('process {} saved to {}'.format(pid, save_path_real))

            patch_count += 1
        


if __name__ == '__main__':
    syn_bg_num = 100000
    num_process = 20
    work_per_process = syn_bg_num // num_process
    p = Pool(num_process)
    p.map(gen_patches, [(pid, work_per_process) for pid in range(num_process)])
    