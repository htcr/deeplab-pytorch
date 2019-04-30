#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   07 January 2019

from __future__ import absolute_import, division, print_function

import json
import multiprocessing
import os

import click
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from PIL import Image
from tensorboardX import SummaryWriter
from torchnet.meter import MovingAverageValueMeter
from tqdm import tqdm

from libs.datasets import get_dataset
from libs.models.deeplabv2_joint_bks import DeepLabV2JointBKSV2
#from libs.utils import DenseCRF, PolynomialLR, scores
from libs.utils import PolynomialLR, scores
from libs.datasets.human_and_stuff_in_dome import HumanAndStuffInDome

import torchvision
from torchvision.utils import make_grid

from evaluate import DomeSegmentationEval

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print("    {}:".format(i), torch.cuda.get_device_name(i))
    else:
        print("Device: CPU")
    return device


def get_params(model, key):
    # For Dilated FCN
    if key == "1x":
        for m in model.named_modules():
            if "layer" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == "10x":
        for m in model.named_modules():
            if "aspp" in m[0] or "upsample" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight
    # For conv bias in the ASPP module
    if key == "20x":
        for m in model.named_modules():
            if "aspp" in m[0] or "upsample" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].bias


def resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    # new_labels = torch.LongTensor(new_labels)
    new_labels = torch.FloatTensor(new_labels)
    return new_labels


@click.group()
@click.pass_context
def main(ctx):
    """
    Training and evaluation
    """
    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)
def train(config_path, cuda):
    """
    Training DeepLab by v2 protocol
    """

    # Configuration
    CONFIG = Dict(yaml.load(config_path))
    device = get_device(cuda)
    torch.backends.cudnn.benchmark = True

    dataset = HumanAndStuffInDome(CONFIG)

    # evaluator
    evaluator = DomeSegmentationEval(CONFIG, all_gray=CONFIG.TRAIN_ALL_GRAY)

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=True,
    )
    loader_iter = iter(loader)

    # Model setup
    model = DeepLabV2JointBKSV2(n_classes=CONFIG.MODEL.N_CLASSES, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])
    state_dict = torch.load(CONFIG.MODEL.INIT_MODEL)
    print("    Init:", CONFIG.MODEL.INIT_MODEL)
    for m in model.state_dict().keys():
        if m not in state_dict.keys():
            print("    Skip init:", m)


    # Process first conv
    input_conv_name = 'layer1.conv1.conv.weight'
    ori_input_conv_weight = state_dict[input_conv_name]
    OCh, ICh, K1, K2 = ori_input_conv_weight.shape
    new_input_conv_weight = torch.zeros(OCh, 6, K1, K2).float()
    new_input_conv_weight[:, :3, :, :] = ori_input_conv_weight
    
    state_dict[input_conv_name] = new_input_conv_weight
    
    model.load_state_dict(state_dict, strict=False)  # to skip ASPP
    model = nn.DataParallel(model)
    model.to(device)

    # Loss definition
    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG.SOLVER.IGNORE_LABEL)
    criterion.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(
        # cf lr_mult and decay_mult in train.prototxt
        params=[
            {
                "params": get_params(model.module, key="1x"),
                "lr": CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="10x"),
                "lr": 10 * CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="20x"),
                "lr": 20 * CONFIG.SOLVER.LR,
                "weight_decay": 0.0,
            },
        ],
        momentum=CONFIG.SOLVER.MOMENTUM,
    )

    # Learning rate scheduler
    scheduler = PolynomialLR(
        optimizer=optimizer,
        step_size=CONFIG.SOLVER.LR_DECAY,
        iter_max=CONFIG.SOLVER.ITER_MAX,
        power=CONFIG.SOLVER.POLY_POWER,
    )

    variant_name = "jointbksv2_all_gray_mat"

    # Setup loss logger
    writer = SummaryWriter(os.path.join(CONFIG.EXP.OUTPUT_DIR, "logs", CONFIG.EXP.ID, variant_name))
    average_loss = MovingAverageValueMeter(CONFIG.SOLVER.AVERAGE_LOSS)
    average_seg_loss = MovingAverageValueMeter(CONFIG.SOLVER.AVERAGE_LOSS)
    average_mat_loss = MovingAverageValueMeter(CONFIG.SOLVER.AVERAGE_LOSS)

    # Path to save models
    checkpoint_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "models",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        variant_name,
    )
    makedirs(checkpoint_dir)
    print("Checkpoint dst:", checkpoint_dir)

    # Freeze the batch norm pre-trained on COCO
    model.train()
    model.module.freeze_bn()

    for iteration in tqdm(
        range(1, CONFIG.SOLVER.ITER_MAX + 1),
        total=CONFIG.SOLVER.ITER_MAX,
        dynamic_ncols=True,
    ):

        # Clear gradients (ready to accumulate)
        optimizer.zero_grad()

        try:
            _, true_fgs, comp_fgs, bgs, labels, is_real_flags, enlarged_labels = next(loader_iter)
        except:
            loader_iter = iter(loader)
            _, true_fgs, comp_fgs, bgs, labels, is_real_flags, enlarged_labels = next(loader_iter)

        # Propagate forward
        logits, cascade_masks = model(comp_fgs.to(device), bgs.to(device))

        # Loss
        iter_loss = 0
        seg_loss = 0
        mat_loss = 0

        """
        for logit in logits:
            # Resize labels for {100%, 75%, 50%, Max} logits
            _, _, H, W = logit.shape
            labels_ = resize_labels(labels, size=(H, W))
            iter_loss += criterion(logit, labels_.to(device))
        """
        
        # segmentation loss
        _, _, lH, lW = logits.shape
        seg_labels = resize_labels(labels, size=(lH, lW))
        # seg_loss += criterion(logits, seg_labels.to(device))
        seg_loss_raw = F.cross_entropy(logits, seg_labels.to(device).long(), 
            ignore_index=CONFIG.SOLVER.IGNORE_LABEL,
            reduction='none')
        seg_loss_sample_reduced = torch.mean(seg_loss_raw, dim=(1, 2))
        #print(seg_loss_sample_reduced.shape)
        seg_loss_mask = torch.tensor(is_real_flags)
        #print(seg_loss_mask)
        seg_loss_masked = seg_loss_mask.to(device).float() * seg_loss_sample_reduced
        sum_seg_loss = torch.sum(seg_loss_masked)
        num_seg_loss = torch.sum(seg_loss_mask)
        
        if num_seg_loss.item() != 0:
            seg_loss += sum_seg_loss / num_seg_loss
        else:
            seg_loss += torch.tensor(0).float().cuda()
            

        # matting loss
        for pred_masks in cascade_masks:
            _, _, mH, mW = pred_masks.shape
            mask_labels = resize_labels(labels, size=(mH, mW))
            mask_labels = torch.unsqueeze(mask_labels, dim=1)
            mask_ignores = mask_labels == CONFIG.SOLVER.IGNORE_LABEL
            mask_labels[mask_ignores] = 0
            mask_labels = mask_labels.to(device).float()

            abs_loss_raw = torch.abs(pred_masks - mask_labels)
            #print(abs_loss_raw.shape)
            abs_loss_sample_reduced = torch.mean(abs_loss_raw, dim=(1, 2, 3))
            #print(abs_loss_sample_reduced)

            abs_loss_mask = 1 - torch.tensor(is_real_flags)
            #print(abs_loss_mask)

            abs_loss_masked = abs_loss_mask.to(device).float() * abs_loss_sample_reduced
            abs_loss = torch.sum(abs_loss_masked)
            syn_sample_cnt = torch.sum(abs_loss_mask)
        
            """
            print(abs_loss)
            print(syn_sample_cnt)
            """

            if syn_sample_cnt.item() != 0:
                mat_loss += abs_loss / syn_sample_cnt
            else:
                mat_loss += torch.tensor(0).float().cuda()

        # Propagate backward (just compute gradients wrt the loss)
        iter_loss = seg_loss + 1.0*mat_loss

        iter_loss.backward()

        if iteration % CONFIG.VIS_ITER == 0:
            # hist
            for name, param in model.module.named_parameters():
                if name == 'layer1.conv1.conv.weight':
                    writer.add_histogram(name+'_bg', param[:, 3:, :, :], iteration, bins="auto")
                    writer.add_histogram(name+'_bg', param.grad[:, 3:, :, :], iteration, bins="auto")

            # image, mask and mat
            """
            images = comp_fgs
            mean = torch.tensor((104.008, 116.669, 122.675))[None, :, None, None]
            images += mean.expand_as(images)
            """

            probs = F.softmax(logits, dim=1)
            
            show_imgs = make_grid(comp_fgs, normalize=True, scale_each=True)
            show_probs = make_grid(probs[:, 1:, :, :], normalize=True, scale_each=True)
            show_masks = [make_grid(pred_masks, normalize=True, scale_each=True) for pred_masks in cascade_masks]
            
            #print(mask_labels.shape)
            show_mask_labels = make_grid(mask_labels, normalize=True, scale_each=True)
            

            writer.add_image('img', show_imgs, iteration)
            writer.add_image('prob', show_probs, iteration)
            [writer.add_image('mask{}'.format(i), show_masks[i], iteration) for i in range(len(show_masks))]
            writer.add_image('mask_label', show_mask_labels, iteration)

            """
            for i in range(images.shape[0]):
                image = images[i, :, :, :].transpose(1, 2, 0).uint8()
                writer.add_image('img_{}'.format(i), image[:, :, ::-1], iteration)
                writer.add_image('prob_{}'.format(i), probs[i, 1, :, :], iteration)
                writer.add_image('mask_{}'.format(i), pred_masks[i, 0, :, :], iteration)
            """

        #print(loss)
        average_loss.add(iter_loss.item())
        average_mat_loss.add(mat_loss.item())
        average_seg_loss.add(seg_loss.item())

        # Update weights with accumulated gradients
        optimizer.step()

        # Update learning rate
        scheduler.step(epoch=iteration)

        # TensorBoard
        if iteration % CONFIG.SOLVER.ITER_TB == 0:
            writer.add_scalar("total_loss/train", average_loss.value()[0], iteration)
            writer.add_scalar("seg_loss/train", average_seg_loss.value()[0], iteration)
            writer.add_scalar("mat_loss/train", average_mat_loss.value()[0], iteration)

            for i, o in enumerate(optimizer.param_groups):
                writer.add_scalar("lr/group_{}".format(i), o["lr"], iteration)
            
            """
            for i in range(torch.cuda.device_count()):
                writer.add_scalar(
                    "gpu/device_{}/memory_cached".format(i),
                    torch.cuda.memory_cached(i) / 1024 ** 3,
                    iteration,
                )

            if False:
                for name, param in model.module.base.named_parameters():
                    name = name.replace(".", "/")
                    # Weight/gradient distribution
                    writer.add_histogram(name, param, iteration, bins="auto")
                    if param.requires_grad:
                        writer.add_histogram(
                            name + "/grad", param.grad, iteration, bins="auto"
                        )
            """

        if iteration % CONFIG.VAL_ITER == 0:
            iou, eval_result_list = evaluator.eval(model)
            model.train()
            model.module.freeze_bn()
            # writer.add_scalar("val_precision", avg_precision, iteration)
            # writer.add_scalar("val_recall", avg_recall, iteration)
            writer.add_scalar("val_iou", iou, iteration)
            total_fp = np.sum([item[2] for item in eval_result_list])
            total_fn = np.sum([item[3] for item in eval_result_list])
            writer.add_scalar("val_fp", total_fp, iteration)
            writer.add_scalar("val_fn", total_fn, iteration)     
        

        # Save a model
        if iteration % CONFIG.SOLVER.ITER_SAVE == 0:
            torch.save(
                model.module.state_dict(),
                os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(iteration)),
            )

    torch.save(
        model.module.state_dict(), os.path.join(checkpoint_dir, "checkpoint_final.pth")
    )


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="PyTorch model to be loaded",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)
def test(config_path, model_path, cuda):
    """
    Evaluation on validation set
    """

    # Configuration
    CONFIG = Dict(yaml.load(config_path))
    device = get_device(cuda)
    torch.set_grad_enabled(False)

    # Dataset
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.VAL,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=False,
    )
    print(dataset)

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TEST,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=False,
    )

    # Model
    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model = nn.DataParallel(model)
    model.eval()
    model.to(device)

    # Path to save logits
    logit_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "features",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.VAL,
        "logit",
    )
    makedirs(logit_dir)
    print("Logit dst:", logit_dir)

    # Path to save scores
    save_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "scores",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.VAL,
    )
    makedirs(save_dir)
    save_path = os.path.join(save_dir, "scores.json")
    print("Score dst:", save_path)

    preds, gts = [], []
    for image_ids, images, gt_labels in tqdm(
        loader, total=len(loader), dynamic_ncols=True
    ):
        # Image
        images = images.to(device)

        # Forward propagation
        logits = model(images)

        """
        # Save on disk for CRF post-processing
        for image_id, logit in zip(image_ids, logits):
            filename = os.path.join(logit_dir, image_id + ".npy")
            np.save(filename, logit.cpu().numpy())
        """

        # Pixel-wise labeling
        _, H, W = gt_labels.shape
        logits = F.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )
        probs = F.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)

        preds += list(labels.cpu().numpy())
        gts += list(gt_labels.numpy())

    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    score = scores(gts, preds, n_class=CONFIG.DATASET.N_CLASSES)

    with open(save_path, "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-j",
    "--n-jobs",
    type=int,
    default=multiprocessing.cpu_count(),
    show_default=True,
    help="Number of parallel jobs",
)
def crf(config_path, n_jobs):
    """
    CRF post-processing on pre-computed logits
    """

    # Configuration
    CONFIG = Dict(yaml.load(config_path))
    torch.set_grad_enabled(False)
    print("# jobs:", n_jobs)

    # Dataset
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.VAL,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=False,
    )
    print(dataset)

    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )

    # Path to logit files
    logit_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "features",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.VAL,
        "logit",
    )
    print("Logit src:", logit_dir)
    if not os.path.isdir(logit_dir):
        print("Logit not found, run first: python main.py test [OPTIONS]")
        quit()

    # Path to save scores
    save_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "scores",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.VAL,
    )
    makedirs(save_dir)
    save_path = os.path.join(save_dir, "scores_crf.json")
    print("Score dst:", save_path)

    # Process per sample
    def process(i):
        image_id, image, gt_label = dataset.__getitem__(i)

        filename = os.path.join(logit_dir, image_id + ".npy")
        logit = np.load(filename)

        _, H, W = image.shape
        logit = torch.FloatTensor(logit)[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8).transpose(1, 2, 0)
        prob = postprocessor(image, prob)
        label = np.argmax(prob, axis=0)

        return label, gt_label

    # CRF in multi-process
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(process)(i) for i in range(len(dataset))]
    )

    preds, gts = zip(*results)

    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    score = scores(gts, preds, n_class=CONFIG.DATASET.N_CLASSES)

    with open(save_path, "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
