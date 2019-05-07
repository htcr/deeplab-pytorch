
import os
import os.path as osp
from evaluate import DomeSegmentationEval
from addict import Dict
import torch
import torch.nn as nn
import yaml
from libs.models.deeplabv2_joint_bks import DeepLabV2JointBKSV2, DeepLabV2JointBKSV2GF
from evaluate import DomeSegmentationEval
import matplotlib.pyplot as plt
import numpy as np

cur_dir = osp.dirname(osp.realpath(__file__))
        
config_path = osp.join(
    cur_dir,
    'configs/jointbksv2_dropbg.yaml'
)
model_path = osp.join(
    cur_dir,
    'data/models/jointbksv2/deeplabv2_resnet101_msc/dropbg/checkpoint_11500.pth'
    # 'data/models/jointbksv2/deeplabv2_resnet101_msc/jointbksv2_ablation_baseline_eval/checkpoint_33000.pth'
    # 'data/models/jointbksv2/deeplabv2_resnet101_msc/jointbksv2_ablation_no_human_prior/checkpoint_4000.pth'
    # 'data/models/jointbksv2/deeplabv2_resnet101_msc/jointbksv2_baseline/checkpoint_4000.pth'
    # 'data/models/jointbksv2/deeplabv2_resnet101_msc/jointbksv2/checkpoint_4000.pth' 
    # 'data/models/jointbksv2/deeplabv2_resnet101_msc/jointbksv2_augall_enlarge2/checkpoint_50000.pth'  #'data/models/jointbksv2/deeplabv2_resnet101_msc/jointbksv2/checkpoint_4000.pth' 
)

device = torch.device('cuda')
CONFIG = Dict(yaml.load(open(config_path, 'r')))

torch.set_grad_enabled(False)

model = DeepLabV2JointBKSV2(n_classes=CONFIG.MODEL.N_CLASSES, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])
# model = DeepLabV2JointBKSV2GF(n_classes=CONFIG.MODEL.N_CLASSES, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])
state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

for key, value in state_dict.items():
    print(key)

model.load_state_dict(state_dict)
model = nn.DataParallel(model)
model.eval()
model.to(device)

evaluator = DomeSegmentationEval(CONFIG, all_gray=CONFIG.TRAIN_ALL_GRAY)
iou, eval_result_list = evaluator.eval(model)
for item in eval_result_list:
    precision, recall, fp_area, fn_area, tp_area, err_map, sample_id = item
    print('id :{} precision: {} recall: {} fp_area: {} fn_area: {}'.format(
        sample_id, precision, recall, fp_area, fn_area
    ))
    plt.imshow(err_map[:, :, ::-1])
    plt.show()
total_fp = np.sum([item[2] for item in eval_result_list])
total_fn = np.sum([item[3] for item in eval_result_list])

print('iou: {} Total FP: {} Total FN: {}'.format(
    iou, total_fp, total_fn
))