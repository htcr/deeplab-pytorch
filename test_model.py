import torch
# from libs.models.deeplabv2_joint_bks import test, DeepLabV2JointBKSV2
from libs.models.deepdiff_e2e import test
"""
state_dict = torch.load('data/models/coco/deeplabv1_resnet101/caffemodel/deeplabv1_resnet101-coco.pth')
for k, v in state_dict.items():
    if k == 'layer1.conv1.conv.weight':
        print(v.shape)
"""
test()