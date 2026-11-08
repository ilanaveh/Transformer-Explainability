"""
12/5/26
check evaluation performance of apvit-LRP model.
Based on code from viz_attn_apvit.py for creating model.
"""
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from PIL import ImageFilter  # for GaussianBlur
import torch.nn as nn
from scipy.special import softmax
import torch
import numpy as np
import json
from baselines.ViT.ViT_LRP import deit_base_patch16_224_apvit as apvit_LRP
import pandas as pd

import os
import sys

sys.path.insert(0, "/home/projects/bagon/ilanaveh/code/Transformers/APViT")
from mmcls.models.classifiers.pool_vit import PoolingVitClassifier
from mmcls.datasets.base_dataset import BaseDataset
from mmcls.datasets.raf import RAF
from mmcls.datasets import build_dataloader
from mmcls.apis import single_gpu_test
from mmcv.parallel import MMDataParallel

test_blur = 8

home_pth = '/home/projects/bagon/ilanaveh/code'

deit_cp_pth = os.path.join(home_pth, 'Transformers/deit/out/jobs_after_adding_seed')
deit_model_name = 'deit_blur0_BS128'

apvit_cp_pth = os.path.join(home_pth, 'Transformers/APViT/work_dirs')
# apvit_model_name = 'RAF_blur8'
apvit_model_name = 'RAF_blur8_pretrained0-8'

im_size = 112
# im_nm = 'catdog_112.png'
# im_nm = 'test_0038_112.jpg'
im_nm = 'test_1261.jpg'
# im_nm = 'test_1697_112.jpg'
sf = 8
normalize = transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

class GaussianBlur(object):
    """
    Apply Gaussian blur filter with the given sigma to the input PIL Image.
    Args:
        sigma (int): Desired Gaussian blur level sigma
    Taken from: W:\dannyh\work\code\PyTorch\vggface2_lookdir\datasets\custom_transforms.
   """

    def __init__(self, sigma):
        assert isinstance(sigma, int)
        self.sigma = sigma

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        img = img.filter(ImageFilter.GaussianBlur(
            radius=self.sigma))

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)


# transform = transforms.Compose([
#     transforms.Resize((im_size, im_size)),
#     transforms.ToTensor(),
#     normalize,
# ])
#
# if test_blur:
#     transform = transforms.Compose([GaussianBlur(test_blur)] + transform.transforms)


class IdentityHeadWithSimpleTest(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

    def simple_test(self, x, *args, **kwargs):
        return x



#
data_prefix = '/home/projects/bagon/ilanaveh/data/RAF-DB/basic/Image/aligned_224'
pipeline = [{'type': 'LoadImageFromFile'},
 {'type': 'GaussianBlur', 'sigma_min': test_blur, 'sigma_max': test_blur},
 {'type': 'Resize', 'size': 112},
 {'type': 'Normalize',
  'mean': [123.675, 116.28, 103.53],
  'std': [58.395, 57.12, 57.375],
  'to_rgb': True},
 {'type': 'ImageToTensor', 'keys': ['img']},
 {'type': 'Collect', 'keys': ['img']}]

ann_file = '/home/projects/bagon/ilanaveh/data/RAF-DB/basic/EmoLabel/test.txt'

dataset = RAF(data_prefix, pipeline, ann_file=ann_file, test_mode=True)
subset_size = 768
generator = torch.Generator().manual_seed(42)
indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
dataloader = build_dataloader(dataset,
                              samples_per_gpu=128,
                              workers_per_gpu=8,
                              shuffle=False,
                              round_up=False)

# dataset = datasets.ImageFolder(root=data_prefix, transform=transform)
# dataloader = DataLoader(dataset, batch_size=128, shuffle=False)


# Load local checkpoint:
with open(os.path.join(home_pth, 'Transformers/APViT/args_for_build_mdl.json'), 'r') as f:
    args = json.load(f)
args['head']['topk'] = (1,)  #
model = PoolingVitClassifier(**args)
model.head = IdentityHeadWithSimpleTest()
model.vit = apvit_LRP(pretrained=False, num_classes=7)
model = model.cuda()
apvit_cp = torch.load(os.path.join(apvit_cp_pth, apvit_model_name, 'epoch_40.pth'))['state_dict']
apvit_cp["vit.pos_embed"] = torch.cat(
    [apvit_cp.pop("vit.cls_pos_embed"), apvit_cp.pop("vit.patch_pos_embed")],
    dim=1,
)
apvit_cp["vit.head.weight"] = apvit_cp.pop("head.fc.weight")
apvit_cp["vit.head.bias"] = apvit_cp.pop("head.fc.bias")
missing, unexpected = model.load_state_dict(apvit_cp, strict=False)
assert ((missing == ['vit.patch_embed.proj.weight', 'vit.patch_embed.proj.bias']) and not unexpected)
model = MMDataParallel(model, device_ids=[0])  # important for moving inputs to cuda.
model.eval()

outputs = single_gpu_test(model, dataloader, track_grad=True)
outputs = [o.detach().cpu() for o in outputs]
scores = np.vstack(outputs)
scores = softmax(scores, axis=1)
pred_score = np.max(scores, axis=1)
pred_label = np.argmax(scores, axis=1)
target_label = dataset.get_gt_labels()
results = dataset.evaluate(outputs, 'accuracy', gt_labels=target_label)
file_names=[]
for i, info in enumerate(dataset.dataset.data_infos):
    if i in dataset.indices:
        file_names.append(info['img_info']['filename'])
df = pd.DataFrame({'img': file_names, 'target_label': target_label, 'pred_label': pred_label, 'correct': target_label == pred_label, 'pred_score': pred_score})