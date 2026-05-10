"""
6/5/26
Based on DeiT_example.ipynb.
"""

from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import os
import json
from samples.CLS2IDX import CLS2IDX, CLS2IDX_RAFDB

from baselines.ViT.ViT_LRP import deit_base_patch16_224 as vit_LRP
from baselines.ViT.ViT_LRP import deit_base_patch16_224_apvit as apvit_LRP
from baselines.ViT.ViT_explanation_generator import LRP
from modules.layers_apvit import LinearClsHeadLRP

import sys
sys.path.insert(0, "/home/projects/bagon/ilanaveh/code/Transformers/APViT")
from mmcls.models.classifiers.pool_vit import PoolingVitClassifier

choose_model = 'apvit'  # 'deit' / 'apvit'

home_pth = '/home/projects/bagon/ilanaveh/code'

deit_cp_pth = os.path.join(home_pth, 'Transformers/deit/out/jobs_after_adding_seed')
deit_model_name = 'deit_blur0_BS128'

apvit_cp_pth = os.path.join(home_pth, 'Transformers/APViT/work_dirs')
apvit_model_name = 'RAF_blur0-8'

if choose_model == 'deit':
    im_size = 224
    im_nm = 'catdog.png'
    sf = 16
elif choose_model == 'apvit':
    im_size = 112
    # im_nm = 'catdog_112.png'
    im_nm = 'test_0038_112.jpg'
    sf = 8

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    normalize,
])


# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


# initialize ViT pretrained with DeiT

# Load local checkpoint:
if choose_model == 'deit':
    model = vit_LRP(pretrained=True).cuda()
    deit_cp = torch.load(os.path.join(deit_cp_pth, deit_model_name, 'best_checkpoint.pth'))
    model.load_state_dict(deit_cp['model'])

elif choose_model == 'apvit':
    with open(os.path.join(home_pth, 'Transformers/APViT/args_for_build_mdl.json'), 'r') as f:
        args = json.load(f)
    args['head']['topk'] = (1, )  #
    model = PoolingVitClassifier(**args)
    # head_type = args['head'].pop('type')
    # model.head = LinearClsHeadLRP(**args['head'])
    model.head = nn.Identity()
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
    assert((missing == ['vit.patch_embed.proj.weight', 'vit.patch_embed.proj.bias']) and not unexpected)

model.eval()

attribution_generator = LRP(model)


def generate_visualization(original_image, class_index=None, return_loss=None):
    if return_loss:
        transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(),
                                                                     method="transformer_attribution",
                                                                     index=class_index,
                                                                     return_loss=return_loss).detach()
    else:
        transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(),
                                                                     method="transformer_attribution",
                                                                     index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=sf, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(im_size, im_size).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
                transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
                image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def print_top_classes(predictions, dataset='imagenet', **kwargs):
    # Print Top-5 predictions
    cls2idx = CLS2IDX if (dataset=='imagenet') else CLS2IDX_RAFDB
    if not torch.is_tensor(predictions):
        predictions = torch.tensor(predictions)
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
    max_str_len = 0
    class_names = []
    for cls_idx in class_indices:
        class_names.append(cls2idx[cls_idx])
        if len(cls2idx[cls_idx]) > max_str_len:
            max_str_len = len(cls2idx[cls_idx])

    print('Top 5 classes:')
    for cls_idx in class_indices:
        output_string = '\t{} : {}'.format(cls_idx, CLS2IDX[cls_idx])
        output_string += ' ' * (max_str_len - len(CLS2IDX[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)


image = Image.open(f'samples/{im_nm}')
dog_cat_image = transform(image)

fig, axs = plt.subplots(1, 3)
axs[0].imshow(image);
axs[0].axis('off');

if choose_model == 'deit':
    output = model(dog_cat_image.unsqueeze(0).cuda())
    print_top_classes(output)
elif choose_model == 'apvit':
    output = model(dog_cat_image.unsqueeze(0).cuda(), return_loss=False)
    print_top_classes(output, 'raf')

# dog
# generate visualization for class 243: 'bull mastiff' - the predicted class
if choose_model == 'deit':
    dog = generate_visualization(dog_cat_image)
    cat = generate_visualization(dog_cat_image, class_index=282)
    axs[2].imshow(cat);
    axs[2].axis('off');
elif choose_model == 'apvit':
    dog = generate_visualization(dog_cat_image, return_loss=False)
    cat = generate_visualization(dog_cat_image, class_index=282, return_loss=False)

axs[1].imshow(dog)
axs[1].axis('off')

plt.show(block=True)
