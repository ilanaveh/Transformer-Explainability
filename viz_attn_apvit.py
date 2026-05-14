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
from PIL import ImageFilter  # for GaussianBlur

import sys

sys.path.insert(0, "/home/projects/bagon/ilanaveh/code/Transformers/APViT")
from mmcls.models.classifiers.pool_vit import PoolingVitClassifier

choose_model = 'apvit'  # 'deit' / 'apvit'
test_blur = 8

home_pth = '/home/projects/bagon/ilanaveh/code'

deit_cp_pth = os.path.join(home_pth, 'Transformers/deit/out/jobs_after_adding_seed')
deit_model_name = 'deit_blur0_BS128'

apvit_cp_pth = os.path.join(home_pth, 'Transformers/APViT/work_dirs')
# apvit_model_name = 'RAF_blur8_freeze'
apvit_model_name = 'RAF_blur8_pretrained0-8_freeze_fix_projs'

if choose_model == 'deit':
    im_size = 224
    im_nm = 'catdog.png'
    sf = 16
    # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

elif choose_model == 'apvit':
    im_size = 112
    # im_nm = 'catdog_112.png'
    # im_nm = 'test_0038_112.jpg'
    # im_nm = 'test_1261.jpg'
    im_nm = 'test_0411.jpg'
    im_nm = 'test_0189.jpg'
    # im_nm = 'test_1697_112.jpg'
    im_nm = f'test_images_apvit/{im_nm}'
    sf = 8
    # normalize = transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    mean = np.array([123.675, 116.28, 103.53])/255
    std = np.array([58.395, 57.12, 57.375])/255
    # normalize = transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])

normalize = transforms.Normalize(mean=mean, std=std)
unnormalize = transforms.Normalize(mean=(-mean / std).tolist(), std=(1.0 / std).tolist())

img_lbl_dict = {
    'test_0038_112.jpg': 4,
    'test_1261_112.jpg': 5,
    'test_1697_112.jpg': 4,
    'test_1261.jpg': 5,
    'test_0411.jpg': 4,
    'test_0189.jpg': 5,
}


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


transform = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    normalize,
])

if test_blur:
    transform = transforms.Compose([GaussianBlur(test_blur)] + transform.transforms)


class IdentityHeadWithSimpleTest(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

    def simple_test(self, x, *args, **kwargs):
        return x


# get image after transforms, and prepare it for visualization
def show_im_trans(img):
    img = img.permute(1, 2, 0).data.cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    img = np.float32(img)
    img = img / np.max(img)
    img = np.uint8(255 * img)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# create heatmap from mask on image
def show_cam_on_image(img, mask, add_alpha=False, threshold=0.3, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    if add_alpha:

        # Alpha is 0 below threshold, grows above threshold
        alpha_map = (mask >= threshold).astype(np.float32)

        # [H, W] -> [H, W, 1]
        alpha_map = alpha_map[..., None]
        cam = img * (1 - alpha_map) + heatmap * alpha_map
        cam = np.clip(cam, 0, 1)
    else:
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
    assert((missing == ['vit.patch_embed.proj.weight', 'vit.patch_embed.proj.bias']) and not unexpected)

model.eval()

attribution_generator = LRP(model)


def generate_visualization(original_image, class_index=None, return_loss=None, add_alpha=False):
    if return_loss is not None:
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
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution, add_alpha=add_alpha)
    vis = np.uint8(255 * vis)
    # vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def print_top_classes(predictions, dataset='imagenet', **kwargs):
    # Print Top-5 predictions
    cls2idx = CLS2IDX if (dataset == 'imagenet') else CLS2IDX_RAFDB
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
        output_string = '\t{} : {}'.format(cls_idx, cls2idx[cls_idx])
        output_string += ' ' * (max_str_len - len(cls2idx[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)


fig, axs = plt.subplots(1, 4)

image = Image.open(f'samples/{im_nm}')
im_trans = transform(image)

im_unnorm = unnormalize(im_trans)
im_trans_show = show_im_trans(im_unnorm)

if choose_model == 'deit':
    output = model(im_trans.unsqueeze(0).cuda())
    print_top_classes(output)
elif choose_model == 'apvit':
    output = model(im_trans.unsqueeze(0).cuda(), return_loss=False)
    print_top_classes(output, 'raf')

# dog
# generate visualization for class 243: 'bull mastiff' - the predicted class
if choose_model == 'deit':
    dog = generate_visualization(im_trans)
    cat = generate_visualization(im_trans, class_index=282)
    axs[3].imshow(cat)
    axs[3].axis('off')
elif choose_model == 'apvit':
    prd = generate_visualization(im_trans, return_loss=False)
    tru = generate_visualization(im_trans, class_index=img_lbl_dict[im_nm.replace('test_images_apvit/', '')], return_loss=False)
    ang = generate_visualization(im_trans, class_index=0, return_loss=False)
    dsg = generate_visualization(im_trans, class_index=1, return_loss=False)
    frt = generate_visualization(im_trans, class_index=2, return_loss=False)
    sad = generate_visualization(im_trans, class_index=3, return_loss=False)
    hpy = generate_visualization(im_trans, class_index=4, return_loss=False)
    srp = generate_visualization(im_trans, class_index=5, return_loss=False)
    ntr = generate_visualization(im_trans, class_index=6, return_loss=False)

    axs[3].imshow(tru)
    axs[3].axis('off')
    axs[3].set_title('True')

axs[0].imshow(image)
axs[0].axis('off')
axs[1].imshow(im_trans_show)
axs[1].axis('off')
axs[2].imshow(prd)
axs[2].axis('off')
axs[2].set_title('Pred')

plt.show(block=True)
