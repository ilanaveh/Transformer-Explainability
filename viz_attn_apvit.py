"""
6/5/26
Based on DeiT_example.ipynb.
"""

from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import os
from samples.CLS2IDX import CLS2IDX, CLS2IDX_RAFDB
from baselines.ViT.ViT_explanation_generator import LRP
from baselines.ViT.ViT_LRP import deit_base_patch16_224 as vit_LRP
from PIL import ImageFilter  # for GaussianBlur
from config import viz_attn_config
from get_attn_map_for_viz import get_model_apvit
from info_dicts import img_lbl_dict, ims, ims_fail
import copy


cfg = viz_attn_config[5]

home_pth = '/home/projects/bagon/ilanaveh/code'

deit_cp_pth = os.path.join(home_pth, 'Transformers/deit/out/jobs_after_adding_seed')
deit_model_name = 'deit_blur0_BS128'

apvit_cp_pth = os.path.join(home_pth, 'Transformers/APViT/work_dirs')

ttls = {'RAF_blur8_pretrained0-8_freeze_fix_projs': 'Mixed',
        'RAF_blur8_freeze': 'Low'}

if cfg['choose_model'] == 'deit':
    im_size = 224
    im_nm = 'catdog.png'
    sf = 16
    # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

elif cfg['choose_model'] == 'apvit':
    im_size = 112
    im_nms = [f'{cfg["im_dir"]}/test_{im_id}.jpg' for im_id in ims_fail]
    sf = 8
    # normalize = transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    mean = np.array([123.675, 116.28, 103.53])/255
    std = np.array([58.395, 57.12, 57.375])/255
    # normalize = transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])

normalize = transforms.Normalize(mean=mean, std=std)
unnormalize = transforms.Normalize(mean=(-mean / std).tolist(), std=(1.0 / std).tolist())


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


transform_before_blur = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    normalize,
])

transforms_with_blur = {}

for tb in cfg['test_blurs']:
    if tb:
        transforms_with_blur[tb] = transforms.Compose([GaussianBlur(tb)] + transform_before_blur.transforms)
    else:
        transforms_with_blur[tb] = transform_before_blur


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
if cfg['choose_model'] == 'deit':
    model = vit_LRP(pretrained=True).cuda()
    deit_cp = torch.load(os.path.join(deit_cp_pth, deit_model_name, 'best_checkpoint.pth'))
    model.load_state_dict(deit_cp['model'])

elif cfg['choose_model'] == 'apvit':
    models = {}
    for mdl_nm in (cfg['model_names'] + list(cfg.get('models_for_hr_viz', {}).values())):
        models[mdl_nm] = get_model_apvit(home_pth, apvit_cp_pth, mdl_nm)


def generate_visualization(original_image, class_index=None, return_loss=None, add_alpha=False, mdl_nm=None):
    attribution_generator = LRP(model) if mdl_nm is None else LRP(models[mdl_nm])
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


cls2idx = CLS2IDX if (cfg['choose_model'] == 'deit') else CLS2IDX_RAFDB
idx2cls = {i: c for c, i in cls2idx.items()}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Option 1: plot "ims" dict: each emotion in separate figure:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
max_tb = np.max(cfg['test_blurs'])
if cfg['split_by_emo']:

    for emo in cfg['emos2plot']:
        emo_im_nms = ims[cfg['ims_type']][emo]

        fig, axs = plt.subplots(len(emo_im_nms), sum(len(items) for items in cfg['cols2show'].values()))
        fig.set_size_inches([fig.get_size_inches()[0]*2, 9.6])

        for i, im_nm in enumerate(emo_im_nms):
            im_trans = {}
            im_trans_show = {}
            output = {mdl: {} for mdl in (cfg['model_names']+ list(cfg.get('models_for_hr_viz', {}).values()))}
            for tb, trans in transforms_with_blur.items():
                # image = Image.open(f'{cfg["im_dir"]}/test_{im_nm}.jpg')
                image = Image.open(f'{cfg["im_dir"]}/{im_nm}')
                im_trans[tb] = trans(image)

                im_unnorm = unnormalize(im_trans[tb])
                im_trans_show[tb] = show_im_trans(im_unnorm)

                for mdl_nm in (cfg['model_names'] + list(cfg.get('models_for_hr_viz', {}).values())):
                    output[mdl_nm][tb] = models[mdl_nm](im_trans[tb].unsqueeze(0).cuda(), return_loss=False)
                    print_top_classes(output[mdl_nm][tb], 'raf')

            cols2show = copy.deepcopy(cfg['cols2show'])

            for ax_i in range(sum(len(items) for items in cols2show.values())):
                if 'img' in cols2show:
                    if 'original' in cols2show['img']:
                        axs[i, ax_i].imshow(image)
                        axs[i, ax_i].axis('off')
                        cols2show['img'].remove('original')
                        continue
                    elif 'trans' in cols2show['img']:
                        axs[i, ax_i].imshow(im_trans_show[max_tb])
                        axs[i, ax_i].axis('off')
                        cols2show['img'].remove('trans')
                        continue
                    else:
                        cols2show.pop('img')

                for mdl_nm in cfg['model_names']:
                    if mdl_nm in cols2show:
                        if 'pred_map_HR' in cols2show[mdl_nm]:
                            if 'models_for_hr_viz' in cfg and mdl_nm in cfg['models_for_hr_viz']:
                                mdl_nm_for_hr = cfg['models_for_hr_viz'][mdl_nm]
                            else:
                                mdl_nm_for_hr = mdl_nm
                            prd_cls = int(np.argmax(output[mdl_nm_for_hr][0].detach().cpu()))
                            pred = generate_visualization(im_trans[0], return_loss=False, mdl_nm=mdl_nm_for_hr)
                            axs[i, ax_i].imshow(pred)
                            axs[i, ax_i].axis('off')
                            if i == 0:
                                axs[i, ax_i].set_title(f'{ttls[mdl_nm]} Model\nPred Label (HR Img)')
                            axs[i, ax_i].text(-0.15, 0.5, cls2idx[prd_cls],
                                              transform=axs[i, ax_i].transAxes, rotation='vertical', va='center',
                                              ha='center')
                            cols2show[mdl_nm].remove('pred_map_HR')
                            break

                        elif 'pred_map' in cols2show[mdl_nm]:
                            prd_cls = int(np.argmax(output[mdl_nm][max_tb].detach().cpu()))
                            prd = generate_visualization(im_trans[max_tb], return_loss=False, mdl_nm=mdl_nm)
                            axs[i, ax_i].imshow(prd)
                            axs[i, ax_i].axis('off')
                            if i == 0:
                                axs[i, ax_i].set_title(f'{ttls[mdl_nm]} Model\nPred Label (LR Img)')
                            axs[i, ax_i].text(-0.15, 0.5, cls2idx[prd_cls],
                                              transform=axs[i, ax_i].transAxes, rotation='vertical', va='center', ha='center')
                            cols2show[mdl_nm].remove('pred_map')
                            break

                        elif 'true_map' in cols2show[mdl_nm]:
                            if 'use_img_lbl_dict' in cfg and cfg['use_img_lbl_dict']:
                                tru_cls = img_lbl_dict[im_nm]
                            else:
                                tru_cls = idx2cls[emo]
                            tru = generate_visualization(im_trans[max_tb], class_index=tru_cls, return_loss=False,
                                                         mdl_nm=mdl_nm)
                            axs[i, ax_i].imshow(tru)
                            axs[i, ax_i].axis('off')
                            if i == 0:
                                axs[i, ax_i].set_title(f'{ttls[mdl_nm]} Model\nTrue Label (LR Img)')
                            axs[i, ax_i].text(-0.15, 0.5, cls2idx[tru_cls],
                                              transform=axs[i, ax_i].transAxes, rotation='vertical', va='center',
                                              ha='center')
                            cols2show[mdl_nm].remove('true_map')
                            break

                        else:
                            cols2show.pop(mdl_nm)

        fig.suptitle(f'Class: {emo}\n{cfg["ims_type"]}')

        plt.tight_layout()
        if cfg['save_figs']:
            fig.savefig(
                f"./figures/from_viz_attn_apvit/{emo}_Test{max_tb}_{cfg['ims_type']}.png")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Option 2: plot "ims_fail":
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
else:
    fig, axs = plt.subplots(len(im_nms), 4)
    fig.set_size_inches([8.8, 9.6])
    for i, im_nm in enumerate(im_nms):
        image = Image.open(f'samples/{im_nm}')
        im_trans = transform(image)

        im_unnorm = unnormalize(im_trans)
        im_trans_show = show_im_trans(im_unnorm)

        if cfg['choose_model'] == 'deit':
            output = model(im_trans.unsqueeze(0).cuda())
            print_top_classes(output)
        elif cfg['choose_model'] == 'apvit':
            output = model(im_trans.unsqueeze(0).cuda(), return_loss=False)
            print_top_classes(output, 'raf')

        prd_cls = int(np.argmax(output.detach().cpu()))
        # dog
        # generate visualization for class 243: 'bull mastiff' - the predicted class
        if cfg['choose_model'] == 'deit':
            dog = generate_visualization(im_trans)
            cat = generate_visualization(im_trans, class_index=282)
            axs[i, 3].imshow(cat)
            axs[i, 3].axis('off')
        elif cfg['choose_model'] == 'apvit':
            tru_cls = img_lbl_dict[im_nm.replace('test_images_apvit/', '')]
            prd = generate_visualization(im_trans, return_loss=False)
            tru = generate_visualization(im_trans, class_index=tru_cls, return_loss=False)
            ang = generate_visualization(im_trans, class_index=0, return_loss=False)
            dsg = generate_visualization(im_trans, class_index=1, return_loss=False)
            frt = generate_visualization(im_trans, class_index=2, return_loss=False)
            sad = generate_visualization(im_trans, class_index=3, return_loss=False)
            hpy = generate_visualization(im_trans, class_index=4, return_loss=False)
            srp = generate_visualization(im_trans, class_index=5, return_loss=False)
            ntr = generate_visualization(im_trans, class_index=6, return_loss=False)

            axs[i, 3].imshow(tru)
            axs[i, 3].axis('off')
            axs[i, 3].text(-0.15, 0.5, idx2cls[tru_cls],
                           transform=axs[i, 3].transAxes, rotation='vertical', va='center', ha='center')
            if i == 0:
                axs[i, 3].set_title('True')

        axs[i, 0].imshow(image)
        axs[i, 0].axis('off')
        axs[i, 1].imshow(im_trans_show)
        axs[i, 1].axis('off')
        axs[i, 2].imshow(prd)
        axs[i, 2].axis('off')
        axs[i, 2].text(-0.15, 0.5, idx2cls[prd_cls],
                       transform=axs[i, 2].transAxes, rotation='vertical', va='center', ha='center')
        if i == 0:
            axs[i, 2].set_title('Pred')

    plt.tight_layout()
    plt.show(block=True)
    if cfg['save_figs']:
        fig.savefig(f"./figures/from_viz_attn_apvit/{apvit_model_name}_test{cfg['test_blur']}_{len(im_nms)}ims.png")

plt.show(block=True)
