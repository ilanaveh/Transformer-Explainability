"""
17/5/26

"""

# ============================================================
# Attention-map saving + prototype analysis for ViT FER models
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import sys
import torch.nn.functional as F
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from baselines.ViT.ViT_explanation_generator import LRP
from samples.CLS2IDX import CLS2IDX_RAFDB
import torch.nn as nn
import json
import copy
from baselines.ViT.ViT_LRP import deit_base_patch16_224_apvit as apvit_LRP

apvit_pth = "/home/projects/bagon/ilanaveh/code/Transformers/APViT"
sys.path.insert(0, apvit_pth)
from mmcls.datasets.raf import RAF
from mmcls.models.classifiers.pool_vit import PoolingVitClassifier
from mmcls.models.vit.vit_siam_merge import PoolingViT
from mmcv.parallel import MMDataParallel

# ============================================================
# 0. Configuration
# ============================================================

CLS2IDX = {i: label for label, i in CLS2IDX_RAFDB.items()}
IDX2CLS = {label: i for label, i in CLS2IDX_RAFDB.items()}
EMOTIONS = list(CLS2IDX_RAFDB.values())

MAP_RES = 'native'  # native / 'input_size'
POOLING = True
# models2save = ['mixed_low_test', 'lowres', 'mixed_hi_test', 'hires']  # more options: 'mixed_test16', 'low_test16'.
models2save = ['mixed_low_test', 'lowres', 'mixed_hi_test', 'hires']
attn_map_pth = 'attention_maps' if MAP_RES == 'input_size' else f'attention_maps/{MAP_RES.upper()}_RESOLUTION'
attn_map_pth = attn_map_pth + '/WITH_POOLING' if POOLING else attn_map_pth

# ============================================================
# 1. Generic normalization utilities
# ============================================================
def normalize_relevance_map(
    relevance: np.ndarray,
    eps: float = 1e-8,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Normalize a relevance map so that all relevance values sum to 1.

    This is the key normalization for comparing allocation strategies.

    Parameters
    ----------
    relevance:
        2D relevance map, shape [H, W].

    eps:
        Numerical stability constant.

    clip_negative:
        If True, negative relevance values are clipped to zero before normalization.
        This is usually appropriate when analyzing attention/relevance allocation.

    Returns
    -------
    normalized:
        2D relevance map summing to 1.
    """
    relevance = relevance.astype(np.float32)

    if clip_negative:
        relevance = np.maximum(relevance, 0)

    total = relevance.sum()

    if total < eps:
        return np.zeros_like(relevance, dtype=np.float32)

    return relevance / total


def ensure_2d_map(relevance: np.ndarray) -> np.ndarray:
    """
    Convert [1, H, W] or [H, W] to [H, W].
    """
    if relevance.ndim == 3:
        if relevance.shape[0] == 1:
            relevance = relevance[0]
        else:
            raise ValueError(f"Unexpected 3D relevance shape: {relevance.shape}")

    if relevance.ndim != 2:
        raise ValueError(f"Expected 2D relevance map, got shape: {relevance.shape}")

    return relevance.astype(np.float32)


def resize_map_numpy(
    relevance: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """
    Resize a 2D relevance map to a common resolution.

    Uses torch interpolation to avoid requiring OpenCV.
    """
    if output_size == relevance.shape:
        return relevance

    relevance = ensure_2d_map(relevance)

    tensor = torch.from_numpy(relevance)[None, None, :, :].float()

    resized = F.interpolate(
        tensor,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )

    return resized[0, 0].cpu().numpy().astype(np.float32)


# ============================================================
# 3. Placeholder for your Chefer relevance method
# ============================================================

def compute_relevance_map(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class_idx: int,
    sf: int = 8,
    map_res: str = 'input_size',
) -> np.ndarray:
    """
    Compute class-specific relevance map using your implementation of:
    'Transformer Interpretability Beyond Attention Visualization' / Chefer et al.

    Parameters
    ----------
    model:
        ViT model.

    image_tensor:
        Single image tensor with shape [1, C, H, W].

    target_class_idx:
        Integer class index for which to compute the relevance map.

    sf:
        Integer scale-factor for interpolation (used only for map_res = 'input_size').

    map_res:
        String depicting resolution of attention-map to be returned:
            'input_size' (after interpolation = 112*112) or 'native' (without interpolation = 14*14).

    Returns
    -------
    relevance:
        2D numpy array, shape [H_map, W_map].

    Notes
    -----
    Replace the body of this function with your own method.
    For example, if your implementation has something like:

        relevance = attribution_generator.generate_LRP(
            image_tensor,
            method="transformer_attribution",
            index=target_class_idx,
        )

    then return that relevance map here.
    """
    im_size = image_tensor.shape[-1]
    attribution_generator = LRP(model.module)
    transformer_attribution = attribution_generator.generate_LRP(image_tensor.cuda(),
                                                                 method="transformer_attribution",
                                                                 index=target_class_idx,
                                                                 return_loss=False).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)

    if map_res == 'input_size':
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=sf, mode='bilinear')
        transformer_attribution = transformer_attribution.reshape(im_size, im_size).cuda().data.cpu().numpy()
    elif map_res == 'native':
        transformer_attribution = transformer_attribution.squeeze().cuda().data.cpu().numpy()

    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
            transformer_attribution.max() - transformer_attribution.min())

    return transformer_attribution


# ============================================================
# 3. Saving attention/relevance maps
# ============================================================

def predict_label(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
) -> tuple[int, np.ndarray]:
    """
    Run model prediction.

    Returns
    -------
    pred_idx:
        Predicted class index.

    probs:
        Softmax probabilities, shape [num_classes].
    """
    model.eval()

    logits = model(image_tensor, return_loss=False)
    probs = torch.softmax(logits, dim=-1)

    pred_idx = int(probs.argmax(dim=-1).item())
    probs_np = probs[0].detach().cpu().numpy()

    return pred_idx, probs_np


def save_attention_maps_for_dataset(
    model: torch.nn.Module,
    model_name: str,
    test_blur: int,
    dataset,
    output_dir: str | Path,
    device: str | torch.device = "cuda",
    target_modes: tuple[str, ...] = ("true_label", "pred_label"),
    save_all_classes: bool = False,
    normalize_before_saving: bool = True,
    clip_negative: bool = True,
    map_output_size: Optional[tuple[int, int]] = None,
    map_res: str = 'input_size',
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Generate and save class-specific attention/relevance maps for a dataset.

    Parameters
    ----------
    model:
        Your trained ViT model.

    model_name:
        Name used in filenames and metadata.
        Example: "mixed_pretrain" or "lowres_pretrain".

    dataset:
        Iterable dataset. Each item should return a dict with at least:

            {
                "image": torch.Tensor [C, H, W],
                "label": int,
                "image_id": str,
                "sigma": int or float,
            }

        If your dataset returns tuples, adapt the few lines marked below.

    output_dir:
        Directory where maps and metadata will be saved.
        If map_res is 'native', add 'NATIVE_RESOLUTION' to path.

    device:
        "cuda" or "cpu".

    target_modes:
        Which target-class maps to save.

        Options:
            "true_label" = map for the ground-truth class
            "pred_label" = map for the predicted class

    save_all_classes:
        If True, saves one relevance map for every emotion class.
        This is useful but expensive.

    normalize_before_saving:
        If True, each saved relevance map is sum-normalized.
        Recommended for prototype analysis.

    clip_negative:
        If True, clip negative relevance values before sum-normalization.

    map_output_size:
        If provided, resize relevance maps before saving.
        Example: (224, 224).

    map_res:
        If 'input_size', do interpolation (to 112*112)
        If 'native', return map in native resolution (14*14). Also, add 'NATIVE_RESOLUTION' to output_dir.

    overwrite:
        If False, skip maps that already exist.

    Returns
    -------
    metadata_df:
        DataFrame with one row per saved map.
    """
    # if map_res == 'native':
    #     output_dir = output_dir.replace('attention_maps', 'attention_maps/NATIVE_RESOLUTION')
    output_dir = Path(output_dir)
    metadata_path = output_dir / f"metadata__{model_name}.csv"

    if metadata_path.exists() and not overwrite:
        metadata_df = pd.read_csv(metadata_path)
        return metadata_df

    map_dir = output_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.eval()

    rows = []

    for item_idx in tqdm(range(len(dataset)), desc=f"Saving maps: {model_name}"):

        # ----------------------------------------------------
        # Adapt this block if your dataset returns tuples
        # ----------------------------------------------------
        item_info = dataset.data_infos[item_idx]
        image = dataset[item_idx]["img"]              # torch.Tensor [C, H, W]
        true_idx = int(item_info['gt_label'])      # int
        image_id = str(item_info['img_info']['filename']).replace('.jpg', '')   # str
        # ----------------------------------------------------

        image_tensor = image.unsqueeze(0).to(device)

        pred_idx, probs = predict_label(model, image_tensor)

        true_label = IDX2CLS[true_idx]
        pred_label = IDX2CLS[pred_idx]
        correct = pred_idx == true_idx

        targets_to_save = {}

        if "true_label" in target_modes:
            targets_to_save[true_label] = true_idx

        if "pred_label" in target_modes:
            targets_to_save[pred_label] = pred_idx

        if save_all_classes:
            for label, idx in CLS2IDX.items():
                targets_to_save[label] = idx

        for target_label, target_idx in targets_to_save.items():

            sigma_str = "none" if test_blur is None else str(test_blur)

            filename = (
                f"image-{image_id}"
                f"__model-{model_name}"
                f"__sigma-{sigma_str}"
                f"__target-{target_label}.npy"
            )

            map_path = map_dir / filename

            if map_path.exists() and not overwrite:
                rows.append({
                    "image_id": image_id,
                    "model": model_name,
                    "sigma": test_blur,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "target_label": target_label,
                    "target_class_idx": target_idx,
                    "correct": correct,
                    "pred_confidence": float(probs[pred_idx]),
                    "true_confidence": float(probs[true_idx]),
                    "map_path": str(map_path),
                    "normalized_before_saving": normalize_before_saving,
                })
                continue

            # This is the method you plug in.
            with torch.enable_grad():
                relevance = compute_relevance_map(model=model, image_tensor=image_tensor, target_class_idx=target_idx,
                                                  map_res=map_res)

            relevance = ensure_2d_map(relevance)

            if map_output_size is not None:
                relevance = resize_map_numpy(relevance, map_output_size)

            if normalize_before_saving:
                relevance = normalize_relevance_map(
                    relevance,
                    clip_negative=clip_negative,
                )

            np.save(map_path, relevance.astype(np.float32))

            rows.append({
                "image_id": image_id,
                "model": model_name,
                "sigma": test_blur,
                "true_label": true_label,
                "pred_label": pred_label,
                "target_label": target_label,
                "target_class_idx": target_idx,
                "correct": correct,
                "pred_confidence": float(probs[pred_idx]),
                "true_confidence": float(probs[true_idx]),
                "map_path": str(map_path),
                "normalized_before_saving": normalize_before_saving,
            })

    metadata_df = pd.DataFrame(rows)

    metadata_df.to_csv(metadata_path, index=False)

    return metadata_df


def merge_metadata_files(
    metadata_paths: list[str | Path],
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Merge metadata CSV files from multiple models.
    """
    dfs = [pd.read_csv(path) for path in metadata_paths]
    df = pd.concat(dfs, ignore_index=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


# ============================================================

# ============================================================
home_pth = '/home/projects/bagon/ilanaveh'
data_prefix = os.path.join(home_pth, 'data/RAF-DB/basic/Image/aligned_224')
pipeline_8 = [{'type': 'LoadImageFromFile'},
              {'type': 'GaussianBlur', 'sigma_min': 8, 'sigma_max': 8},
              {'type': 'Resize', 'size': 112},
              {'type': 'Normalize',
               'mean': [123.675, 116.28, 103.53],
               'std': [58.395, 57.12, 57.375],
               'to_rgb': True},
              {'type': 'ImageToTensor', 'keys': ['img']},
              {'type': 'Collect', 'keys': ['img']}]

pipeline_0 = copy.deepcopy(pipeline_8)
pipeline_0.remove({'type': 'GaussianBlur', 'sigma_min': 8, 'sigma_max': 8})
pipeline_16 = copy.deepcopy(pipeline_8)
pipeline_16[1] = {'type': 'GaussianBlur', 'sigma_min': 16, 'sigma_max': 16}
pipelines = {0: pipeline_0, 8: pipeline_8, 16: pipeline_16}
ann_file = '/home/projects/bagon/ilanaveh/data/RAF-DB/basic/EmoLabel/test.txt'


class IdentityHeadWithSimpleTest(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

    def simple_test(self, x, *args, **kwargs):
        return x


def create_model(mdl_pth, mdl_args):
    model = PoolingVitClassifier(**mdl_args)
    model.head = IdentityHeadWithSimpleTest()
    model.vit = apvit_LRP(pretrained=False, use_pooling=POOLING, num_classes=7)
    # model.vit =PoolingViT(input_type='feature', patch_num=196,
    #         embed_dim=768, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True, norm_layer_eps=1e-06,
    #         in_channels=[256],
    #         attn_method='SUM_ABS_1',    # CNN Attention method， SUM_ABS_1
    #         cnn_pool_config=dict(keep_num=160, exclude_first=False),
    #         vit_pool_configs=dict(keep_rates=[1.] * 4 + [0.9] * 8, exclude_first=True, attn_method='SUM'))
    model = model.cuda()
    apvit_cp = torch.load(os.path.join(mdl_pth, 'epoch_40.pth'))['state_dict']
    # apvit_cp["vit.pos_embed"] = torch.cat(
    #     [apvit_cp.pop("vit.cls_pos_embed"), apvit_cp.pop("vit.patch_pos_embed")],
    #     dim=1,
    # )
    apvit_cp["vit.head.weight"] = apvit_cp.pop("head.fc.weight")
    apvit_cp["vit.head.bias"] = apvit_cp.pop("head.fc.bias")
    missing, unexpected = model.load_state_dict(apvit_cp, strict=False)
    assert ((missing == ['vit.pos_embed', 'vit.patch_embed.proj.weight', 'vit.patch_embed.proj.bias']) and not unexpected)
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    return model


if __name__ == '__main__':

    with open(os.path.join(apvit_pth, 'args_for_build_mdl.json'), 'r') as f:
        args = json.load(f)
    args['head']['topk'] = (1,)  #
    model_names = {
        'mixed_low_test': 'RAF_blur8_pretrained0-8_freeze_fix_projs',
        'lowres': 'RAF_blur8_freeze',
        'mixed_hi_test': 'RAF_blur0_pretrained0-8_freeze',
        'low_test16': 'RAF_blur16_pretrained8_freeze',
        'mixed_test16': 'RAF_blur16_pretrained0-8_freeze',
        'hires': 'RAF_blur0_freeze',
    }

    test_blurs = {
        'mixed_low_test': 8,
        'lowres': 8,
        'mixed_hi_test': 0,
        'low_test16': 16,
        'mixed_test16': 16,
        'hires': 0,
    }
    models = {mdl_nm: create_model(os.path.join(apvit_pth, 'work_dirs', model_names[mdl_nm]), args) for mdl_nm in models2save}
    test_datasets = {bl: RAF(data_prefix, pipelines[bl], ann_file=ann_file, test_mode=True) for bl in [0, 8, 16]}

    # Each dataset item should look like:
    #
    #     {
    #         "image": image_tensor,    # [C, H, W]
    #         "label": label_idx,       # int from 0 to 6
    #         "image_id": image_id,     # unique string
    #         "sigma": sigma_value,     # e.g. 0, 2, 4, 6, 8
    #     }

    # ------------------------------------------------------------
    # Step 1: Save attention maps for each model
    # ------------------------------------------------------------
    metadata = {}
    for mdl in models2save:
        metadata[mdl] = save_attention_maps_for_dataset(
            model=models[mdl],
            model_name=model_names[mdl],
            test_blur=test_blurs[mdl],
            dataset=test_datasets[test_blurs[mdl]],
            output_dir=f"{attn_map_pth}/{mdl}",
            device="cuda",
            target_modes=("true_label", "pred_label"),
            save_all_classes=False,
            normalize_before_saving=True,
            clip_negative=True,
            map_res=MAP_RES,
            overwrite=False,
        )

    df = merge_metadata_files(
        metadata_paths=[f"{attn_map_pth}/{mdl}/metadata__{model_names[mdl]}.csv" for mdl in models2save],
        output_path=f"{attn_map_pth}/metadata_all_models_{MAP_RES}.csv",
    )

