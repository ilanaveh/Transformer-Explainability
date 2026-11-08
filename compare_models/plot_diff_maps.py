"""
19/7/26
Assumes file "difference_maps/RAF_blur8_pretrained0-8_freeze_fix_projs_minus_RAF_blur8_freeze_pred_label/
    metadata_difference_maps_pred_label.csv" exists
    (created by create_diff_attn_maps.py).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from plot_prototypes_confusion import compute_average_map_and_face, make_alpha_map

TARGET_LABEL = 'pred_label'

# AGGREGATE_BY = ['model_A_wrong', 'model_B_correct']  # choose from: ['all', 'true_label', 'model_A_{correct/wrong}', 'model_B_{correct/wrong}']
AGGREGATE_BY = {
    'All': ['all'],
    'Mixed Wins': ['model_A_wrong', 'model_B_correct'],
    'Uniform Wins': ['model_A_correct', 'model_B_wrong'],
    'Both Correct': ['model_A_correct', 'model_B_correct']
}

CENTER_FUNC = 'Median'  # 'Mean' / 'Median'

MODELS = [
    "RAF_blur8_freeze",
    "RAF_blur8_pretrained0-8_freeze_fix_projs",
]

MODEL_LABELS = {"RAF_blur8_freeze": 'Uniform', "RAF_blur8_pretrained0-8_freeze_fix_projs": 'Mixed'}

DIFF_PATH = Path(f"difference_maps/{MODELS[1]}_minus_{MODELS[0]}_{TARGET_LABEL}")
METADATA_PATH = DIFF_PATH / Path('metadata_difference_maps_pred_label.csv')
MAP_PATH = DIFF_PATH / Path('maps')
OUTPUT_DIR = Path("figures")

LABELS = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happy",
    "Sad",
    "Angry",
    "Neutral",
]

INTERPOLATION = 'nearest'  # 'bilinear' for smoothing, 'nearest' for keeping patch-resolution (relevant for native prototypes)
DPI = 300
CMAP = "jet"
USE_SHARED_COLOR_SCALE = True
SPLIT_MODELS = False


def load_face_image(
    path: str | Path,
    output_size: tuple[int, int] = (224, 224),
    grayscale: bool = False,
) -> np.ndarray:
    """
    Load and resize a face image.

    Returns
    -------
    image:
        If grayscale=False: [H, W, 3], float32, range [0, 1]
        If grayscale=True:  [H, W], float32, range [0, 1]
    """
    path = Path(path)

    if grayscale:
        img = Image.open(path).convert("L")
    else:
        img = Image.open(path).convert("RGB")

    out_h, out_w = output_size
    img = img.resize((out_w, out_h), Image.BILINEAR)

    arr = np.asarray(img).astype(np.float32) / 255.0

    return arr


def get_subset_df(metadata_df, aggregate_by):
    if 'all' in aggregate_by:
        return metadata_df

    if 'model_A_correct' in aggregate_by:
        metadata_df = metadata_df.loc[metadata_df["correct_A"]].copy()

    if 'model_B_correct' in aggregate_by:
        metadata_df = metadata_df.loc[metadata_df["correct_B"]].copy()

    if 'model_A_wrong' in aggregate_by:
        metadata_df = metadata_df.loc[~metadata_df["correct_A"]].copy()

    if 'model_B_wrong' in aggregate_by:
        metadata_df = metadata_df.loc[~metadata_df["correct_B"]].copy()

    return metadata_df


def create_diff_figure(metadata_df, aggregate_by, split_by_model=False, use_median=False):

    fig, axes = plt.subplots(2 if split_by_model else 1, len(aggregate_by), figsize=(12.5, 4.5 if split_by_model else 3))

    mean_maps = {}
    mean_maps_split = {agg: {} for agg in aggregate_by.keys()}
    mean_faces = {}
    n_used = {}

    for agg_lbl, agg_list in aggregate_by.items():
        subset_df = get_subset_df(metadata_df, agg_list)

        mean_maps[agg_lbl], mean_faces[agg_lbl], n_used[agg_lbl] = compute_average_map_and_face(
            subset_df["diff_map_path"], subset_df["image_id"], use_median=use_median
        )

        if split_by_model:
            mean_maps_split[agg_lbl][MODELS[0]] = mean_maps[agg_lbl].copy()
            mean_maps_split[agg_lbl][MODELS[0]][mean_maps[agg_lbl] >= 0] = 0
            mean_maps_split[agg_lbl][MODELS[0]] *= -1
            mean_maps_split[agg_lbl][MODELS[1]] = mean_maps[agg_lbl].copy()
            mean_maps_split[agg_lbl][MODELS[1]][mean_maps[agg_lbl] <= 0] = 0

    if USE_SHARED_COLOR_SCALE:
        if split_by_model:
            vmin = 0
            vmax = max(
                np.nanmax(mean_maps_split[agg][mdl])
                for agg in aggregate_by.keys() for mdl in MODELS
            )

            for m, mdl in enumerate(MODELS):

                for i, agg_lbl in enumerate(aggregate_by.keys()):
                    ax = axes[m, i]

                    mean_map = mean_maps_split[agg_lbl][mdl]
                    mean_face = mean_faces[agg_lbl]

                    ax.imshow(np.clip(mean_face, 0, 1))

                    if mean_map.shape != mean_face.shape[:2]:
                        display_proto = torch.nn.functional.interpolate(
                            torch.from_numpy(mean_map).reshape(1, 1, 14, 14),
                            scale_factor=mean_face.shape[0] / mean_map.shape[0],
                            mode=INTERPOLATION).reshape(mean_face.shape[:2]).numpy()

                    alpha_map = make_alpha_map(
                        display_proto,
                        min_alpha=.5,
                        max_alpha=1,
                        gamma=1.5,
                    )
                    last_image = ax.imshow(
                        display_proto,
                        cmap=CMAP,
                        vmin=vmin,
                        vmax=vmax,
                        alpha=alpha_map
                    )
                    if m == 0:
                        ax.set_title(f'{agg_lbl} (N={n_used[agg_lbl]})')
                        if i == 0:
                            ax.set_ylabel(f'{MODEL_LABELS[MODELS[0]]} > {MODEL_LABELS[MODELS[1]]}')
                    elif i == 0:
                        ax.set_ylabel(f'{MODEL_LABELS[MODELS[1]]} > {MODEL_LABELS[MODELS[0]]}')

            colorbar_ax = fig.add_axes([.93, .15, 0.015, 0.7])
            colorbar = fig.colorbar(last_image, cax=colorbar_ax)
            colorbar.set_label("Difference", fontsize=11)
            fig.suptitle(f'{CENTER_FUNC} Differences')
            return fig

        else:
            vmin = min(
                np.nanmin(mean_maps[agg])
                for agg in aggregate_by.keys()
            )
            vmax = max(
                np.nanmax(mean_maps[agg])
                for agg in aggregate_by.keys()
            )
    else:
        vmin = None
        vmax = None

    for i, agg_lbl in enumerate(aggregate_by.keys()):
        ax = axes[i]

        mean_map = mean_maps[agg_lbl]
        mean_face = mean_faces[agg_lbl]

        ax.imshow(np.clip(mean_face, 0, 1))

        if mean_map.shape != mean_face.shape[:2]:
            display_proto = torch.nn.functional.interpolate(
                torch.from_numpy(mean_map).reshape(1, 1, 14, 14),
                scale_factor=mean_face.shape[0] / mean_map.shape[0],
                mode=INTERPOLATION).reshape(mean_face.shape[:2]).numpy()

        alpha_map = make_alpha_map(
            display_proto,
            min_alpha=.3,
            max_alpha=1,
            gamma=1.5,
        )
        alpha_map_mask = ((display_proto < .003) & (display_proto > -.003))
        alpha_map[alpha_map_mask] = 0

        last_image = ax.imshow(
            display_proto,
            cmap=CMAP,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha_map
        )
        ax.set_title(f'{agg_lbl} (N={n_used[agg_lbl]})')

    colorbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
    colorbar = fig.colorbar(last_image, cax=colorbar_ax)
    colorbar.set_label("Difference", fontsize=11)

    fig.suptitle(f'{CENTER_FUNC} Differences')
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_csv(METADATA_PATH)

    fig = create_diff_figure(metadata_df=metadata_df, aggregate_by=AGGREGATE_BY, split_by_model=SPLIT_MODELS,
                             use_median=CENTER_FUNC == 'Median')

    png_path = (
        OUTPUT_DIR
        / f"difference_maps_{TARGET_LABEL}_{CENTER_FUNC}_{'split_models' if SPLIT_MODELS else 'joint'}.png"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()