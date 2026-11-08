"""
19/7/26
Compute maps of the difference between pairs of attentio maps.
Current usage: per-image difference between attn-map of uniform vs. mixed model.
Average separately for both-models-correct; mixed-win; uniform-wins.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# =============================================================================
# Configuration
# =============================================================================

METADATA_PATH = Path("attention_maps/NATIVE_RESOLUTION/metadata_all_models_native.csv")

MODELS = [
    "RAF_blur8_freeze",
    "RAF_blur8_pretrained0-8_freeze_fix_projs",
]

LABELS = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happy",
    "Sad",
    "Angry",
    "Neutral",
]

TARGET_LABEL = 'pred_label'  # 'pred_label' / 'true_label'

OUTPUT_DIR = Path("difference_maps")

DPI = 300
CMAP = "jet"

# When True, each individual map is independently min-max normalized before
# averaging. Usually leave False if the saved maps are already normalized.
NORMALIZE_EACH_MAP = False


# =============================================================================
# Loading and preprocessing
# =============================================================================
def load_attention_map(map_path):
    """
    Load a saved .npy attention/relevance map and return a 2D float array.
    """
    map_path = Path(map_path).expanduser()

    # Resolve relative map paths relative to the metadata file.
    # if not map_path.is_absolute():
    #     map_path = METADATA_PATH.parent / map_path

    if not map_path.exists():
        raise FileNotFoundError(f"Map file not found: {map_path}")

    if map_path.suffix.lower() != ".npy":
        raise ValueError(f"Expected a .npy file, got: {map_path}")

    attention_map = np.load(map_path)
    attention_map = np.asarray(attention_map, dtype=float)
    attention_map = np.squeeze(attention_map)

    if attention_map.ndim != 2:
        raise ValueError(
            f"Expected a 2D map, but {map_path} has shape "
            f"{attention_map.shape}"
        )

    return attention_map


def normalize_map(attention_map, eps=1e-12):
    """
    Min-max normalize one map to [0, 1].
    """
    minimum = np.nanmin(attention_map)
    maximum = np.nanmax(attention_map)

    if maximum - minimum < eps:
        return np.zeros_like(attention_map, dtype=float)

    return (attention_map - minimum) / (maximum - minimum)


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
    img = img.resize((out_w, out_h))

    arr = np.asarray(img).astype(np.float32) / 255.0

    return arr


def compute_difference_maps(models_df, output_dir, overwrite=False):
    """
    Load all maps of both models and compute image-wise differences.

    Returns
    -------
    mean_map : np.ndarray or None
        Mean map across successfully loaded images.
    n_used : int
        Number of maps included in the mean.
    """
    output_dir = Path(output_dir)
    metadata_path = output_dir / f"metadata_difference_maps_{TARGET_LABEL}.csv"

    if metadata_path.exists() and not overwrite:
        metadata_df = pd.read_csv(metadata_path)
        return metadata_df

    map_dir = output_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    expected_shape = None
    for face_id in models_df['image_id'].unique():
        filename = f'{face_id}.npy'
        diff_map_path = map_dir / filename

        cur_face_df = models_df.loc[models_df['image_id'] == face_id]

        assert(len(cur_face_df) == 2)

        map_paths = {mdl: cur_face_df['map_path'].loc[models_df['model'] == mdl].item() for mdl in MODELS}
        attn_maps = {mdl: load_attention_map(map_paths[mdl]) for mdl in MODELS}

        if NORMALIZE_EACH_MAP:
            attn_maps = {mdl: normalize_map(attn_maps[mdl]) for mdl in MODELS}

        if expected_shape is None:
            expected_shape = attn_maps[MODELS[0]].shape

        for m in [0, 1]:
            if attn_maps[MODELS[m]].shape != expected_shape:
                raise ValueError(
                    f"Map shape {attn_maps[MODELS[m]].shape} of model {MODELS[m]} does not match "
                    f"expected shape {expected_shape}"
                )

        diff_map = attn_maps[MODELS[1]] - attn_maps[MODELS[0]]

        np.save(diff_map_path, diff_map.astype(np.float32))

        rows.append({
            "image_id": face_id,
            "model_A": MODELS[0],
            "model_B": MODELS[1],
            "sigma": cur_face_df["sigma"].iloc[0],
            "true_label": cur_face_df["true_label"].iloc[0],
            "pred_label_A": cur_face_df["pred_label"].loc[cur_face_df["model"] == MODELS[0]].item(),
            "pred_label_B": cur_face_df["pred_label"].loc[cur_face_df["model"] == MODELS[1]].item(),
            "target_label": TARGET_LABEL,
            "correct_A": cur_face_df["correct"].loc[cur_face_df["model"] == MODELS[0]].item(),
            "correct_B": cur_face_df["correct"].loc[cur_face_df["model"] == MODELS[1]].item(),
            "pred_confidence_A": cur_face_df["pred_confidence"].loc[cur_face_df["model"] == MODELS[0]].item(),
            "pred_confidence_B": cur_face_df["pred_confidence"].loc[cur_face_df["model"] == MODELS[1]].item(),
            "true_confidence_A": cur_face_df["true_confidence"].loc[cur_face_df["model"] == MODELS[0]].item(),
            "true_confidence_B": cur_face_df["true_confidence"].loc[cur_face_df["model"] == MODELS[1]].item(),
            "ori_map_path_A": cur_face_df["map_path"].loc[cur_face_df["model"] == MODELS[0]].item(),
            "ori_map_path_B": cur_face_df["map_path"].loc[cur_face_df["model"] == MODELS[1]].item(),
            "diff_map_path": diff_map_path,
            "positive_values_mean_higher_attention_in_model": MODELS[1],
        })

    metadata_df = pd.DataFrame(rows)

    metadata_df.to_csv(metadata_path, index=False)

    return metadata_df


def main():
    metadata = pd.read_csv(METADATA_PATH)
    models_df = metadata.loc[
        ((metadata["model"] == MODELS[0]) | (metadata["model"] == MODELS[1]))
        &
        (metadata["target_label"] == metadata[TARGET_LABEL])
        ].copy()

    compute_difference_maps(models_df, output_dir=f"{OUTPUT_DIR}/{MODELS[1]}_minus_{MODELS[0]}_{TARGET_LABEL}")


if __name__ == '__main__':
    main()

