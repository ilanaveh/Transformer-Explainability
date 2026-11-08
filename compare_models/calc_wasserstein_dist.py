"""
25/5/26
Assumes file "attention_maps/metadata_all_models.csv" contains df with maps for all models in 'models2comp' below.
(dfs were created by create_attn_maps.py).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ============================================================
# 1. Map utilities
# ============================================================

def load_attention_map(path: str | Path) -> np.ndarray:
    """
    Load a saved attention/relevance map from .npy.

    Expected shape:
        [H, W] or [1, H, W]
    """
    arr = np.load(path)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            raise ValueError(f"Unexpected 3D map shape: {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D map, got shape: {arr.shape}")

    return arr.astype(np.float64)


def normalize_map(
    x: np.ndarray,
    eps: float = 1e-12,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Convert attention/relevance map into a probability distribution.

    Required before Wasserstein comparison.
    """
    x = np.asarray(x, dtype=np.float64)

    if clip_negative:
        x = np.maximum(x, 0)

    total = x.sum()

    if total < eps:
        return np.zeros_like(x, dtype=np.float64)

    return x / total


def resize_map(
    x: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """
    Resize a 2D map using torch bilinear interpolation.
    """
    x = np.asarray(x, dtype=np.float64)

    tensor = torch.tensor(x, dtype=torch.float32)[None, None, :, :]

    resized = F.interpolate(
        tensor,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )

    return resized[0, 0].cpu().numpy().astype(np.float64)


def preprocess_map_for_wasserstein(
    path: str | Path,
    output_size: tuple[int, int] = (32, 32),
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Load, resize, and normalize an attention map.

    For Wasserstein, downsampling is strongly recommended because exact
    optimal transport becomes expensive at high resolution.

    Example:
        32 x 32 = 1024 spatial bins
        64 x 64 = 4096 spatial bins, much slower
    """
    x = load_attention_map(path)
    x = resize_map(x, output_size)
    x = normalize_map(x, clip_negative=clip_negative)
    return x


# ============================================================
# 2. Wasserstein / EMD distance
# ============================================================

def make_2d_cost_matrix(
    height: int,
    width: int,
    normalize_cost: bool = True,
) -> np.ndarray:
    """
    Create pairwise Euclidean distance matrix between all pixels/patches.

    For a 32x32 map:
        cost matrix shape = [1024, 1024]
    """
    ys, xs = np.meshgrid(
        np.arange(height),
        np.arange(width),
        indexing="ij",
    )

    coords = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float64)

    diff = coords[:, None, :] - coords[None, :, :]
    cost = np.sqrt((diff ** 2).sum(axis=-1))

    if normalize_cost:
        max_cost = cost.max()
        if max_cost > 0:
            cost = cost / max_cost

    return cost


def wasserstein_distance_2d(
    map_a: np.ndarray,
    map_b: np.ndarray,
    cost_matrix: np.ndarray | None = None,
    normalize_cost: bool = True,
) -> float:
    """
    Compute exact 2D Wasserstein / Earth Mover's Distance.

    Parameters
    ----------
    map_a, map_b:
        2D non-negative maps, ideally already normalized to sum to 1.

    cost_matrix:
        Optional precomputed cost matrix for speed.

    Returns
    -------
    emd:
        Wasserstein / EMD distance.
    """
    import ot

    map_a = normalize_map(map_a)
    map_b = normalize_map(map_b)

    if map_a.shape != map_b.shape:
        raise ValueError(f"Shape mismatch: {map_a.shape} vs {map_b.shape}")

    h, w = map_a.shape

    if cost_matrix is None:
        cost_matrix = make_2d_cost_matrix(
            height=h,
            width=w,
            normalize_cost=normalize_cost,
        )

    a = map_a.ravel()
    b = map_b.ravel()

    emd = ot.emd2(a, b, cost_matrix)

    return float(emd)


# ============================================================
# 3. Pair maps from two models and compare with Wasserstein
# ============================================================

def compare_models_wasserstein(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    output_size: tuple[int, int] = (32, 32),
    target_mode: str = "true_label",
    require_same_prediction_status: bool = False,
    clip_negative: bool = True,
    normalize_cost: bool = True,
) -> pd.DataFrame:
    """
    Compare attention maps between two models using 2D Wasserstein distance.

    The function pairs maps by:
        image_id, sigma, target_label

    Parameters
    ----------
    df:
        Metadata dataframe from merge_metadata_files.

    model_a, model_b:
        Names of the two models to compare.

    output_size:
        Spatial size used for Wasserstein comparison.
        Recommended: (32, 32) or (48, 48).
        Larger is slower.

    target_mode:
        Which maps to compare.

        "true_label":
            Compare maps where target_label == true_label.
            This asks:
                Do the models allocate true-class relevance differently?

        "pred_label":
            Compare maps where target_label == pred_label.
            This asks:
                Do the models allocate relevance differently for their own decisions?

            Note: if models predict different classes, the target_label may differ,
            so fewer pairs may be available unless both predictions match.

        "all":
            Use all available target_label maps.

    require_same_prediction_status:
        If True, only compare images where both models are either correct or incorrect.
        Usually False is better.

    clip_negative:
        Clip negative relevance before normalization.

    normalize_cost:
        If True, Wasserstein distance is normalized by image diagonal,
        so values are roughly in [0, 1].

    Returns
    -------
    results_df:
        One row per paired map comparison.
    """
    required_cols = {
        "image_id",
        "model",
        "sigma",
        "true_label",
        "pred_label",
        "target_label",
        "correct",
        "map_path",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    work = df.copy()

    if target_mode == "true_label":
        work = work[work["target_label"] == work["true_label"]]
    elif target_mode == "pred_label":
        work = work[work["target_label"] == work["pred_label"]]
    elif target_mode == "all":
        pass
    else:
        raise ValueError("target_mode must be one of: 'true_label', 'pred_label', 'all'")

    df_a = work[work["model"] == model_a].copy()
    df_b = work[work["model"] == model_b].copy()

    # Pair maps from model A and model B.
    pair_cols = ["image_id", "target_label"]

    paired = df_a.merge(
        df_b,
        on=pair_cols,
        suffixes=("_a", "_b"),
        how="inner",
    )

    if require_same_prediction_status:
        paired = paired[paired["correct_a"] == paired["correct_b"]]

    h, w = output_size
    cost_matrix = make_2d_cost_matrix(
        height=h,
        width=w,
        normalize_cost=normalize_cost,
    )

    rows = []

    for _, row in tqdm(
        paired.iterrows(),
        total=len(paired),
        desc=f"Wasserstein: {model_a} vs {model_b}",
    ):
        try:
            map_a = preprocess_map_for_wasserstein(
                row["map_path_a"],
                output_size=output_size,
                clip_negative=clip_negative,
            )

            map_b = preprocess_map_for_wasserstein(
                row["map_path_b"],
                output_size=output_size,
                clip_negative=clip_negative,
            )

            emd = wasserstein_distance_2d(
                map_a,
                map_b,
                cost_matrix=cost_matrix,
                normalize_cost=normalize_cost,
            )

            rows.append({
                "image_id": row["image_id"],
                "target_label": row["target_label"],

                "model_a": model_a,
                "model_b": model_b,

                "sigma_a": row["sigma_a"],
                "sigma_b": row["sigma_b"],

                "true_label_a": row["true_label_a"],
                "true_label_b": row["true_label_b"],

                "pred_label_a": row["pred_label_a"],
                "pred_label_b": row["pred_label_b"],

                "correct_a": row["correct_a"],
                "correct_b": row["correct_b"],

                "pred_confidence_a": row.get("pred_confidence_a", np.nan),
                "pred_confidence_b": row.get("pred_confidence_b", np.nan),

                "true_confidence_a": row.get("true_confidence_a", np.nan),
                "true_confidence_b": row.get("true_confidence_b", np.nan),

                "wasserstein_distance": emd,

                "map_path_a": row["map_path_a"],
                "map_path_b": row["map_path_b"],
            })

        except Exception as exc:
            print(
                f"Skipping image_id={row['image_id']}, "
                f"target={row['target_label']} because: {exc}"
            )

    return pd.DataFrame(rows)


# ============================================================
# 3. Pair maps from two models and compare with Wasserstein
# ============================================================
models2comp = ['RAF_blur0_pretrained0-8_freeze',
               'RAF_blur0_freeze']

tar_mode = 'true_label'

df = pd.read_csv("attention_maps/NATIVE_RESOLUTION/metadata_all_models_native.csv")

wasserstein_df = compare_models_wasserstein(
    df=df,
    model_a=models2comp[0],
    model_b=models2comp[1],
    output_size=(32, 32),
    target_mode=tar_mode,
    require_same_prediction_status=False,
    clip_negative=True,
    normalize_cost=True,
)

wasserstein_df.to_csv(
    f"attention_comparisons/wasserstein_"
    f"{models2comp[0].replace('_','').replace('-','')}_vs_{models2comp[1].replace('_','').replace('-','')}_"
    "native_"
    f"{tar_mode}.csv",
    index=False,
)

wasserstein_df.head()

# ============================================================
# Summaries
# ============================================================

# ============================================================
# 1. compare emotions
# ============================================================

summary_by_emotion = (
    wasserstein_df
    .groupby("target_label")
    .agg(
        mean_wasserstein=("wasserstein_distance", "mean"),
        median_wasserstein=("wasserstein_distance", "median"),
        std_wasserstein=("wasserstein_distance", "std"),
        n=("wasserstein_distance", "size"),
    )
    .reset_index()
)

summary_by_emotion

# ============================================================
# 2. Only A-correct, B-wrong
# ============================================================
a_correct_b_wrong = wasserstein_df[
    (wasserstein_df["correct_a"] == True) &
    (wasserstein_df["correct_b"] == False)
].copy()

summary_a_correct_b_wrong = (
    a_correct_b_wrong
    .groupby("target_label")
    .agg(
        mean_wasserstein=("wasserstein_distance", "mean"),
        median_wasserstein=("wasserstein_distance", "median"),
        n=("wasserstein_distance", "size"),
    )
    .reset_index()
)

summary_a_correct_b_wrong

# ============================================================
# 3. Only mixed-correct, lowres-correct
# ============================================================
both_correct = wasserstein_df[
    (wasserstein_df["correct_a"] == True) &
    (wasserstein_df["correct_b"] == True)
].copy()

summary_both_correct = (
    both_correct
    .groupby("target_label")
    .agg(
        mean_wasserstein=("wasserstein_distance", "mean"),
        median_wasserstein=("wasserstein_distance", "median"),
        n=("wasserstein_distance", "size"),
    )
    .reset_index()
)

summary_both_correct
