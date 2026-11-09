"""
2/7/26
Calculate separability based on individual image-maps and within- vs between-class comparison.
"""
from __future__ import annotations

import re
import json
from tqdm.auto import tqdm
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Hashable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    import ot
except ImportError as exc:
    raise ImportError(
        "Python Optimal Transport is required. Install it using:\n"
        "pip install POT"
    ) from exc


if __package__:
    from .wasserstein import make_2d_cost_matrix, wasserstein_distance_2d, prepare_map
else:
    from wasserstein import make_2d_cost_matrix, wasserstein_distance_2d, prepare_map


# ============================================================
# 1. Load and validate normalized attention maps
# ============================================================

def load_normalized_attention_map(
    map_path: str | Path,
    clip_negative: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Load a saved attention/relevance map and ensure that it is a valid
    probability distribution over spatial positions.

    Even if maps were normalized before saving, the function renormalizes
    them to remove small numerical discrepancies.

    Parameters
    ----------
    map_path:
        Path to a saved .npy map.

    clip_negative:
        If True, negative values are set to zero.

    eps:
        Numerical tolerance used to detect empty maps.

    Returns
    -------
    relevance:
        A finite, nonnegative 2D map summing to 1.
    """
    map_path = Path(map_path)

    if not map_path.exists():
        raise FileNotFoundError(
            f"Could not find attention map: {map_path}"
        )

    relevance = np.load(map_path)
    relevance = np.asarray(relevance, dtype=np.float64)

    if relevance.ndim == 3 and relevance.shape[0] == 1:
        relevance = relevance[0]

    if relevance.ndim != 2:
        raise ValueError(
            f"Expected a 2D map at {map_path}, "
            f"but got shape {relevance.shape}."
        )

    if not np.all(np.isfinite(relevance)):
        raise ValueError(
            f"Map contains NaN or infinite values: {map_path}"
        )

    if clip_negative:
        relevance = np.maximum(relevance, 0.0)
    elif np.any(relevance < 0):
        raise ValueError(
            f"Map contains negative values: {map_path}"
        )

    total = relevance.sum()

    if total <= eps:
        raise ValueError(
            f"Map contains no positive relevance mass: {map_path}"
        )

    return relevance / total


def resize_probability_map(
    relevance: np.ndarray,
    output_size: tuple[int, int] = (32, 32),
) -> np.ndarray:
    """
    Resize a normalized map and renormalize it.

    Downsampling is strongly recommended because exact optimal transport
    becomes expensive for large maps.
    """
    relevance = np.asarray(relevance, dtype=np.float64)

    tensor = torch.as_tensor(
        relevance,
        dtype=torch.float64,
    )[None, None]

    resized = F.interpolate(
        tensor,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()

    resized = np.maximum(resized, 0.0)

    total = resized.sum()

    if total <= 0:
        raise ValueError(
            "Resized relevance map contains no positive mass."
        )

    return resized / total


# ============================================================
# 2. Wasserstein distance
# ============================================================

def create_spatial_cost_matrix(height, width, normalize_coordinates=True):
    """Shared Euclidean cost: diagonal units by default, grid units otherwise."""
    return make_2d_cost_matrix(height, width, normalize_coordinates)


def wasserstein_2d_distance_from_prepared_maps(
    map_a: np.ndarray,
    map_b: np.ndarray,
    cost_matrix: np.ndarray,
) -> float:
    """
    Compute exact 2D Wasserstein-1 distance between two already resized,
    normalized probability maps.
    """
    return wasserstein_distance_2d(map_a, map_b, cost_matrix=cost_matrix)


# ============================================================
# 3. Prepare a matched dataset across models
# ============================================================

def prepare_matched_attention_metadata(
    metadata_df: pd.DataFrame,
    models: Sequence[Hashable] | None = None,
    sigma: Hashable | None = None,
    target_map_type: str | None = None,
    require_correct_all_models: bool = True,
    image_id_col: str = "image_id",
    model_col: str = "model",
    label_col: str = "true_label",
    map_path_col: str = "map_path",
    correct_col: str = "correct",
    target_label_col: str = "target_label",
    pred_label_col: str = "pred_label",
) -> pd.DataFrame:
    """
    Prepare one selected map per image and model and retain only matched images.

    Typical primary analysis
    ------------------------
    Use predicted-class maps from images correctly classified by every model:

        target_map_type="predicted"
        require_correct_all_models=True

    Parameters
    ----------
    metadata_df:
        Map metadata dataframe.

    models:
        Models to include. If None, all models are used.

    sigma:
        Optional test blur level to select, such as 8.

    target_map_type:
        Optional map filtering rule:

        "predicted":
            Keep rows where target_label == pred_label.

        "true":
            Keep rows where target_label == true_label.

        None:
            Assume metadata already contains exactly one desired map per
            image and model.

    require_correct_all_models:
        If True, retain only images classified correctly by every selected
        model.

    Returns
    -------
    matched_df:
        One row per matched image and model.
    """
    required_cols = {
        image_id_col,
        model_col,
        label_col,
        map_path_col,
    }

    if require_correct_all_models:
        required_cols.add(correct_col)

    if target_map_type is not None:
        required_cols.add(target_label_col)

        if target_map_type == "predicted":
            required_cols.add(pred_label_col)

    missing = required_cols - set(metadata_df.columns)

    if missing:
        raise ValueError(
            f"Metadata is missing required columns: {sorted(missing)}"
        )

    work = metadata_df.copy()

    if models is None:
        models = list(pd.unique(work[model_col]))
    else:
        models = list(models)

    if len(models) < 1:
        raise ValueError("At least one model must be selected.")

    work = work[
        work[model_col].isin(models)
    ].copy()

    if sigma is not None:
        if "sigma" not in work.columns:
            raise ValueError(
                "A sigma was requested, but metadata has no 'sigma' column."
            )

        work = work[
            work["sigma"] == sigma
        ].copy()

    if target_map_type == "predicted":
        work = work[
            work[target_label_col] == work[pred_label_col]
        ].copy()

    elif target_map_type == "true":
        work = work[
            work[target_label_col] == work[label_col]
        ].copy()

    elif target_map_type is not None:
        raise ValueError(
            "target_map_type must be 'predicted', 'true', or None."
        )

    duplicate_cols = [
        image_id_col,
        model_col,
    ]

    if "sigma" in work.columns:
        duplicate_cols.append("sigma")

    duplicate_mask = work.duplicated(
        subset=duplicate_cols,
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_rows = work.loc[
            duplicate_mask,
            duplicate_cols
            + [
                col
                for col in [
                    target_label_col,
                    pred_label_col,
                    map_path_col,
                ]
                if col in work.columns
            ],
        ]

        raise ValueError(
            "More than one selected map was found for an image/model "
            "condition. Ensure that only one map type is selected.\n"
            f"{duplicate_rows.head(30).to_string(index=False)}"
        )

    # Retain images present for every selected model.
    model_counts = (
        work
        .groupby(image_id_col)[model_col]
        .nunique()
    )

    matched_image_ids = model_counts[
        model_counts == len(models)
    ].index

    work = work[
        work[image_id_col].isin(matched_image_ids)
    ].copy()

    if require_correct_all_models:
        correctness = (
            work
            .groupby(image_id_col)[correct_col]
            .all()
        )

        shared_correct_ids = correctness[
            correctness
        ].index

        work = work[
            work[image_id_col].isin(shared_correct_ids)
        ].copy()

    # Ensure that the true label is identical across models for each image.
    label_counts = (
        work
        .groupby(image_id_col)[label_col]
        .nunique()
    )

    inconsistent_ids = label_counts[
        label_counts != 1
    ].index

    if len(inconsistent_ids) > 0:
        raise ValueError(
            "Some image IDs have inconsistent true labels across models: "
            f"{list(inconsistent_ids[:20])}"
        )

    if work.empty:
        raise ValueError(
            "No matched images remain after filtering."
        )

    return work.sort_values(
        [model_col, image_id_col]
    ).reset_index(drop=True)


# ============================================================
# 4. Load and prepare all attention maps
# ============================================================

def load_prepared_attention_maps(
    matched_df: pd.DataFrame,
    transport_size: tuple[int, int] = (32, 32),
    image_id_col: str = "image_id",
    model_col: str = "model",
    label_col: str = "true_label",
    map_path_col: str = "map_path",
    clip_negative: bool = True,
) -> dict[Hashable, dict[str, object]]:
    """
    Load, resize, and organize maps by model.

    Returns
    -------
    prepared:
        {
            model_name: {
                "image_ids": np.ndarray,
                "labels": np.ndarray,
                "maps": np.ndarray [N, H, W],
                "metadata": pd.DataFrame,
            }
        }
    """
    prepared = {}

    for model, model_df in matched_df.groupby(
        model_col,
        sort=False,
    ):
        model_df = model_df.sort_values(
            image_id_col
        ).reset_index(drop=True)

        maps = []

        for map_path in model_df[map_path_col]:
            relevance = load_normalized_attention_map(
                map_path,
                clip_negative=clip_negative,
            )

            # relevance = resize_probability_map(
            #     relevance,
            #     output_size=transport_size,
            # )

            maps.append(relevance)

        prepared[model] = {
            "image_ids": model_df[image_id_col].to_numpy(),
            "labels": model_df[label_col].to_numpy(),
            "maps": np.stack(maps, axis=0),
            "metadata": model_df,
        }

    # Verify that all models contain identical images in identical order.
    models = list(prepared)

    reference_ids = prepared[models[0]]["image_ids"]
    reference_labels = prepared[models[0]]["labels"]

    for model in models[1:]:
        if not np.array_equal(
            reference_ids,
            prepared[model]["image_ids"],
        ):
            raise ValueError(
                f"Image ordering differs for model {model!r}."
            )

        if not np.array_equal(
            reference_labels,
            prepared[model]["labels"],
        ):
            raise ValueError(
                f"Labels differ for model {model!r}."
            )

    return prepared


# ============================================================
# 5. Compute all image-to-image distances
# ============================================================



def _safe_filename(value: object) -> str:
    """
    Convert a model name into a safe filename component.
    """
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_") or "unnamed_model"


def _pairwise_cache_paths(
    cache_dir: str | Path,
    model: Hashable,
) -> tuple[Path, Path]:
    """
    Return data and metadata paths for one model's cached results.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    safe_model = _safe_filename(model)

    data_path = cache_dir / f"pairwise_distances__{safe_model}.pkl"
    metadata_path = cache_dir / f"pairwise_distances__{safe_model}.json"

    return data_path, metadata_path


def _make_pairwise_cache_metadata(
    model: Hashable,
    image_ids: np.ndarray,
    labels: np.ndarray,
    transport_size: tuple[int, int],
    normalize_coordinates: bool,
) -> dict:
    """
    Create metadata used to verify that a cached result matches the
    current analysis inputs.
    """
    return {
        "distance_metric_version": "w1_diagonal_v1",
        "model": str(model),
        "n_images": int(len(image_ids)),
        "image_ids": [str(x) for x in image_ids],
        "labels": [str(x) for x in labels],
        "transport_size": [
            int(transport_size[0]),
            int(transport_size[1]),
        ],
        "normalize_coordinates": bool(normalize_coordinates),
        "expected_n_pairs": int(
            len(image_ids) * (len(image_ids) - 1) // 2
        ),
    }


def _cache_metadata_matches(
    cached_metadata: dict,
    current_metadata: dict,
) -> bool:
    """
    Check whether cached results correspond to the current model inputs
    and analysis settings.
    """
    fields_to_compare = [
        "distance_metric_version",
        "model",
        "n_images",
        "image_ids",
        "labels",
        "transport_size",
        "normalize_coordinates",
        "expected_n_pairs",
    ]

    return all(
        cached_metadata.get(field) == current_metadata.get(field)
        for field in fields_to_compare
    )


def compute_pairwise_image_distances(
    prepared_maps: dict[Hashable, dict[str, object]],
    transport_size: tuple[int, int] = (32, 32),
    normalize_coordinates: bool = True,
    cache_dir: str | Path | None = None,
    overwrite_cache: bool = False,
    validate_cache: bool = True,
) -> pd.DataFrame:
    """
    Compute Wasserstein distance between every pair of images separately
    for each model, with model-level caching.

    For each model:
        1. Check whether cached results already exist.
        2. If valid cached results exist, load them.
        3. Otherwise compute all image-pair distances.
        4. Save that model's results immediately after completion.

    Parameters
    ----------
    prepared_maps:
        Dictionary returned by load_prepared_attention_maps().

    transport_size:
        Spatial resolution used for Wasserstein transport.

    normalize_coordinates:
        If True, Euclidean costs are divided by the grid diagonal (range [0, 1]).

    cache_dir:
        Directory for model-level cached pairwise results.

        If None, caching is disabled.

    overwrite_cache:
        If True, recalculate every model even when cached files exist.

    validate_cache:
        If True, verify that cached files use the same:
            image IDs
            labels
            transport size
            coordinate normalization

        Strongly recommended.

    Returns
    -------
    pairwise_df:
        Combined dataframe containing all selected models.
    """
    height, width = transport_size

    cost_matrix = create_spatial_cost_matrix(
        height,
        width,
        normalize_coordinates=normalize_coordinates,
    )

    model_dataframes = []

    for model, model_data in prepared_maps.items():
        image_ids = np.asarray(model_data["image_ids"])
        labels = np.asarray(model_data["labels"])
        maps = np.asarray(model_data["maps"])

        n_images = len(image_ids)
        expected_n_pairs = n_images * (n_images - 1) // 2

        current_metadata = _make_pairwise_cache_metadata(
            model=model,
            image_ids=image_ids,
            labels=labels,
            transport_size=transport_size,
            normalize_coordinates=normalize_coordinates,
        )

        cached_df = None

        if cache_dir is not None:
            data_path, metadata_path = _pairwise_cache_paths(
                cache_dir=cache_dir,
                model=model,
            )

            cache_exists = (
                data_path.exists()
                and metadata_path.exists()
            )

            if cache_exists and not overwrite_cache:
                use_cache = True

                if validate_cache:
                    try:
                        with open(
                            metadata_path,
                            "r",
                            encoding="utf-8",
                        ) as file:
                            cached_metadata = json.load(file)

                        use_cache = _cache_metadata_matches(
                            cached_metadata=cached_metadata,
                            current_metadata=current_metadata,
                        )

                        if not use_cache:
                            print(
                                f"Cache does not match current inputs "
                                f"for model {model!r}; recalculating."
                            )

                    except Exception as exc:
                        print(
                            f"Could not validate cache for model "
                            f"{model!r}: {exc}. Recalculating."
                        )
                        use_cache = False

                if use_cache:
                    try:
                        cached_df = pd.read_pickle(data_path)

                        required_cols = {
                            "model",
                            "image_id_a",
                            "image_id_b",
                            "label_a",
                            "label_b",
                            "same_emotion",
                            "wasserstein",
                        }

                        missing_cols = (
                            required_cols
                            - set(cached_df.columns)
                        )

                        if missing_cols:
                            raise ValueError(
                                f"Cached dataframe is missing columns: "
                                f"{sorted(missing_cols)}"
                            )

                        if len(cached_df) != expected_n_pairs:
                            raise ValueError(
                                f"Expected {expected_n_pairs} pairs, "
                                f"but cache contains "
                                f"{len(cached_df)}."
                            )

                        print(
                            f"Loaded cached pairwise distances for "
                            f"{model!r}: {len(cached_df):,} pairs."
                        )

                    except Exception as exc:
                        print(
                            f"Could not load valid cache for model "
                            f"{model!r}: {exc}. Recalculating."
                        )
                        cached_df = None

        if cached_df is not None:
            model_dataframes.append(cached_df)
            continue

        rows = []

        progress = tqdm(
            total=expected_n_pairs,
            desc=f"Wasserstein pairs: {model}",
            unit="pair",
        )

        try:
            for i in range(n_images - 1):
                for j in range(i + 1, n_images):
                    distance = (
                        wasserstein_2d_distance_from_prepared_maps(
                            maps[i],
                            maps[j],
                            cost_matrix=cost_matrix,
                        )
                    )

                    rows.append({
                        "model": model,
                        "image_id_a": image_ids[i],
                        "image_id_b": image_ids[j],
                        "label_a": labels[i],
                        "label_b": labels[j],
                        "same_emotion": (
                            labels[i] == labels[j]
                        ),
                        "wasserstein": distance,
                    })

                    progress.update(1)

        finally:
            progress.close()

        model_df = pd.DataFrame(rows)

        if len(model_df) != expected_n_pairs:
            raise RuntimeError(
                f"Pairwise calculation for model {model!r} "
                f"produced {len(model_df)} rows, but "
                f"{expected_n_pairs} were expected."
            )

        model_dataframes.append(model_df)

        if cache_dir is not None:
            data_path, metadata_path = _pairwise_cache_paths(
                cache_dir=cache_dir,
                model=model,
            )

            # Write to temporary files first, then rename.
            # This reduces the risk of leaving a corrupted cache if the
            # process is interrupted during saving.
            temporary_data_path = data_path.with_suffix(
                ".tmp.pkl"
            )
            temporary_metadata_path = metadata_path.with_suffix(
                ".tmp.json"
            )

            model_df.to_pickle(
                temporary_data_path
            )

            with open(
                temporary_metadata_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    current_metadata,
                    file,
                    indent=2,
                )

            temporary_data_path.replace(data_path)
            temporary_metadata_path.replace(metadata_path)

            print(
                f"Saved pairwise distances for {model!r} to:\n"
                f"  {data_path}"
            )

    if not model_dataframes:
        raise ValueError(
            "No pairwise image distances were available."
        )

    pairwise_df = pd.concat(
        model_dataframes,
        ignore_index=True,
    )

    return pairwise_df

# ============================================================
# 6. Compute one separability score per image
# ============================================================


def compute_per_image_separability(
    pairwise_df: pd.DataFrame,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    For every image and model, compute:

        within_distance:
            Mean distance to images from the same emotion.

        between_distance:
            Mean distance to images from other emotions.

        distance_difference:
            between_distance - within_distance

        distance_ratio:
            between_distance / within_distance

    Larger difference and ratio indicate greater emotion separability.
    """
    required_cols = {
        "model",
        "image_id_a",
        "image_id_b",
        "label_a",
        "label_b",
        "same_emotion",
        "wasserstein",
    }

    missing = required_cols - set(pairwise_df.columns)

    if missing:
        raise ValueError(
            f"pairwise_df is missing columns: {sorted(missing)}"
        )

    # Expand each unordered pair into two directed image-level observations.
    side_a = pairwise_df.rename(
        columns={
            "image_id_a": "image_id",
            "image_id_b": "other_image_id",
            "label_a": "true_label",
            "label_b": "other_label",
        }
    )[
        [
            "model",
            "image_id",
            "other_image_id",
            "true_label",
            "other_label",
            "same_emotion",
            "wasserstein",
        ]
    ]

    side_b = pairwise_df.rename(
        columns={
            "image_id_b": "image_id",
            "image_id_a": "other_image_id",
            "label_b": "true_label",
            "label_a": "other_label",
        }
    )[
        [
            "model",
            "image_id",
            "other_image_id",
            "true_label",
            "other_label",
            "same_emotion",
            "wasserstein",
        ]
    ]

    directed_df = pd.concat(
        [side_a, side_b],
        ignore_index=True,
    )

    within_df = (
        directed_df[
            directed_df["same_emotion"]
        ]
        .groupby(
            ["model", "image_id", "true_label"],
            as_index=False,
        )
        .agg(
            within_distance=("wasserstein", "mean"),
            within_median=("wasserstein", "median"),
            n_within=("wasserstein", "size"),
        )
    )

    between_df = (
        directed_df[
            ~directed_df["same_emotion"]
        ]
        .groupby(
            ["model", "image_id", "true_label"],
            as_index=False,
        )
        .agg(
            between_distance=("wasserstein", "mean"),
            between_median=("wasserstein", "median"),
            n_between=("wasserstein", "size"),
        )
    )

    per_image_df = within_df.merge(
        between_df,
        on=["model", "image_id", "true_label"],
        how="outer",
        validate="one_to_one",
    )

    missing_within = per_image_df[
        per_image_df["within_distance"].isna()
    ]

    if not missing_within.empty:
        missing_classes = sorted(
            missing_within["true_label"]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "Some images have no other image from the same emotion class. "
            "At least two images per emotion are required. "
            f"Affected classes: {missing_classes}"
        )

    missing_between = per_image_df[
        per_image_df["between_distance"].isna()
    ]

    if not missing_between.empty:
        raise ValueError(
            "Some images have no images from another emotion class."
        )

    per_image_df["distance_difference"] = (
        per_image_df["between_distance"]
        - per_image_df["within_distance"]
    )

    per_image_df["distance_ratio"] = (
        per_image_df["between_distance"]
        / np.maximum(
            per_image_df["within_distance"],
            eps,
        )
    )

    return per_image_df.sort_values(
        ["model", "true_label", "image_id"]
    ).reset_index(drop=True)


# ============================================================
# 7. Model-level and emotion-level summaries
# ============================================================

def summarize_separability(
    per_image_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Produce:

    1. Per-model summary across images.
    2. Per-model, per-emotion summary.
    3. Macro-average across emotions, giving each emotion equal weight.
    """
    model_summary_df = (
        per_image_df
        .groupby(
            "model",
            as_index=False,
        )
        .agg(
            mean_within_distance=(
                "within_distance",
                "mean",
            ),
            mean_between_distance=(
                "between_distance",
                "mean",
            ),
            mean_distance_difference=(
                "distance_difference",
                "mean",
            ),
            median_distance_difference=(
                "distance_difference",
                "median",
            ),
            mean_distance_ratio=(
                "distance_ratio",
                "mean",
            ),
            median_distance_ratio=(
                "distance_ratio",
                "median",
            ),
            std_difference_across_images=(
                "distance_difference",
                "std",
            ),
            sem_difference_across_images=(
                "distance_difference",
                lambda x: (
                    x.std(ddof=1) / np.sqrt(x.notna().sum())
                    if x.notna().sum() > 1
                    else np.nan
                ),
            ),
            n_images=(
                "image_id",
                "nunique",
            ),
        )
    )

    emotion_summary_df = (
        per_image_df
        .groupby(
            ["model", "true_label"],
            as_index=False,
        )
        .agg(
            mean_within_distance=(
                "within_distance",
                "mean",
            ),
            mean_between_distance=(
                "between_distance",
                "mean",
            ),
            mean_distance_difference=(
                "distance_difference",
                "mean",
            ),
            median_distance_difference=(
                "distance_difference",
                "median",
            ),
            mean_distance_ratio=(
                "distance_ratio",
                "mean",
            ),
            median_distance_ratio=(
                "distance_ratio",
                "median",
            ),
            std_difference_across_images=(
                "distance_difference",
                "std",
            ),
            sem_difference_across_images=(
                "distance_difference",
                lambda x: (
                    x.std(ddof=1) / np.sqrt(x.notna().sum())
                    if x.notna().sum() > 1
                    else np.nan
                ),
            ),
            n_images=(
                "image_id",
                "nunique",
            ),
        )
    )

    # Each emotion gets equal weight, regardless of sample size.
    macro_summary_df = (
        emotion_summary_df
        .groupby(
            "model",
            as_index=False,
        )
        .agg(
            macro_mean_within_distance=(
                "mean_within_distance",
                "mean",
            ),
            macro_mean_between_distance=(
                "mean_between_distance",
                "mean",
            ),
            macro_mean_distance_difference=(
                "mean_distance_difference",
                "mean",
            ),
            macro_mean_distance_ratio=(
                "mean_distance_ratio",
                "mean",
            ),
            n_emotions=(
                "true_label",
                "nunique",
            ),
        )
    )

    return (
        model_summary_df,
        emotion_summary_df,
        macro_summary_df,
    )


# ============================================================
# 8. Paired comparison between models
# ============================================================

def compare_models_paired(
    per_image_df: pd.DataFrame,
    model_a: Hashable,
    model_b: Hashable,
    score_col: str = "distance_difference",
    test: str = "wilcoxon",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare two models using matched image-level scores.

    Difference is:

        score(model_a) - score(model_b)

    Positive values indicate greater separability for model_a.
    """
    from scipy.stats import ttest_rel, wilcoxon

    required_cols = {
        "model",
        "image_id",
        "true_label",
        score_col,
    }

    missing = required_cols - set(per_image_df.columns)

    if missing:
        raise ValueError(
            f"per_image_df is missing columns: {sorted(missing)}"
        )

    data_a = (
        per_image_df[
            per_image_df["model"] == model_a
        ][
            [
                "image_id",
                "true_label",
                score_col,
            ]
        ]
        .rename(
            columns={
                score_col: "score_model_a"
            }
        )
    )

    data_b = (
        per_image_df[
            per_image_df["model"] == model_b
        ][
            [
                "image_id",
                "true_label",
                score_col,
            ]
        ]
        .rename(
            columns={
                score_col: "score_model_b"
            }
        )
    )

    matched_df = data_a.merge(
        data_b,
        on=["image_id", "true_label"],
        how="inner",
        validate="one_to_one",
    )

    if matched_df.empty:
        raise ValueError(
            "No matched images were found between the selected models."
        )

    matched_df["difference_a_minus_b"] = (
        matched_df["score_model_a"]
        - matched_df["score_model_b"]
    )

    a = matched_df["score_model_a"].to_numpy()
    b = matched_df["score_model_b"].to_numpy()

    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]

    if len(a) < 2:
        raise ValueError(
            "At least two matched images are required."
        )

    if test == "wilcoxon":
        statistic, p_value = wilcoxon(
            a,
            b,
            alternative="two-sided",
        )

    elif test == "ttest":
        statistic, p_value = ttest_rel(
            a,
            b,
            nan_policy="omit",
        )

    else:
        raise ValueError(
            "test must be 'wilcoxon' or 'ttest'."
        )

    difference = a - b

    summary_df = pd.DataFrame([{
        "model_a": model_a,
        "model_b": model_b,
        "score": score_col,
        "test": test,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "mean_model_a": float(np.mean(a)),
        "mean_model_b": float(np.mean(b)),
        "mean_difference_a_minus_b": float(
            np.mean(difference)
        ),
        "median_difference_a_minus_b": float(
            np.median(difference)
        ),
        "std_difference": float(
            np.std(difference, ddof=1)
        ),
        "sem_difference": float(
            np.std(difference, ddof=1)
            / np.sqrt(len(difference))
        ),
        "proportion_model_a_greater": float(
            np.mean(difference > 0)
        ),
        "n_matched_images": len(difference),
    }])

    return matched_df, summary_df


# ============================================================
# 9. Label-permutation test
# ============================================================

def compute_global_separability_statistic(
    pairwise_model_df: pd.DataFrame,
    labels_by_image: dict[Hashable, Hashable],
) -> float:
    """
    Compute global between-minus-within Wasserstein distance.

    Pair labels are derived from labels_by_image rather than from the
    original dataframe, allowing shuffled-label permutation tests.
    """
    label_a = pairwise_model_df[
        "image_id_a"
    ].map(labels_by_image)

    label_b = pairwise_model_df[
        "image_id_b"
    ].map(labels_by_image)

    same = label_a.to_numpy() == label_b.to_numpy()
    distances = pairwise_model_df[
        "wasserstein"
    ].to_numpy()

    if not np.any(same):
        return np.nan

    if not np.any(~same):
        return np.nan

    within = distances[same].mean()
    between = distances[~same].mean()

    return float(between - within)


def label_permutation_test_by_model(
    pairwise_df: pd.DataFrame,
    n_permutations: int = 5000,
    random_state: int | None = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Test whether attention maps cluster by true emotion more strongly than
    expected under random emotion labels.

    The maps and pairwise distances remain fixed. Emotion labels are
    shuffled across images.

    Null hypothesis
    ---------------
    Emotion labels are unrelated to attention-map geometry.

    Statistic
    ---------
    Mean between-emotion distance minus mean within-emotion distance.

    Positive observed values indicate emotion separability.
    """
    rng = np.random.default_rng(random_state)

    required_cols = {
        "model",
        "image_id_a",
        "image_id_b",
        "label_a",
        "label_b",
        "wasserstein",
    }

    missing = required_cols - set(pairwise_df.columns)

    if missing:
        raise ValueError(
            f"pairwise_df is missing columns: {sorted(missing)}"
        )

    summary_rows = []
    permutation_rows = []

    for model, model_df in pairwise_df.groupby(
        "model",
        sort=False,
    ):
        label_a_df = model_df[
            ["image_id_a", "label_a"]
        ].rename(
            columns={
                "image_id_a": "image_id",
                "label_a": "label",
            }
        )

        label_b_df = model_df[
            ["image_id_b", "label_b"]
        ].rename(
            columns={
                "image_id_b": "image_id",
                "label_b": "label",
            }
        )

        image_labels_df = (
            pd.concat(
                [label_a_df, label_b_df],
                ignore_index=True,
            )
            .drop_duplicates()
        )

        label_counts = (
            image_labels_df
            .groupby("image_id")["label"]
            .nunique()
        )

        if (label_counts != 1).any():
            raise ValueError(
                f"Inconsistent labels found for model {model!r}."
            )

        image_labels_df = (
            image_labels_df
            .drop_duplicates("image_id")
            .reset_index(drop=True)
        )

        image_ids = image_labels_df[
            "image_id"
        ].to_numpy()

        original_labels = image_labels_df[
            "label"
        ].to_numpy()

        original_mapping = dict(
            zip(image_ids, original_labels)
        )

        observed = compute_global_separability_statistic(
            model_df,
            original_mapping,
        )

        null_statistics = np.empty(
            n_permutations,
            dtype=np.float64,
        )

        for permutation_idx in range(n_permutations):
            shuffled_labels = rng.permutation(
                original_labels
            )

            shuffled_mapping = dict(
                zip(image_ids, shuffled_labels)
            )

            null_statistics[permutation_idx] = (
                compute_global_separability_statistic(
                    model_df,
                    shuffled_mapping,
                )
            )

            permutation_rows.append({
                "model": model,
                "permutation": permutation_idx,
                "separability_statistic": (
                    null_statistics[permutation_idx]
                ),
            })

        valid_null = null_statistics[
            np.isfinite(null_statistics)
        ]

        # One-sided test: observed separability greater than null.
        p_value = (
            1
            + np.sum(valid_null >= observed)
        ) / (
            len(valid_null) + 1
        )

        summary_rows.append({
            "model": model,
            "observed_between_minus_within": observed,
            "null_mean": float(
                np.mean(valid_null)
            ),
            "null_std": float(
                np.std(valid_null, ddof=1)
            ),
            "permutation_p_value_one_sided": float(
                p_value
            ),
            "n_permutations": len(valid_null),
            "n_images": len(image_ids),
        })

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(permutation_rows),
    )


# ============================================================
# 10. Paired permutation test for model differences
# ============================================================

def paired_model_difference_permutation_test(
    matched_scores_df: pd.DataFrame,
    n_permutations: int = 10000,
    random_state: int | None = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Paired sign-flip permutation test for the difference in image-level
    separability between two models.

    Null hypothesis
    ---------------
    The two models are exchangeable within each image.

    This test uses image-level model differences and does not treat all
    image-pair distances as independent.
    """
    rng = np.random.default_rng(random_state)

    differences = matched_scores_df[
        "difference_a_minus_b"
    ].to_numpy(dtype=np.float64)

    differences = differences[
        np.isfinite(differences)
    ]

    if len(differences) < 2:
        raise ValueError(
            "At least two finite paired differences are required."
        )

    observed = float(
        np.mean(differences)
    )

    null_statistics = np.empty(
        n_permutations,
        dtype=np.float64,
    )

    for permutation_idx in range(n_permutations):
        signs = rng.choice(
            [-1.0, 1.0],
            size=len(differences),
            replace=True,
        )

        null_statistics[permutation_idx] = np.mean(
            differences * signs
        )

    # Two-sided test.
    p_value = (
        1
        + np.sum(
            np.abs(null_statistics)
            >= abs(observed)
        )
    ) / (
        n_permutations + 1
    )

    summary_df = pd.DataFrame([{
        "observed_mean_model_difference": observed,
        "null_mean": float(
            np.mean(null_statistics)
        ),
        "null_std": float(
            np.std(null_statistics, ddof=1)
        ),
        "permutation_p_value_two_sided": float(
            p_value
        ),
        "n_permutations": n_permutations,
        "n_images": len(differences),
    }])

    return summary_df, null_statistics


# ============================================================
# 11. Complete wrapper
# ============================================================

def analyze_attention_map_class_separability(
    metadata_df: pd.DataFrame,
    models: Sequence[Hashable] | None = None,
    sigma: Hashable | None = None,
    target_map_type: str | None = "predicted",
    require_correct_all_models: bool = True,
    transport_size: tuple[int, int] = (32, 32),
    normalize_coordinates: bool = True,
    clip_negative: bool = True,
    n_label_permutations: int = 5000,
    random_state: int | None = 42,
    save_dir: str | Path | None = None,
    overwrite_pairwise_cache: bool = False,
) -> dict[str, object]:
    """
    Run the complete image-level attention-map separability analysis.
    """
    matched_metadata_df = prepare_matched_attention_metadata(
        metadata_df=metadata_df,
        models=models,
        sigma=sigma,
        target_map_type=target_map_type,
        require_correct_all_models=require_correct_all_models,
    )

    prepared_maps = load_prepared_attention_maps(
        matched_df=matched_metadata_df,
        transport_size=transport_size,
        clip_negative=clip_negative,
    )

    if save_dir is not None:
        save_dir = Path(save_dir)

        pairwise_cache_dir = (
                save_dir
                / "pairwise_model_cache"
        )
    else:
        pairwise_cache_dir = None

    pairwise_df = compute_pairwise_image_distances(
        prepared_maps=prepared_maps,
        transport_size=transport_size,
        normalize_coordinates=normalize_coordinates,
        cache_dir=pairwise_cache_dir,
        overwrite_cache=overwrite_pairwise_cache,
        validate_cache=True,
    )

    per_image_df = compute_per_image_separability(
        pairwise_df
    )

    (
        model_summary_df,
        emotion_summary_df,
        macro_summary_df,
    ) = summarize_separability(
        per_image_df
    )

    (
        label_permutation_summary_df,
        label_permutation_distribution_df,
    ) = label_permutation_test_by_model(
        pairwise_df=pairwise_df,
        n_permutations=n_label_permutations,
        random_state=random_state,
    )

    results = {
        "matched_metadata": matched_metadata_df,
        "pairwise_distances": pairwise_df,
        "per_image": per_image_df,
        "model_summary": model_summary_df,
        "emotion_summary": emotion_summary_df,
        "macro_summary": macro_summary_df,
        "label_permutation_summary": (
            label_permutation_summary_df
        ),
        "label_permutation_distribution": (
            label_permutation_distribution_df
        ),
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for result_name, result_value in results.items():
            if isinstance(result_value, pd.DataFrame):
                result_value.to_csv(
                    save_dir / f"{result_name}.csv",
                    index=False,
                )

        pd.to_pickle(
            results,
            save_dir / "attention_map_separability_results.pkl",
        )

    return results

if __name__ == "__main__":
    metadata_df = pd.read_csv(
        "attention_maps/NATIVE_RESOLUTION/metadata_all_models_native.csv"
    )

    models_to_compare = [
        "RAF_blur8_pretrained0-8_freeze_fix_projs",
        "RAF_blur8_freeze",
    ]

    image_subsets = {
        'all_pred':
            {
                'target_map_type': 'predicted',
                'save_dir': "attention_maps/image_attention_separability_sigma8_ALL_PRED",
                'require_correct_all_models': False,
            },
        'all_true':
            {
                'target_map_type': 'true',
                'save_dir': "attention_maps/image_attention_separability_sigma8_ALL_TRUE",
                'require_correct_all_models': False,
            },
        'correct_pred':
            {
                'target_map_type': 'predicted',
                'save_dir': "attention_maps/image_attention_separability_sigma8_CORRECT_PRED",
                'require_correct_all_models': True,
            }
    }

    for img_sub, cfg in image_subsets.items():
        print(f"Current image subset: {img_sub}")
        results = analyze_attention_map_class_separability(
            metadata_df=metadata_df,
            models=models_to_compare,
            sigma=8,
            target_map_type=cfg['target_map_type'],
            require_correct_all_models=cfg['require_correct_all_models'],
            transport_size=(14, 14),  # (14, 14) is the native size of the attention-maps, so no need to downscale.
            normalize_coordinates=True,
            clip_negative=True,
            n_label_permutations=5000,
            random_state=42,
            save_dir=cfg['save_dir'],
            overwrite_pairwise_cache=False,
        )

    # Separability for every image:
    results["per_image"][
        [
            "model",
            "image_id",
            "true_label",
            "within_distance",
            "between_distance",
            "distance_difference",
            "distance_ratio",
        ]
    ]

    # Model-level summary:
    results["model_summary"]

    # Per-emotion summary:
    results["emotion_summary"]

    # Macro-average giving each emotion equal weight:
    results["macro_summary"]

    # Label-permutation test:
    results["label_permutation_summary"]

    #
