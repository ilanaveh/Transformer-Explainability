"""
29/6/26
"""

from __future__ import annotations

import ast
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Hashable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json

# RAF order from samples/CLS2IDX.py, without importing the model-generation stack.
EMOTIONS = ["Angry", "Disgust", "Fear", "Sad", "Happy", "Surprise", "Neutral"]

try:
    import ot
except ImportError as exc:
    raise ImportError(
        "This analysis requires Python Optimal Transport. "
        "Install it with: pip install POT"
    ) from exc


if __package__:
    from .wasserstein import make_2d_cost_matrix, wasserstein_distance_2d, prepare_map, normalize_map
else:
    from wasserstein import make_2d_cost_matrix, wasserstein_distance_2d, prepare_map, normalize_map


# ============================================================
# 1. Load saved prototypes
# ============================================================

def load_saved_prototypes(
    prototype_dir: str | Path,
    summary_filename: str = "prototype_summary.csv",
) -> dict:
    """
    Load previously saved prototypes from disk.

    Expected prototype key format
    -----------------------------
    The default analysis below assumes keys such as:

        (model, sigma, emotion)

    For example:

        ("mixed_pretrain", 8, "happy")

    Expected summary columns
    ------------------------
    Required:
        key
        prototype_path
        n

    Optional:
        mean_face_path

    Returns
    -------
    prototypes:
        Dictionary in the same format returned by
        build_attention_prototypes():

        {
            ("mixed_pretrain", 8, "happy"): {
                "prototype": np.ndarray,
                "mean_face": np.ndarray or None,
                "n": int,
                "map_paths": [],
            }
        }
    """
    prototype_dir = Path(prototype_dir)
    summary_path = prototype_dir / summary_filename

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Could not find prototype summary file: {summary_path}"
        )

    summary_df = pd.read_csv(summary_path)

    required_cols = {"key", "prototype_path", "n"}
    missing_cols = required_cols - set(summary_df.columns)

    if missing_cols:
        raise ValueError(
            f"Summary file is missing required columns: "
            f"{sorted(missing_cols)}"
        )

    has_mean_face_path = "mean_face_path" in summary_df.columns
    prototypes = {}

    for row_idx, row in summary_df.iterrows():
        try:
            key = ast.literal_eval(str(row["key"]))
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"Could not parse prototype key in row {row_idx}: "
                f"{row['key']!r}"
            ) from exc

        if not isinstance(key, tuple):
            raise ValueError(
                f"Expected key to be a tuple, but row {row_idx} "
                f"contains: {key!r}"
            )

        prototype_path = _resolve_saved_path(
            saved_path=row["prototype_path"],
            base_dir=prototype_dir,
        )

        if not prototype_path.exists():
            raise FileNotFoundError(
                f"Could not find prototype file: {prototype_path}"
            )

        mean_face = None

        if has_mean_face_path and pd.notna(row["mean_face_path"]):
            mean_face_path = _resolve_saved_path(
                saved_path=row["mean_face_path"],
                base_dir=prototype_dir,
            )

            if mean_face_path.exists():
                mean_face = np.load(mean_face_path)

        prototype = np.load(prototype_path)

        if prototype.ndim == 3 and prototype.shape[0] == 1:
            prototype = prototype[0]

        if prototype.ndim != 2:
            raise ValueError(
                f"Prototype for key {key!r} must be 2D, "
                f"but its shape is {prototype.shape}"
            )

        prototypes[key] = {
            "prototype": prototype.astype(np.float64),
            "mean_face": mean_face,
            "n": int(row["n"]),
            "map_paths": [],
        }

    return prototypes


def _resolve_saved_path(
    saved_path: str | Path,
    base_dir: Path,
) -> Path:
    """
    Resolve a path stored in the prototype summary.

    Supports:
        absolute paths
        relative paths
        paths whose original parent directory has moved
    """
    saved_path = Path(str(saved_path))

    if saved_path.exists():
        return saved_path

    relative_candidate = base_dir / saved_path

    if relative_candidate.exists():
        return relative_candidate

    filename_candidate = base_dir / saved_path.name

    if filename_candidate.exists():
        return filename_candidate

    return filename_candidate


# ============================================================
# 2. Prepare relevance maps as spatial probability maps
# ============================================================

def ensure_2d_map(relevance: np.ndarray) -> np.ndarray:
    """
    Ensure that a relevance map is a finite 2D float64 array.
    """
    relevance = np.asarray(relevance, dtype=np.float64)

    if relevance.ndim == 3 and relevance.shape[0] == 1:
        relevance = relevance[0]

    if relevance.ndim != 2:
        raise ValueError(
            f"Expected a 2D relevance map, got shape {relevance.shape}"
        )

    if not np.all(np.isfinite(relevance)):
        raise ValueError(
            "Relevance map contains NaN or infinite values."
        )

    return relevance


def normalize_probability_map(
    relevance: np.ndarray,
    clip_negative: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Convert a relevance map into a spatial probability distribution.

    The returned map:
        is nonnegative
        sums to 1

    Parameters
    ----------
    relevance:
        Two-dimensional relevance map.

    clip_negative:
        If True, negative relevance is set to zero.

        This is normally appropriate when the analysis concerns positive
        relevance allocation.

    eps:
        Numerical threshold used to detect empty maps.
    """
    return normalize_map(relevance, eps=eps, clip_negative=clip_negative)


def resize_probability_map(
    relevance: np.ndarray,
    output_size: tuple[int, int] = (32, 32),
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Resize and renormalize a relevance map.

    Downsampling substantially reduces the cost of exact optimal transport.
    Coordinates are normalized later, so the resulting Wasserstein distances
    use diagonal units; interpolation can still change the distributions.
    """
    relevance = normalize_probability_map(
        relevance,
        clip_negative=clip_negative,
    )

    tensor = torch.as_tensor(
        relevance,
        dtype=torch.float64,
    )[None, None, :, :]

    resized = F.interpolate(
        tensor,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()

    return normalize_probability_map(
        resized,
        clip_negative=True,
    )


# ============================================================
# 3. Exact two-dimensional Wasserstein-1 distance
# ============================================================

def create_spatial_cost_matrix(height, width, normalize_coordinates=True):
    """Shared Euclidean cost: diagonal units by default, grid units otherwise."""
    return make_2d_cost_matrix(height, width, normalize_coordinates)


def wasserstein_2d_distance(
    map_a: np.ndarray,
    map_b: np.ndarray,
    transport_size: tuple[int, int] = (32, 32),
    normalize_coordinates: bool = True,
    clip_negative: bool = True,
) -> float:
    """
    Compute exact 2D Wasserstein-1 distance between two relevance maps.

    This uses Earth Mover's Distance with Euclidean spatial transport cost.

    Parameters
    ----------
    map_a, map_b:
        Two-dimensional relevance maps.

    transport_size:
        Spatial resolution used for optimal transport.

        A value such as (32, 32) usually provides a good balance between
        spatial precision and computation time.

    normalize_coordinates:
        If True, Euclidean costs are divided by the grid diagonal (range [0, 1]).

        This makes the result comparable across map resolutions.

    clip_negative:
        If True, negative values are removed before normalization.

    Returns
    -------
    distance:
        Exact Wasserstein-1 distance.

        Larger values mean that the relevance distributions are more
        spatially separated.
    """
    if len(transport_size) != 2:
        raise ValueError(
            "transport_size must contain exactly two values: "
            "(height, width)"
        )

    height, width = transport_size

    prob_a = resize_probability_map(
        map_a,
        output_size=transport_size,
        clip_negative=clip_negative,
    )

    prob_b = resize_probability_map(
        map_b,
        output_size=transport_size,
        clip_negative=clip_negative,
    )

    return wasserstein_distance_2d(
        prob_a, prob_b, normalize_cost=normalize_coordinates,
    )


# ============================================================
# 4. Parse prototype keys
# ============================================================

def prototypes_to_records(
    prototypes: dict,
    model_key_index: int = 0,
    sigma_key_index: int = 1,
    class_key_index: int = 2,
) -> pd.DataFrame:
    """
    Convert the prototype dictionary into a metadata dataframe.

    The default assumes keys have the structure:

        (model, sigma, class_name)

    Parameters
    ----------
    prototypes:
        Dictionary returned by load_saved_prototypes().

    model_key_index:
        Position of the model name in each prototype key.

    sigma_key_index:
        Position of sigma in each prototype key.

    class_key_index:
        Position of the emotion/class name in each prototype key.
    """
    rows = []

    required_max_index = max(
        model_key_index,
        sigma_key_index,
        class_key_index,
    )

    for key, value in prototypes.items():
        if not isinstance(key, tuple):
            raise ValueError(
                f"Prototype key must be a tuple, got {key!r}"
            )

        if len(key) <= required_max_index:
            raise ValueError(
                f"Prototype key {key!r} is too short for indices "
                f"model={model_key_index}, sigma={sigma_key_index}, "
                f"class={class_key_index}."
            )

        rows.append({
            "key": key,
            "model": key[model_key_index],
            "sigma": key[sigma_key_index],
            "class_name": key[class_key_index],
            "prototype": value["prototype"],
            "n": int(value["n"]),
        })

    records_df = pd.DataFrame(rows)

    if records_df.empty:
        raise ValueError("No prototypes were loaded.")

    duplicate_mask = records_df.duplicated(
        subset=["model", "sigma", "class_name"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = records_df.loc[
            duplicate_mask,
            ["model", "sigma", "class_name", "key"],
        ]

        raise ValueError(
            "Multiple prototypes map to the same model, sigma, and class:\n"
            f"{duplicates.to_string(index=False)}"
        )

    return records_df


# ============================================================
# 5. Compute emotion separability within each model
# ============================================================

def compute_emotion_separability(
    prototypes: dict,
    emotions: Sequence[Hashable] | None = None,
    transport_size: tuple[int, int] = (32, 32),
    normalize_coordinates: bool = True,
    clip_negative: bool = True,
    model_key_index: int = 0,
    sigma_key_index: int = 1,
    class_key_index: int = 2,
    require_complete_class_set: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise emotion separability within each model and sigma.

    Separability is defined as the Wasserstein-1 distance between every pair
    of emotion prototypes.

    Parameters
    ----------
    prototypes:
        Dictionary returned by load_saved_prototypes().

    emotions:
        Classes to include, in the desired order.

        If None, all classes found in the prototypes are used.

    transport_size:
        Downsampled size used for optimal transport.

    normalize_coordinates:
        Divide Euclidean transport costs by the grid diagonal (range [0, 1]).

    clip_negative:
        Remove negative relevance before probability normalization.

    model_key_index, sigma_key_index, class_key_index:
        Positions of model, sigma, and class in each prototype key.

    require_complete_class_set:
        If True, raise an error when a model/sigma condition is missing one
        or more requested classes.

        If False, compute distances using whatever classes are available.

    Returns
    -------
    pairwise_df:
        One row per model, sigma, and emotion pair.

    separability_by_sigma_df:
        Summary separability for each model and sigma.

        The main separability statistic is:
            mean_wasserstein

    separability_by_model_df:
        Overall summary for each model, averaging the class-pair distances
        across its sigma conditions.
    """
    records_df = prototypes_to_records(
        prototypes=prototypes,
        model_key_index=model_key_index,
        sigma_key_index=sigma_key_index,
        class_key_index=class_key_index,
    )

    if emotions is None:
        emotions = list(
            pd.unique(records_df["class_name"])
        )
    else:
        emotions = list(emotions)

    if len(emotions) < 2:
        raise ValueError(
            "At least two emotion classes are required."
        )

    expected_pairs = len(emotions) * (len(emotions) - 1) // 2
    pairwise_rows = []

    grouped = records_df.groupby(
        ["model", "sigma"],
        sort=False,
        dropna=False,
    )

    for (model, sigma), condition_df in grouped:
        condition_by_class = {
            row["class_name"]: row
            for _, row in condition_df.iterrows()
        }

        missing_emotions = [
            emotion
            for emotion in emotions
            if emotion not in condition_by_class
        ]

        if missing_emotions and require_complete_class_set:
            raise ValueError(
                f"Missing prototypes for model={model!r}, "
                f"sigma={sigma!r}: {missing_emotions}"
            )

        available_emotions = [
            emotion
            for emotion in emotions
            if emotion in condition_by_class
        ]

        if len(available_emotions) < 2:
            continue

        for class_a, class_b in combinations(
            available_emotions,
            2,
        ):
            row_a = condition_by_class[class_a]
            row_b = condition_by_class[class_b]

            distance = wasserstein_2d_distance(
                map_a=row_a["prototype"],
                map_b=row_b["prototype"],
                transport_size=transport_size,
                normalize_coordinates=normalize_coordinates,
                clip_negative=clip_negative,
            )

            pairwise_rows.append({
                "model": model,
                "sigma": sigma,
                "class_a": class_a,
                "class_b": class_b,
                "wasserstein": distance,
                "n_class_a": int(row_a["n"]),
                "n_class_b": int(row_b["n"]),
            })

    pairwise_df = pd.DataFrame(pairwise_rows)

    if pairwise_df.empty:
        raise ValueError(
            "No valid emotion pairs could be compared."
        )

    separability_by_sigma_df = (
        pairwise_df
        .groupby(
            ["model", "sigma"],
            as_index=False,
            dropna=False,
        )
        .agg(
            mean_wasserstein=("wasserstein", "mean"),
            median_wasserstein=("wasserstein", "median"),
            std_wasserstein=("wasserstein", "std"),
            min_wasserstein=("wasserstein", "min"),
            max_wasserstein=("wasserstein", "max"),
            n_class_pairs=("wasserstein", "size"),
        )
    )

    separability_by_sigma_df["expected_class_pairs"] = expected_pairs
    separability_by_sigma_df["complete_class_set"] = (
        separability_by_sigma_df["n_class_pairs"]
        == expected_pairs
    )

    # This gives every available class pair equal weight.
    separability_by_model_df = (
        pairwise_df
        .groupby("model", as_index=False)
        .agg(
            mean_wasserstein=("wasserstein", "mean"),
            median_wasserstein=("wasserstein", "median"),
            std_wasserstein=("wasserstein", "std"),
            min_wasserstein=("wasserstein", "min"),
            max_wasserstein=("wasserstein", "max"),
            n_pairwise_comparisons=("wasserstein", "size"),
            n_sigmas=("sigma", "nunique"),
        )
    )

    separability_by_sigma_df = separability_by_sigma_df.sort_values(
        ["sigma", "mean_wasserstein"],
        ascending=[True, False],
    ).reset_index(drop=True)

    separability_by_model_df = separability_by_model_df.sort_values(
        "mean_wasserstein",
        ascending=False,
    ).reset_index(drop=True)

    pairwise_df = pairwise_df.sort_values(
        ["sigma", "class_a", "class_b", "model"],
    ).reset_index(drop=True)

    return (
        pairwise_df,
        separability_by_sigma_df,
        separability_by_model_df,
    )


# ============================================================
# 6. Directly compare separability between models
# ============================================================

def compare_model_separability(
    pairwise_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare models using matched class-pair and sigma conditions.

    For each pair of models, the function matches distances using:

        sigma
        class_a
        class_b

    It then computes:

        distance_difference =
            Wasserstein(model_a) - Wasserstein(model_b)

    Positive values mean model_a has greater class separability.

    Returns
    -------
    model_comparison_df:
        One row for each matched sigma/class-pair comparison.

    model_comparison_summary_df:
        Average paired difference for each pair of models.

    Notes
    -----
    This is a descriptive comparison. It does not perform an inferential
    statistical test, because class-pair distances are not necessarily
    independent observations.
    """
    required_cols = {
        "model",
        "sigma",
        "class_a",
        "class_b",
        "wasserstein",
    }

    missing = required_cols - set(pairwise_df.columns)

    if missing:
        raise ValueError(
            f"pairwise_df is missing columns: {sorted(missing)}"
        )

    models = list(pd.unique(pairwise_df["model"]))
    comparison_rows = []

    for model_a, model_b in combinations(models, 2):
        distances_a = (
            pairwise_df.loc[
                pairwise_df["model"] == model_a,
                [
                    "sigma",
                    "class_a",
                    "class_b",
                    "wasserstein",
                ],
            ]
            .rename(
                columns={
                    "wasserstein": "wasserstein_model_a"
                }
            )
        )

        distances_b = (
            pairwise_df.loc[
                pairwise_df["model"] == model_b,
                [
                    "sigma",
                    "class_a",
                    "class_b",
                    "wasserstein",
                ],
            ]
            .rename(
                columns={
                    "wasserstein": "wasserstein_model_b"
                }
            )
        )

        matched = distances_a.merge(
            distances_b,
            on=["sigma", "class_a", "class_b"],
            how="inner",
            validate="one_to_one",
        )

        for _, row in matched.iterrows():
            distance_a = float(row["wasserstein_model_a"])
            distance_b = float(row["wasserstein_model_b"])

            comparison_rows.append({
                "model_a": model_a,
                "model_b": model_b,
                "sigma": row["sigma"],
                "class_a": row["class_a"],
                "class_b": row["class_b"],
                "wasserstein_model_a": distance_a,
                "wasserstein_model_b": distance_b,
                "difference_a_minus_b": distance_a - distance_b,
            })

    model_comparison_df = pd.DataFrame(comparison_rows)

    if model_comparison_df.empty:
        raise ValueError(
            "No matching model comparisons were found. Models must share "
            "at least one sigma and class pair."
        )

    model_comparison_summary_df = (
        model_comparison_df
        .groupby(
            ["model_a", "model_b"],
            as_index=False,
        )
        .agg(
            mean_difference_a_minus_b=(
                "difference_a_minus_b",
                "mean",
            ),
            median_difference_a_minus_b=(
                "difference_a_minus_b",
                "median",
            ),
            std_difference_a_minus_b=(
                "difference_a_minus_b",
                "std",
            ),
            n_matched_comparisons=(
                "difference_a_minus_b",
                "size",
            ),
            proportion_a_greater=(
                "difference_a_minus_b",
                lambda x: float(np.mean(np.asarray(x) > 0)),
            ),
        )
    )

    return (
        model_comparison_df,
        model_comparison_summary_df,
    )


# ============================================================
# 7. Complete wrapper
# ============================================================

def analyze_saved_prototype_separability(
    prototype_dir: str | Path,
    emotions: Sequence[Hashable] | None = None,
    summary_filename: str = "prototype_summary.csv",
    transport_size: tuple[int, int] = (32, 32),
    normalize_coordinates: bool = True,
    clip_negative: bool = True,
    require_complete_class_set: bool = True,
    model_key_index: int = 0,
    sigma_key_index: int = 1,
    class_key_index: int = 2,
    save_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load saved prototypes and perform the full separability analysis.

    Returns
    -------
    results:
        Dictionary containing:

        "pairwise"
        "by_sigma"
        "by_model"
        "model_comparisons"
        "model_comparison_summary"
    """
    prototypes = load_saved_prototypes(
        prototype_dir=prototype_dir,
        summary_filename=summary_filename,
    )

    (
        pairwise_df,
        separability_by_sigma_df,
        separability_by_model_df,
    ) = compute_emotion_separability(
        prototypes=prototypes,
        emotions=emotions,
        transport_size=transport_size,
        normalize_coordinates=normalize_coordinates,
        clip_negative=clip_negative,
        model_key_index=model_key_index,
        sigma_key_index=sigma_key_index,
        class_key_index=class_key_index,
        require_complete_class_set=require_complete_class_set,
    )

    models = pd.unique(pairwise_df["model"])

    if len(models) >= 2:
        (
            model_comparison_df,
            model_comparison_summary_df,
        ) = compare_model_separability(pairwise_df)
    else:
        model_comparison_df = pd.DataFrame()
        model_comparison_summary_df = pd.DataFrame()

    results = {
        "pairwise": pairwise_df,
        "by_sigma": separability_by_sigma_df,
        "by_model": separability_by_model_df,
        "model_comparisons": model_comparison_df,
        "model_comparison_summary": model_comparison_summary_df,
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for result_name, dataframe in results.items():
            dataframe.to_csv(
                save_dir / f"{result_name}.csv",
                index=False,
            )

    return results


def save_separability_results(
    results: dict[str, pd.DataFrame],
    output_dir: str | Path,
    save_csv: bool = True,
    save_pickle: bool = True,
) -> None:
    """
    Save a dictionary of separability-result DataFrames.

    Parameters
    ----------
    results:
        Dictionary mapping result names to pandas DataFrames.

        Example:
            {
                "pairwise": pairwise_df,
                "by_sigma": separability_by_sigma_df,
                "by_model": separability_by_model_df,
                "model_comparisons": model_comparison_df,
                "model_comparison_summary": model_comparison_summary_df,
            }

    output_dir:
        Directory where results will be saved.

    save_csv:
        If True, save each DataFrame as a separate CSV file.

    save_pickle:
        If True, save the entire results dictionary as one pickle file.

        Pickle preserves pandas dtypes more reliably than CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "result_names": [],
        "csv_files": {},
        "pickle_file": None,
    }

    for result_name, dataframe in results.items():
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Result {result_name!r} must be a pandas DataFrame, "
                f"got {type(dataframe).__name__}."
            )

        metadata["result_names"].append(result_name)

        if save_csv:
            csv_filename = f"{result_name}.csv"
            csv_path = output_dir / csv_filename

            dataframe.to_csv(
                csv_path,
                index=False,
            )

            metadata["csv_files"][result_name] = csv_filename

    if save_pickle:
        pickle_filename = "separability_results.pkl"
        pickle_path = output_dir / pickle_filename

        pd.to_pickle(
            results,
            pickle_path,
        )

        metadata["pickle_file"] = pickle_filename

    metadata_path = output_dir / "separability_results_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(f"Saved separability results to: {output_dir}")


if __name__ == "__main__":

    results = analyze_saved_prototype_separability(
        prototype_dir="attention_prototypes/NATIVE_RESOLUTION/true_label_correct",
        emotions=EMOTIONS,

        # Exact optimal transport is computed on 32 x 32 maps.
        transport_size=(14, 14),

        # Distances are expressed relative to normalized image coordinates.
        normalize_coordinates=True,

        # Analyze positive relevance allocation.
        clip_negative=True,

        # Require all seven emotion prototypes for every model/sigma.
        require_complete_class_set=True,

        # Assumes keys are: (model, sigma, emotion)
        model_key_index=0,
        sigma_key_index=1,
        class_key_index=2,

        save_dir="attention_prototypes/separability_results",
    )

    print("\nPairwise class distances:")
    print(results["pairwise"].to_string(index=False))

    print("\nSeparability by model and sigma:")
    print(results["by_sigma"].to_string(index=False))

    print("\nOverall separability by model:")
    print(results["by_model"].to_string(index=False))

    if not results["model_comparison_summary"].empty:
        print("\nPaired comparison between models:")
        print(
            results["model_comparison_summary"].to_string(
                index=False
            )
        )

    save_separability_results(
        results=results,
        output_dir="attention_prototypes/separability_results",
        save_csv=True,
        save_pickle=True,
    )

    print('done')
