"""
22/7/26
Define "mouth" and "eyes" ROIs, using patch-indexes, and quantify proportion of attention in each ROI for each mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AggregationMode = Literal["pooled", "per_emotion", "balanced"]
ErrorType = Literal["sem", "std", "ci95", "none"]
METADATA_PATH = Path("attention_maps/NATIVE_RESOLUTION/metadata_all_models_native.csv")


def calculate_roi_relevance_proportions(
    metadata: pd.DataFrame,
    rois: Mapping[str, Sequence[int]],
    *,
    model_col: str = "model",
    emotion_col: str = "true_label",
    map_path_col: str = "map_path",
    image_id_col: str | None = "image_id",
    absolute_relevance: bool = False,
    clip_negative: bool = True,
    class_token_position: Literal["none", "first", "last", "auto"] = "auto",
    expected_num_patches: int | None = None,
    normalize_over: Literal["all_patches", "defined_rois"] = "all_patches",
    add_other: bool = True,                 # NEW
    other_name: str = "Other",             # NEW
) -> pd.DataFrame:
    """
    Calculate the proportion of relevance falling inside each ROI for each image.

    For image i and ROI r:

        proportion(i, r) =
            sum(relevance values in ROI r) / sum(relevance values in denominator)

    Parameters
    ----------
    metadata
        One row per image. Must include model, emotion, and map-path columns.

    rois
        Mapping from ROI name to flattened patch indices.

        Example:
            rois = {
                "Eyes": [31, 32, 33, 34, 45, 46, 47, 48],
                "Mouth": [130, 131, 132, 133, 144, 145, 146, 147],
            }

    absolute_relevance
        If True, use abs(relevance). This can be useful when maps contain
        signed relevance and total contribution magnitude is of interest.

    clip_negative
        If True and absolute_relevance=False, negative values are set to zero.
        This is usually appropriate for positive Chefer relevance maps.

    class_token_position
        How to remove a class-token value if it is present.

        "none"  : do not remove anything.
        "first" : remove the first value.
        "last"  : remove the last value.
        "auto"  : infer from expected_num_patches or ROI indices.

    expected_num_patches
        Usually 196 for a 14x14 patch grid. Recommended when maps might include
        a class-token value.

    normalize_over
        "all_patches":
            denominator is relevance over the complete patch map.

        "defined_rois":
            denominator is relevance over the union of all defined ROI patches.
            In this case, ROI proportions sum to one only when ROIs do not
            overlap and together define the complete normalization region.

    Returns
    -------
    pd.DataFrame
        Long-format table with one row per image × ROI.
    """
    required_columns = {model_col, emotion_col, map_path_col}
    missing = required_columns.difference(metadata.columns)

    if missing:
        raise ValueError(
            f"Metadata is missing required columns: {sorted(missing)}"
        )

    if not rois:
        raise ValueError("`rois` must contain at least one ROI.")

    clean_rois: dict[str, np.ndarray] = {}

    for roi_name, indices in rois.items():
        roi_indices = np.asarray(indices, dtype=int).reshape(-1)

        if roi_indices.size == 0:
            raise ValueError(f"ROI {roi_name!r} contains no patch indices.")

        if np.any(roi_indices < 0):
            raise ValueError(
                f"ROI {roi_name!r} contains negative patch indices."
            )

        clean_rois[roi_name] = np.unique(roi_indices)

    maximum_roi_index = max(
        int(indices.max()) for indices in clean_rois.values()
    )

    if expected_num_patches is None:
        inferred_num_patches = maximum_roi_index + 1
    else:
        inferred_num_patches = expected_num_patches

    all_defined_indices = np.unique(
        np.concatenate(list(clean_rois.values()))
    )

    if add_other and other_name in clean_rois:
        raise ValueError(
            f"`other_name={other_name!r}` conflicts with an existing ROI name."
        )

    records: list[dict] = []

    for row_index, row in metadata.iterrows():
        map_path = Path(row[map_path_col])

        if not map_path.exists():
            raise FileNotFoundError(
                f"Relevance map not found for row {row_index}: {map_path}"
            )

        relevance = np.asarray(np.load(map_path), dtype=float).squeeze()

        if relevance.ndim == 2:
            relevance = relevance.reshape(-1)
        elif relevance.ndim != 1:
            raise ValueError(
                f"Expected a 1D or 2D relevance map, but {map_path} "
                f"has shape {relevance.shape}."
            )

        relevance = _remove_class_token_if_needed(
            relevance=relevance,
            class_token_position=class_token_position,
            expected_num_patches=inferred_num_patches,
        )

        if maximum_roi_index >= relevance.size:
            raise IndexError(
                f"ROI patch index {maximum_roi_index} is outside relevance "
                f"map {map_path}, which contains {relevance.size} patches."
            )

        if not np.all(np.isfinite(relevance)):
            raise ValueError(
                f"Relevance map contains non-finite values: {map_path}"
            )

        if add_other:
            all_patch_indices = np.arange(relevance.size)

            other_indices = np.setdiff1d(
                all_patch_indices,
                all_defined_indices,
                assume_unique=True,
            )

        if absolute_relevance:
            relevance = np.abs(relevance)
        elif clip_negative:
            relevance = np.clip(relevance, a_min=0, a_max=None)

        if normalize_over == "all_patches":
            denominator = relevance.sum()
        elif normalize_over == "defined_rois":
            denominator = relevance[all_defined_indices].sum()
        else:
            raise ValueError(
                "`normalize_over` must be 'all_patches' or 'defined_rois'."
            )

        if denominator <= 0:
            # Returning NaN prevents a zero-relevance image from contributing
            # an artificial zero to the group average.
            denominator = np.nan

        common_record = {
            "source_row": row_index,
            "model": row[model_col],
            "emotion": row[emotion_col],
            "map_path": str(map_path),
        }

        if image_id_col is not None and image_id_col in metadata.columns:
            common_record["image_id"] = row[image_id_col]

        # for roi_name, roi_indices in clean_rois.items():
        #     roi_relevance = relevance[roi_indices].sum()
        #     proportion = roi_relevance / denominator
        #
        #     records.append(
        #         {
        #             **common_record,
        #             "roi": roi_name,
        #             "roi_relevance": roi_relevance,
        #             "total_relevance": denominator,
        #             "proportion": proportion,
        #         }
        #     )

        roi_indices_to_save = dict(clean_rois)

        if add_other:
            roi_indices_to_save[other_name] = other_indices

        for roi_name, roi_indices in roi_indices_to_save.items():
            roi_relevance = relevance[roi_indices].sum()
            proportion = roi_relevance / denominator

            records.append(
                {
                    **common_record,
                    "roi": roi_name,
                    "roi_relevance": roi_relevance,
                    "total_relevance": denominator,
                    "proportion": proportion,
                }
            )

    return pd.DataFrame.from_records(records)


def _remove_class_token_if_needed(
    relevance: np.ndarray,
    class_token_position: Literal["none", "first", "last", "auto"],
    expected_num_patches: int,
) -> np.ndarray:
    """Remove an optional class-token entry from a flattened relevance map."""
    if class_token_position == "none":
        return relevance

    if class_token_position == "first":
        return relevance[1:]

    if class_token_position == "last":
        return relevance[:-1]

    if class_token_position != "auto":
        raise ValueError(
            "`class_token_position` must be 'none', 'first', 'last', or 'auto'."
        )

    if relevance.size == expected_num_patches:
        return relevance

    if relevance.size == expected_num_patches + 1:
        # Most ViT representations place CLS before the patch tokens.
        return relevance[1:]

    raise ValueError(
        "Could not infer whether the relevance map contains a class token. "
        f"Expected {expected_num_patches} or {expected_num_patches + 1} "
        f"values, but found {relevance.size}. Set `class_token_position` "
        "explicitly if needed."
    )


def summarize_roi_relevance(
    per_image_df: pd.DataFrame,
    *,
    aggregation: AggregationMode = "pooled",
    error: ErrorType = "sem",
) -> pd.DataFrame:
    """
    Summarize per-image ROI relevance proportions.

    Aggregation modes
    -----------------
    pooled
        Pool all images together. Emotions with more images receive more weight.

    per_emotion
        Produce a separate result for every emotion.

    balanced
        First calculate the mean for each emotion, then average the emotion
        means. Every emotion receives equal weight regardless of sample size.

        Error bars are calculated across emotion-level means, not across images.
    """
    required_columns = {"model", "emotion", "roi", "proportion"}
    missing = required_columns.difference(per_image_df.columns)

    if missing:
        raise ValueError(
            f"`per_image_df` is missing columns: {sorted(missing)}"
        )

    data = per_image_df.dropna(subset=["proportion"]).copy()

    if aggregation == "pooled":
        summary = _summarize_groups(
            data,
            group_columns=["model", "roi"],
            error=error,
        )
        summary["emotion"] = "All images"
        summary["aggregation"] = "pooled"

    elif aggregation == "per_emotion":
        summary = _summarize_groups(
            data,
            group_columns=["model", "emotion", "roi"],
            error=error,
        )
        summary["aggregation"] = "per_emotion"

    elif aggregation == "balanced":
        emotion_means = (
            data.groupby(
                ["model", "emotion", "roi"],
                observed=True,
                as_index=False,
            )["proportion"]
            .mean()
        )

        # Here, each observation supplied to the final mean is one emotion mean.
        summary = _summarize_groups(
            emotion_means,
            group_columns=["model", "roi"],
            error=error,
        )

        image_counts = (
            data.groupby(["model", "roi"], observed=True)
            .size()
            .rename("n_images")
            .reset_index()
        )

        summary = summary.merge(
            image_counts,
            on=["model", "roi"],
            how="left",
        )

        summary = summary.rename(columns={"n": "n_emotions"})
        summary["emotion"] = "Balanced across emotions"
        summary["aggregation"] = "balanced"

    else:
        raise ValueError(
            "`aggregation` must be 'pooled', 'per_emotion', or 'balanced'."
        )

    return summary


def _summarize_groups(
    data: pd.DataFrame,
    *,
    group_columns: list[str],
    error: ErrorType,
) -> pd.DataFrame:
    """Calculate mean and requested error measure for grouped proportions."""
    grouped = data.groupby(group_columns, observed=True)["proportion"]

    summary = grouped.agg(
        mean="mean",
        std="std",
        n="count",
    ).reset_index()

    summary["sem"] = summary["std"] / np.sqrt(summary["n"])

    if error == "sem":
        summary["error"] = summary["sem"]
    elif error == "std":
        summary["error"] = summary["std"]
    elif error == "ci95":
        summary["error"] = 1.96 * summary["sem"]
    elif error == "none":
        summary["error"] = 0.0
    else:
        raise ValueError(
            "`error` must be 'sem', 'std', 'ci95', or 'none'."
        )

    # A group containing one observation has undefined SD/SEM.
    summary["error"] = summary["error"].fillna(0.0)

    return summary


def plot_roi_relevance_bars(
    summary_df: pd.DataFrame,
    *,
    aggregation: AggregationMode,
    model_order: Sequence[str] | None = None,
    roi_order: Sequence[str] | None = None,
    emotion_order: Sequence[str] | None = None,
    model_colors: Mapping[str, Sequence[float] | str] | None = None,
    error_label: str = "SEM",
    multiply_by_100: bool = True,
    figsize_per_roi: float = 1.35,
    minimum_fig_width: float = 7.0,
    title: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot grouped bars for ROI relevance proportions.

    For pooled or balanced aggregation:
        x-axis = ROI
        bars = models

    For per_emotion aggregation:
        one panel per emotion
        x-axis = ROI
        bars = models
    """
    required_columns = {"model", "emotion", "roi", "mean", "error"}
    missing = required_columns.difference(summary_df.columns)

    if missing:
        raise ValueError(
            f"`summary_df` is missing columns: {sorted(missing)}"
        )

    data = summary_df.copy()

    models = (
        list(model_order)
        if model_order is not None
        else list(pd.unique(data["model"]))
    )
    rois = (
        list(roi_order)
        if roi_order is not None
        else list(pd.unique(data["roi"]))
    )

    unknown_models = set(data["model"]).difference(models)
    if unknown_models:
        raise ValueError(
            f"`model_order` does not include models: {sorted(unknown_models)}"
        )

    unknown_rois = set(data["roi"]).difference(rois)
    if unknown_rois:
        raise ValueError(
            f"`roi_order` does not include ROIs: {sorted(unknown_rois)}"
        )

    if model_colors is None:
        color_values = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = {
            model: color_values[i % len(color_values)]
            for i, model in enumerate(models)
        }
    else:
        colors = {
            model: _normalize_matplotlib_color(model_colors[model])
            for model in models
        }

    if aggregation == "per_emotion":
        emotions = (
            list(emotion_order)
            if emotion_order is not None
            else list(pd.unique(data["emotion"]))
        )
    else:
        emotions = [str(data["emotion"].iloc[0])]

    n_panels = len(emotions)
    n_columns = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_columns))

    figure_width = max(
        minimum_fig_width,
        figsize_per_roi * len(rois) * n_columns,
    )
    figure_height = 4.8 * n_rows

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(figure_width, figure_height),
        squeeze=False,
        sharey=True,
    )

    axes_flat = axes.reshape(-1)

    x = np.arange(len(rois))
    group_width = 0.8
    bar_width = group_width / max(len(models), 1)

    scale = 100.0 if multiply_by_100 else 1.0

    for panel_index, emotion in enumerate(emotions):
        ax = axes_flat[panel_index]

        panel_data = data[data["emotion"] == emotion]

        for model_index, model in enumerate(models):
            model_data = (
                panel_data[panel_data["model"] == model]
                .set_index("roi")
                .reindex(rois)
            )

            means = model_data["mean"].to_numpy(dtype=float) * scale
            errors = model_data["error"].to_numpy(dtype=float) * scale

            offsets = (
                model_index - (len(models) - 1) / 2
            ) * bar_width

            ax.bar(
                x + offsets,
                means,
                width=bar_width,
                yerr=errors,
                capsize=3,
                label=model,
                color=colors[model],
                edgecolor="black",
                linewidth=0.6,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(rois, rotation=35, ha="right")
        ax.set_xlabel("ROI")

        if panel_index % n_columns == 0:
            unit = "%" if multiply_by_100 else "Proportion"
            ax.set_ylabel(f"Relevance in ROI ({unit})")

        if aggregation == "per_emotion":
            ax.set_title(str(emotion))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    for unused_index in range(n_panels, len(axes_flat)):
        axes_flat[unused_index].set_visible(False)

    legend_handles, legend_labels = axes_flat[0].get_legend_handles_labels()

    fig.legend(
        legend_handles,
        legend_labels,
        title="Model",
        loc="lower center",
        ncol=max(1, len(models)),
        frameon=False,
    )

    if title is None:
        title_map = {
            "pooled": "ROI relevance pooled across images",
            "per_emotion": "ROI relevance by emotion",
            "balanced": "Class-balanced ROI relevance",
        }
        title = title_map[aggregation]

    fig.suptitle(f"{title}\nError bars: {error_label}")
    fig.subplots_adjust(
        bottom=0.18,
        top=0.86,
        wspace=0.25,
        hspace=0.45,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes


def _normalize_matplotlib_color(
    color: Sequence[float] | str,
) -> Sequence[float] | str:
    """
    Allow RGB colors in either [0, 1] or [0, 255] representation.
    """
    if isinstance(color, str):
        return color

    color_array = np.asarray(color, dtype=float)

    if color_array.shape not in {(3,), (4,)}:
        raise ValueError(
            "A model color must be a Matplotlib color string or an RGB/RGBA "
            "sequence."
        )

    if np.any(color_array > 1):
        color_array = color_array / 255.0

    return tuple(color_array)


metadata = pd.read_csv(METADATA_PATH)

models = [
    "RAF_blur8_freeze",
    "RAF_blur8_pretrained0-8_freeze_fix_projs",
]

metadata = metadata[metadata["model"].isin(models)].copy()

rois = {
    'eyes': [
        58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
        86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109
    ],
    'mouth': [
        143, 144, 145, 146, 147, 148, 149, 150,
        157, 158, 159, 160, 161, 162, 163, 164,
        171, 172, 173, 174, 175, 176, 177, 178,
        185, 186, 187, 188, 189, 190, 191, 192
    ],
    # 'nose': [
    #     115, 116, 117, 118, 119, 120, 121, 122,
    #     129, 130, 131, 132, 133, 134, 135, 136
    # ],
    # 'forehead': [
    #     17, 18, 19, 20, 21, 22, 23, 24,
    #     31, 32, 33, 34, 35, 36, 37, 38,
    #     45, 46, 47, 48, 49, 50, 51, 52
    # ],

}

per_image_roi_df = calculate_roi_relevance_proportions(
    metadata,
    rois,
    model_col="model",
    emotion_col="true_label",
    map_path_col="map_path",
    image_id_col="image_id",
    expected_num_patches=196,
    normalize_over="all_patches",
    add_other=True,
    other_name="Other",
)

check = (
    per_image_roi_df
    .groupby(["source_row", "model"])["proportion"]
    .sum()
)

print(check.describe())
# per_image_roi_df = calculate_roi_relevance_proportions(
#     metadata,
#     rois,
#     model_col="model",
#     emotion_col="true_label",
#     map_path_col="map_path",
#     image_id_col="image_id",
#     expected_num_patches=196,
#     class_token_position="auto",
#     normalize_over="all_patches",
# )

pooled_summary = summarize_roi_relevance(
    per_image_roi_df,
    aggregation="pooled",
    error="sem",
)

# plot_roi_relevance_bars(
#     pooled_summary,
#     aggregation="pooled",
#     model_order=models,
#     roi_order=list(rois),
#     model_colors={
#         models[0]: [128, 201, 128],
#         models[1]: [190, 173, 212],
#     },
#     error_label="SEM across images",
#     output_path="roi_relevance_pooled.png",
# )

roi_order = [*rois.keys(), "Other"]

plot_roi_relevance_bars(
    pooled_summary,
    aggregation="pooled",
    model_order=models,
    roi_order=roi_order,
    model_colors={
        models[0]: [128, 201, 128],
        models[1]: [190, 173, 212],
    },
    error_label="SEM across images",
)

emotion_order = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happy",
    "Sad",
    "Angry",
    "Neutral",
]

emotion_summary = summarize_roi_relevance(
    per_image_roi_df,
    aggregation="per_emotion",
    error="sem",
)

plot_roi_relevance_bars(
    emotion_summary,
    aggregation="per_emotion",
    model_order=models,
    roi_order=list(rois),
    emotion_order=emotion_order,
    model_colors={
        models[0]: [128, 201, 128],
        models[1]: [190, 173, 212],
    },
    error_label="SEM across images",
    output_path="roi_relevance_by_emotion.png",
)
