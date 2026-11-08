"""
29/6/26
"""

from __future__ import annotations

from pathlib import Path
from typing import Hashable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from create_attn_maps import EMOTIONS


def plot_emotion_pair_separability(
    pairwise_df: pd.DataFrame,
    models: Sequence[Hashable] | None = None,
    sigmas: Sequence[Hashable] | None = None,
    emotions: Sequence[Hashable] | None = None,
    value_col: str = "wasserstein",
    shared_scale: bool = True,
    annotate: bool = True,
    annotation_decimals: int = 3,
    output_dir: str | Path | None = None,
    filename_prefix: str = "emotion_separability",
    show: bool = True,
    mask_bottom: bool = True,
) -> dict[tuple[Hashable, Hashable], plt.Figure]:
    """
    Plot pairwise emotion separability as a heatmap for each model and sigma.

    One separate figure is created for every selected model/sigma condition.

    Parameters
    ----------
    pairwise_df:
        Pairwise separability dataframe returned by
        compute_emotion_separability().

        Required columns:
            model
            sigma
            class_a
            class_b
            wasserstein

    models:
        Models to plot.

        Example:
            ["mixed_pretrain", "lowres_pretrain"]

        If None, all models are plotted.

    sigmas:
        Sigma conditions to plot.

        Example:
            [8]

        If None, all sigmas are plotted.

    emotions:
        Emotion classes and their display order.

        Example:
            [
                "anger",
                "disgust",
                "fear",
                "happy",
                "sad",
                "surprise",
                "neutral",
            ]

        If None, all classes found in class_a and class_b are used.

    value_col:
        Column containing the separability metric.

        Default:
            "wasserstein"

    shared_scale:
        If True, all figures use the same minimum and maximum values.
        Recommended when comparing models.

    annotate:
        If True, print the numeric Wasserstein distance in each cell.

    annotation_decimals:
        Number of decimal places used in cell annotations.

    output_dir:
        Optional output directory. If provided, every figure is saved.

    filename_prefix:
        Prefix used for saved figure names.

    show:
        If True, display figures interactively.

    Returns
    -------
    figures:
        Dictionary mapping:

            (model, sigma) -> matplotlib Figure
    """
    required_cols = {
        "model",
        "sigma",
        "class_a",
        "class_b",
        value_col,
    }

    missing_cols = required_cols - set(pairwise_df.columns)

    if missing_cols:
        raise ValueError(
            f"pairwise_df is missing required columns: "
            f"{sorted(missing_cols)}"
        )

    work = pairwise_df.copy()

    if models is None:
        models = list(pd.unique(work["model"]))
    else:
        models = list(models)

    if sigmas is None:
        sigmas = list(pd.unique(work["sigma"]))
    else:
        sigmas = list(sigmas)

    if emotions is None:
        emotions = list(
            pd.unique(
                pd.concat(
                    [
                        work["class_a"],
                        work["class_b"],
                    ],
                    ignore_index=True,
                )
            )
        )
    else:
        emotions = list(emotions)

    if len(models) == 0:
        raise ValueError("No models were selected.")

    if len(sigmas) == 0:
        raise ValueError("No sigma conditions were selected.")

    if len(emotions) < 2:
        raise ValueError(
            "At least two emotions are required."
        )

    work = work[
        work["model"].isin(models)
        & work["sigma"].isin(sigmas)
        & work["class_a"].isin(emotions)
        & work["class_b"].isin(emotions)
    ].copy()

    if work.empty:
        raise ValueError(
            "No rows remain after filtering by models, sigmas, "
            "and emotions."
        )

    available_models = set(work["model"])
    missing_models = [
        model for model in models
        if model not in available_models
    ]

    if missing_models:
        print(
            "Warning: no data were found for models: "
            f"{missing_models}"
        )

    available_sigmas = set(work["sigma"])
    missing_sigmas = [
        sigma for sigma in sigmas
        if sigma not in available_sigmas
    ]

    if missing_sigmas:
        print(
            "Warning: no data were found for sigmas: "
            f"{missing_sigmas}"
        )

    # A common scale is important for visual comparison across models.
    if shared_scale:
        global_vmin = 0.0
        global_vmax = float(work[value_col].max())

        if not np.isfinite(global_vmax):
            raise ValueError(
                f"Column {value_col!r} contains non-finite values."
            )

        if global_vmax <= global_vmin:
            global_vmax = global_vmin + 1e-12
    else:
        global_vmin = None
        global_vmax = None

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    figures = {}

    for model in models:
        for sigma in sigmas:
            condition_df = work[
                (work["model"] == model)
                & (work["sigma"] == sigma)
            ]

            if condition_df.empty:
                continue

            distance_matrix = create_emotion_distance_matrix(
                condition_df=condition_df,
                emotions=emotions,
                value_col=value_col,
            )

            if shared_scale:
                vmin = global_vmin
                vmax = global_vmax
            else:
                finite_values = distance_matrix.to_numpy()
                finite_values = finite_values[
                    np.isfinite(finite_values)
                ]

                vmin = 0.0

                if finite_values.size == 0:
                    vmax = 1.0
                else:
                    vmax = float(finite_values.max())

                if vmax <= vmin:
                    vmax = vmin + 1e-12

            fig, ax = plt.subplots(
                figsize=(
                    max(6, len(emotions) * 1.1),
                    max(5, len(emotions) * 0.95),
                )
            )

            # Mask lower triangle and diagonal, leaving only upper triangle.
            mask = np.tril(
                np.ones_like(distance_matrix, dtype=bool),
                k=0,
            )

            masked_matrix = np.ma.array(
                distance_matrix,
                mask=mask,
            )
            if mask_bottom:
                image = ax.imshow(
                    masked_matrix,
                    vmin=vmin,
                    vmax=vmax,
                )
            else:
                image = ax.imshow(
                    distance_matrix.to_numpy(),
                    vmin=vmin,
                    vmax=vmax,
                )

            ax.set_xticks(np.arange(len(emotions)))
            ax.set_yticks(np.arange(len(emotions)))

            ax.set_xticklabels(
                emotions,
                rotation=45,
                ha="right",
            )
            ax.set_yticklabels(emotions)

            ax.set_xlabel("Emotion")
            ax.set_ylabel("Emotion")

            ax.set_title(
                f"Emotion-prototype separability\n"
                f"Model: {model} | sigma: {sigma}"
            )

            colorbar = fig.colorbar(image, ax=ax)
            colorbar.set_label(
                "2D Wasserstein distance"
            )

            if annotate:
                if mask_bottom:
                    _annotate_distance_matrix(
                        ax=ax,
                        matrix=masked_matrix,
                        decimals=annotation_decimals,
                    )
                else:
                    _annotate_distance_matrix(
                        ax=ax,
                        matrix=distance_matrix.to_numpy(),
                        decimals=annotation_decimals,
                    )

            # Draw minor-grid boundaries between cells.
            ax.set_xticks(
                np.arange(-0.5, len(emotions), 1),
                minor=True,
            )
            ax.set_yticks(
                np.arange(-0.5, len(emotions), 1),
                minor=True,
            )
            ax.grid(
                which="minor",
                linewidth=0.5,
            )
            ax.tick_params(
                which="minor",
                bottom=False,
                left=False,
            )

            fig.tight_layout()

            if output_dir is not None:
                safe_model = _make_filename_safe(model)
                safe_sigma = _make_filename_safe(sigma)

                if mask_bottom:
                    output_path = (
                            output_dir
                            / (
                                f"{filename_prefix}"
                                f"__model-{safe_model}"
                                f"__sigma-{safe_sigma}_mask_bottom.png"
                            )
                    )
                else:
                    output_path = (
                        output_dir
                        / (
                            f"{filename_prefix}"
                            f"__model-{safe_model}"
                            f"__sigma-{safe_sigma}.png"
                        )
                    )

                fig.savefig(
                    output_path,
                    dpi=300,
                    bbox_inches="tight",
                )

            figures[(model, sigma)] = fig

            if show:
                plt.show()
            else:
                plt.close(fig)

    return figures


def create_emotion_distance_matrix(
    condition_df: pd.DataFrame,
    emotions: Sequence[Hashable],
    value_col: str = "wasserstein",
) -> pd.DataFrame:
    """
    Convert long-form pairwise distances into a symmetric matrix.

    The diagonal is set to zero because the distance between an emotion
    prototype and itself is zero.
    """
    emotions = list(emotions)

    matrix = pd.DataFrame(
        np.nan,
        index=emotions,
        columns=emotions,
        dtype=float,
    )

    np.fill_diagonal(matrix.values, 0.0)

    duplicated_pairs = condition_df.duplicated(
        subset=["class_a", "class_b"],
        keep=False,
    )

    if duplicated_pairs.any():
        duplicate_rows = condition_df.loc[
            duplicated_pairs,
            ["class_a", "class_b", value_col],
        ]

        raise ValueError(
            "Multiple distances were found for the same emotion pair "
            "within one model/sigma condition:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    for _, row in condition_df.iterrows():
        class_a = row["class_a"]
        class_b = row["class_b"]
        value = float(row[value_col])

        if not np.isfinite(value):
            raise ValueError(
                f"Non-finite distance for pair "
                f"{class_a!r}, {class_b!r}."
            )

        matrix.loc[class_a, class_b] = value
        matrix.loc[class_b, class_a] = value

    return matrix


def _annotate_distance_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    decimals: int = 3,
) -> None:
    """
    Add numeric labels to finite heatmap cells.

    Text brightness is selected according to the normalized cell value.
    """
    finite_values = matrix[np.isfinite(matrix)]

    if finite_values.size == 0:
        return

    min_value = float(finite_values.min())
    max_value = float(finite_values.max())
    value_range = max_value - min_value

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]

            if not np.isfinite(value):
                continue

            if value_range > 0:
                normalized_value = (
                    value - min_value
                ) / value_range
            else:
                normalized_value = 0.0

            text_color = (
                "white"
                if normalized_value > 0.55
                else "black"
            )

            ax.text(
                col_idx,
                row_idx,
                f"{value:.{decimals}f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )


def _make_filename_safe(value: object) -> str:
    """
    Convert a model name or sigma value into a safe filename component.
    """
    text = str(value)

    for character in [
        " ",
        "/",
        "\\",
        ":",
        ";",
        ",",
        "(",
        ")",
        "[",
        "]",
    ]:
        text = text.replace(character, "_")

    return text


def plot_emotion_separability_difference(
    pairwise_df: pd.DataFrame,
    model_a: Hashable,
    model_b: Hashable,
    sigma: Hashable,
    emotions: Sequence[Hashable] | None = None,
    value_col: str = "wasserstein",
    annotate: bool = True,
    annotation_decimals: int = 3,
    output_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, pd.DataFrame]:
    """
    Plot the difference in emotion-pair separability between two models.

    Each cell is calculated as:

        difference = separability(model_a) - separability(model_b)

    Interpretation
    --------------
    Positive value:
        model_a has greater separability than model_b for that emotion pair.

    Negative value:
        model_b has greater separability than model_a for that emotion pair.

    Zero:
        the two models have equal separability for that emotion pair.

    Parameters
    ----------
    pairwise_df:
        DataFrame returned by compute_emotion_separability().

        Required columns:
            model
            sigma
            class_a
            class_b
            wasserstein

    model_a:
        First model. Its values are the minuend.

    model_b:
        Second model. Its values are subtracted from model_a.

    sigma:
        Sigma condition to compare.

    emotions:
        Emotion classes and display order.

        If None, all emotions available for the chosen models and sigma
        are used.

    value_col:
        Separability metric column.

        Default:
            "wasserstein"

    annotate:
        If True, show the numeric difference in each cell.

    annotation_decimals:
        Number of decimal places used in annotations.

    output_path:
        Optional path for saving the figure.

    show:
        If True, display the figure.

    Returns
    -------
    fig:
        Matplotlib figure.

    difference_matrix:
        Symmetric dataframe containing:

            model_a - model_b
    """
    required_cols = {
        "model",
        "sigma",
        "class_a",
        "class_b",
        value_col,
    }

    missing_cols = required_cols - set(pairwise_df.columns)

    if missing_cols:
        raise ValueError(
            f"pairwise_df is missing required columns: "
            f"{sorted(missing_cols)}"
        )

    condition_df = pairwise_df[
        (pairwise_df["sigma"] == sigma)
        & pairwise_df["model"].isin([model_a, model_b])
    ].copy()

    if condition_df.empty:
        raise ValueError(
            f"No rows found for sigma={sigma!r} and models "
            f"{model_a!r}, {model_b!r}."
        )

    available_models = set(condition_df["model"])

    if model_a not in available_models:
        raise ValueError(
            f"No data found for model_a={model_a!r} at sigma={sigma!r}."
        )

    if model_b not in available_models:
        raise ValueError(
            f"No data found for model_b={model_b!r} at sigma={sigma!r}."
        )

    if emotions is None:
        emotions = list(
            pd.unique(
                pd.concat(
                    [
                        condition_df["class_a"],
                        condition_df["class_b"],
                    ],
                    ignore_index=True,
                )
            )
        )
    else:
        emotions = list(emotions)

    if len(emotions) < 2:
        raise ValueError(
            "At least two emotions are required."
        )

    model_a_df = _canonicalize_emotion_pairs(
        condition_df[condition_df["model"] == model_a],
        emotions=emotions,
        value_col=value_col,
    ).rename(
        columns={value_col: "separability_model_a"}
    )

    model_b_df = _canonicalize_emotion_pairs(
        condition_df[condition_df["model"] == model_b],
        emotions=emotions,
        value_col=value_col,
    ).rename(
        columns={value_col: "separability_model_b"}
    )

    matched_df = model_a_df.merge(
        model_b_df,
        on=["class_1", "class_2"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = matched_df[matched_df["_merge"] != "both"]

    if not unmatched.empty:
        raise ValueError(
            "The two models do not contain matching emotion pairs.\n"
            f"{unmatched[['class_1', 'class_2', '_merge']].to_string(index=False)}"
        )

    matched_df["difference"] = (
        matched_df["separability_model_a"]
        - matched_df["separability_model_b"]
    )

    difference_matrix = pd.DataFrame(
        np.nan,
        index=emotions,
        columns=emotions,
        dtype=float,
    )

    # The diagonal is zero because each emotion is identical to itself.
    np.fill_diagonal(difference_matrix.values, 0.0)

    for _, row in matched_df.iterrows():
        class_1 = row["class_1"]
        class_2 = row["class_2"]
        difference = float(row["difference"])

        difference_matrix.loc[class_1, class_2] = difference
        difference_matrix.loc[class_2, class_1] = difference

    finite_values = difference_matrix.to_numpy()
    finite_values = finite_values[np.isfinite(finite_values)]

    if finite_values.size == 0:
        raise ValueError(
            "No finite model-difference values were found."
        )

    # Symmetric limits ensure zero is always the center of the color scale.
    max_abs_difference = float(np.max(np.abs(finite_values)))

    if max_abs_difference == 0:
        max_abs_difference = 1e-12

    fig, ax = plt.subplots(
        figsize=(
            max(7, len(emotions) * 1.15),
            max(6, len(emotions)),
        )
    )

    matrix_values = difference_matrix.to_numpy()

    # Mask lower triangle and diagonal, leaving only upper triangle.
    mask = np.tril(
        np.ones_like(matrix_values, dtype=bool),
        k=0,
    )

    masked_matrix = np.ma.array(
        matrix_values,
        mask=mask,
    )

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")

    image = ax.imshow(
        masked_matrix,
        cmap=cmap,
        vmin=-max_abs_difference,
        vmax=max_abs_difference,
    )

    ax.set_xticks(np.arange(len(emotions)))
    ax.set_yticks(np.arange(len(emotions)))

    ax.set_xticklabels(
        emotions,
        rotation=45,
        ha="right",
    )
    ax.set_yticklabels(emotions)

    ax.set_xlabel("Emotion")
    ax.set_ylabel("Emotion")

    ax.set_title(
        "Difference in emotion-prototype separability\n"
        f"{model_a} − {model_b} | sigma={sigma}\n"
        # f"Positive: {model_a} more separable; "
        # f"Negative: {model_b} more separable"
    )

    colorbar = fig.colorbar(image, ax=ax)

    colorbar.set_label(
        f"Difference in {value_col}\n"
        f"{model_a} − {model_b}"
    )

    if annotate:
        _annotate_signed_matrix(
            ax=ax,
            matrix=masked_matrix,
            max_abs_value=max_abs_difference,
            decimals=annotation_decimals,
        )

    ax.set_xticks(
        np.arange(-0.5, len(emotions), 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, len(emotions), 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        linewidth=0.5,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, difference_matrix


def _canonicalize_emotion_pairs(
    dataframe: pd.DataFrame,
    emotions: Sequence[Hashable],
    value_col: str,
) -> pd.DataFrame:
    """
    Convert emotion pairs into a consistent order.

    For example:
        fear-surprise
        surprise-fear

    are treated as the same pair.
    """
    emotion_order = {
        emotion: index
        for index, emotion in enumerate(emotions)
    }

    rows = []

    for _, row in dataframe.iterrows():
        class_a = row["class_a"]
        class_b = row["class_b"]

        if class_a not in emotion_order:
            continue

        if class_b not in emotion_order:
            continue

        if emotion_order[class_a] <= emotion_order[class_b]:
            class_1 = class_a
            class_2 = class_b
        else:
            class_1 = class_b
            class_2 = class_a

        rows.append({
            "class_1": class_1,
            "class_2": class_2,
            value_col: float(row[value_col]),
        })

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(
            "No emotion pairs remain after filtering."
        )

    duplicated = result.duplicated(
        subset=["class_1", "class_2"],
        keep=False,
    )

    if duplicated.any():
        duplicate_rows = result.loc[
            duplicated,
            ["class_1", "class_2", value_col],
        ]

        raise ValueError(
            "Multiple values were found for the same emotion pair:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    return result


def _annotate_signed_matrix(
    ax: plt.Axes,
    matrix: np.ndarray | np.ma.MaskedArray,
    max_abs_value: float,
    decimals: int = 3,
) -> None:
    """
    Annotate the visible cells of a signed heatmap.

    Positive values are displayed with an explicit '+' sign.
    Masked cells are ignored.
    """
    matrix = np.ma.asarray(matrix)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):

            if np.ma.is_masked(matrix[row_idx, col_idx]):
                continue

            value = float(matrix[row_idx, col_idx])

            if not np.isfinite(value):
                continue

            relative_magnitude = (
                abs(value) / max_abs_value
                if max_abs_value > 0
                else 0
            )

            text_color = (
                "white"
                if relative_magnitude > 0.55
                else "black"
            )

            ax.text(
                col_idx,
                row_idx,
                f"{value:+.{decimals}f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )


def load_separability_results(
    input_dir: str | Path,
    prefer_pickle: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Load previously saved separability results.

    Parameters
    ----------
    input_dir:
        Directory created by save_separability_results().

    prefer_pickle:
        If True and the pickle file exists, load the entire results
        dictionary from pickle.

        If False, or if the pickle file is missing, load the individual
        CSV files listed in the metadata JSON.

    Returns
    -------
    results:
        Dictionary mapping result names to pandas DataFrames.
    """
    input_dir = Path(input_dir)

    metadata_path = input_dir / "separability_results_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not find metadata file: {metadata_path}"
        )

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    pickle_filename = metadata.get("pickle_file")

    if prefer_pickle and pickle_filename is not None:
        pickle_path = input_dir / pickle_filename

        if pickle_path.exists():
            results = pd.read_pickle(pickle_path)

            if not isinstance(results, dict):
                raise TypeError(
                    f"Expected pickle file to contain a dictionary, "
                    f"got {type(results).__name__}."
                )

            return results

    results = {}

    csv_files = metadata.get("csv_files", {})

    for result_name in metadata.get("result_names", []):
        csv_filename = csv_files.get(result_name)

        if csv_filename is None:
            results[result_name] = pd.DataFrame()
            continue

        csv_path = input_dir / csv_filename

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Could not find CSV file for {result_name!r}: "
                f"{csv_path}"
            )

        results[result_name] = pd.read_csv(csv_path)

    return results


if __name__ == '__main__':
    results = load_separability_results("attention_prototypes/separability_results")
    figures = plot_emotion_pair_separability(
        pairwise_df=results["pairwise"],
        models=[
            "RAF_blur8_pretrained0-8_freeze_fix_projs",
            "RAF_blur8_freeze",
        ],
        sigmas=[8],
        emotions=EMOTIONS,
        shared_scale=True,
        annotate=True,
        annotation_decimals=3,
        output_dir=(
            "attention_prototypes/"
            "separability_results/figures"
        ),
    )

    fig, difference_matrix = plot_emotion_separability_difference(
        pairwise_df=results["pairwise"],
        model_a="RAF_blur8_pretrained0-8_freeze_fix_projs",
        model_b="RAF_blur8_freeze",
        sigma=8,
        emotions=EMOTIONS,
        annotate=True,
        annotation_decimals=3,
        output_path=(
            "attention_prototypes/separability_results/figures/"
            "separability_difference__mixed-minus-lowres__sigma-8.png"
        ),
    )

    print('done')
