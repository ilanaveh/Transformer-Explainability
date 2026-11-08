"""
4/7/26
Assumes matched_balanced_scores_df & emotion_tests_df were created by comp_image_separability.py and saved in pickle in:
attention_maps/emotion_separability_tests/comp_image_separability_data/{img_subset}/comp_image_separability_results_per_emotion.pkl,
where img_subset is ALL_PRED (all images, predicted class) / ALL_TRUE (all images, true class) / CORRECT_PRED (only
images correctly classified by both models, predicted class - which is the same as true class).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Hashable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from create_attn_maps import EMOTIONS


def _safe_filename_component(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_") or "unnamed"


def plot_separability_by_emotion_bars(
    matched_scores_df: pd.DataFrame,
    emotion_tests_df: pd.DataFrame,
    model_a: Hashable,
    model_b: Hashable,
    emotions: Sequence[Hashable] | None = None,
    model_labels: dict[Hashable, str] | None = None,
    score_a_col: str = "score_model_a",
    score_b_col: str = "score_model_b",
    corrected_p_col: str = "p_value_corrected",
    ylabel: str = "Separability",
    estimator: str = "mean",
    error_type: str = "sem",
    bar_width: float = 0.65,
    output_dir: str | Path | None = None,
    filename_prefix: str = "emotion_separability_barplot",
    dpi: int = 300,
    show: bool = True,
) -> dict[Hashable, plt.Figure]:
    """
    Plot one bar plot per emotion, comparing model separability.

    Parameters
    ----------
    matched_scores_df:
        Output from paired_ttests_by_emotion_class_balanced().

        Required columns:
            image_id
            true_label
            score_model_a
            score_model_b

    emotion_tests_df:
        Emotion-specific test table.

        Required columns:
            emotion
            p_value_corrected

    model_a, model_b:
        Model identifiers.

    emotions:
        Emotions to plot and their order.

    model_labels:
        Optional display names for the two models.

    score_a_col, score_b_col:
        Columns containing model A and model B scores.

    corrected_p_col:
        Column containing corrected p-values.

    estimator:
        "mean" or "median"

    error_type:
        "sem", "std", or "none"

    Returns
    -------
    figures:
        Dictionary mapping emotion -> figure.
    """
    required_score_cols = {
        "true_label",
        score_a_col,
        score_b_col,
    }
    missing_score_cols = required_score_cols - set(matched_scores_df.columns)
    if missing_score_cols:
        raise ValueError(
            "matched_scores_df is missing required columns: "
            f"{sorted(missing_score_cols)}"
        )

    required_test_cols = {"emotion", corrected_p_col}
    missing_test_cols = required_test_cols - set(emotion_tests_df.columns)
    if missing_test_cols:
        raise ValueError(
            "emotion_tests_df is missing required columns: "
            f"{sorted(missing_test_cols)}"
        )

    if estimator not in {"mean", "median"}:
        raise ValueError("estimator must be 'mean' or 'median'.")

    if error_type not in {"sem", "std", "none"}:
        raise ValueError("error_type must be 'sem', 'std', or 'none'.")

    if emotions is None:
        emotions = list(pd.unique(matched_scores_df["true_label"]))
    else:
        emotions = list(emotions)

    if model_labels is None:
        model_labels = {
            model_a: str(model_a),
            model_b: str(model_b),
        }

    display_label_a = model_labels.get(model_a, str(model_a))
    display_label_b = model_labels.get(model_b, str(model_b))

    # Requested colors, normalized to matplotlib scale [0, 1]
    color_a = np.array([128, 201, 128]) / 255.0
    color_b = np.array([190, 173, 212]) / 255.0

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    figures = {}

    for emotion in emotions:
        emotion_scores = (
            matched_scores_df.loc[
                matched_scores_df["true_label"] == emotion,
                [score_a_col, score_b_col],
            ]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .reset_index(drop=True)
        )

        if emotion_scores.empty:
            print(f"Skipping {emotion!r}: no valid matched scores.")
            continue

        test_rows = emotion_tests_df.loc[
            emotion_tests_df["emotion"].astype(str) == str(emotion)
        ]

        if len(test_rows) > 1:
            raise ValueError(
                f"Multiple rows found in emotion_tests_df for emotion {emotion!r}."
            )

        corrected_p = np.nan
        if len(test_rows) == 1 and pd.notna(test_rows.iloc[0][corrected_p_col]):
            corrected_p = float(test_rows.iloc[0][corrected_p_col])

        scores_a = emotion_scores[score_a_col].to_numpy(dtype=float)
        scores_b = emotion_scores[score_b_col].to_numpy(dtype=float)

        n_images = len(emotion_scores)

        if estimator == "mean":
            center_a = float(np.mean(scores_a))
            center_b = float(np.mean(scores_b))
        else:
            center_a = float(np.median(scores_a))
            center_b = float(np.median(scores_b))

        if error_type == "sem":
            error_a = (
                float(np.std(scores_a, ddof=1) / np.sqrt(n_images))
                if n_images > 1
                else 0.0
            )
            error_b = (
                float(np.std(scores_b, ddof=1) / np.sqrt(n_images))
                if n_images > 1
                else 0.0
            )
        elif error_type == "std":
            error_a = float(np.std(scores_a, ddof=1)) if n_images > 1 else 0.0
            error_b = float(np.std(scores_b, ddof=1)) if n_images > 1 else 0.0
        else:
            error_a = 0.0
            error_b = 0.0

        fig, ax = plt.subplots(figsize=(3, 3))

        x_positions = np.array([0, 1], dtype=float)
        heights = [center_a, center_b]
        errors = [error_a, error_b]

        ax.bar(
            x_positions[0],
            heights[0],
            width=bar_width,
            yerr=None if error_type == "none" else errors[0],
            capsize=3 if error_type != "none" else 0,
            color=color_a,
            edgecolor="black",
            linewidth=.5,
        )

        ax.bar(
            x_positions[1],
            heights[1],
            width=bar_width,
            yerr=None if error_type == "none" else errors[1],
            capsize=6 if error_type != "none" else 0,
            color=color_b,
            edgecolor="black",
            linewidth=1.0,
        )

        # Significance stars based on corrected p-value
        stars = ""
        if np.isfinite(corrected_p):
            if corrected_p < 0.001:
                stars = "***"
            elif corrected_p < 0.01:
                stars = "**"
            elif corrected_p < 0.05:
                stars = "*"

        score_min = min(np.min(scores_a), np.min(scores_b), 0.0)
        score_max = max(np.max(scores_a), np.max(scores_b), heights[0], heights[1])

        if error_type != "none":
            score_max = max(score_max, heights[0] + errors[0], heights[1] + errors[1])

        score_range = score_max - score_min
        if score_range <= 0:
            score_range = max(abs(score_max), 1.0) * 0.2

        y_lower = -.02 if 'CORRECT' in output_dir.parent.name else -.002
        y_upper = .06 if 'CORRECT' in output_dir.parent.name else .015

        if stars:
            bracket_y = (y_upper - .01) if 'CORRECT' in output_dir.parent.name else (y_upper - .002)
            bracket_h = 0.001  # .04 * score_range

            ax.plot(
                [x_positions[0], x_positions[0], x_positions[1], x_positions[1]],
                [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
                color="black",
                linewidth=1.5,
                clip_on=False,
            )

            ax.text(
                np.mean(x_positions),
                bracket_y + bracket_h,
                stars,
                ha="center",
                va="bottom",
                color="black",
                fontsize=8,
                # fontweight="bold",
            )

            # y_upper = bracket_y + bracket_h + 0.15 * score_range
        # else:
        #     y_upper = score_max + 0.15 * score_range

        # y_lower = score_min - 0.08 * score_range
        # if y_lower > 0 and score_min < 0.25 * score_range:
        #     y_lower = 0.0

        ax.set_ylim(y_lower, y_upper)
        tick_dif = .02 if 'CORRECT' in output_dir.parent.name else .01
        ax.set_yticks(np.arange(np.round(y_lower, 2), y_upper+.001, tick_dif))

        ax.axhline(0, linestyle="--", linewidth=1, color="black", alpha=0.4)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([display_label_a, display_label_b])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{emotion} " f"(N={n_images})")

        # ax.spines["top"].set_visible(False)
        # ax.spines["right"].set_visible(False)

        fig.tight_layout()

        if output_dir is not None:
            safe_emotion = _safe_filename_component(emotion)
            output_path = output_dir / f"{filename_prefix}__emotion-{safe_emotion}.png"
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

        figures[emotion] = fig

        if show:
            plt.show()
        else:
            plt.close(fig)

    return figures


img_subset = 'ALL_PRED'  # 'CORRECT_PRED' / 'ALL_PRED' / 'ALL_TRUE'
pickle_path = f"attention_maps/emotion_separability_tests/{img_subset}/comp_image_separability_data/comp_image_separability_results_per_emotion.pkl"
results = pd.read_pickle(pickle_path)


figures = plot_separability_by_emotion_bars(
    matched_scores_df=results['matched_balanced_scores'],
    emotion_tests_df=results['emotion_tests'],
    model_a="mixed_pretrain",
    model_b="lowres_pretrain",
    emotions=EMOTIONS,
    model_labels={
        "mixed_pretrain": "Mixed-resolution",
        "lowres_pretrain": "Low-resolution",
    },
    estimator="mean",
    error_type="sem",
    output_dir=f"attention_maps/emotion_separability_tests/{img_subset}/figures_bars",
    show=False,
)
print('')
