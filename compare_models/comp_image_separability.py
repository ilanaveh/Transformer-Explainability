"""
3/7/26
Assumes results file was created by 'calc_image_separability.py' and saved in
'attention_maps/image_attention_separability_sigma8_{img_subset}/attention_map_separability_results.pkl', where
img_subset is ALL_PRED (all images, predicted class) / ALL_TRUE (all images, true class) / CORRECT_PRED (only images
correctly classified by both models, predicted class - which is the same as true class).
"""

from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon, ttest_1samp, t
from itertools import combinations
from pathlib import Path

from create_attn_maps import EMOTIONS

save_results = False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. paired ttest across emotions, without accounting for class imbalance (all images have equal contribution)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paired_image_level_separability_test(
    per_image_df: pd.DataFrame,
    model_a: Hashable,
    model_b: Hashable,
    score_col: str = "distance_difference",
    alternative: str = "two-sided",
    run_wilcoxon: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare two models using paired per-image separability scores.

    Parameters
    ----------
    per_image_df:
        DataFrame containing one row per image and model.

        Required columns:
            model
            image_id
            true_label
            score_col

    model_a, model_b:
        Models to compare.

    score_col:
        Image-level separability score.

        Recommended:
            "distance_difference"

        Also possible:
            "distance_ratio"

    alternative:
        "two-sided":
            Tests whether the models differ.

        "greater":
            Tests whether model_a has greater separability than model_b.

        "less":
            Tests whether model_a has lower separability than model_b.

    run_wilcoxon:
        If True, also return a Wilcoxon signed-rank test as a
        nonparametric sensitivity analysis.

    Returns
    -------
    matched_df:
        One row per matched image, including both model scores and their
        paired difference.

    test_summary_df:
        Statistical-test summary.
    """
    required_cols = {
        "model",
        "image_id",
        "true_label",
        score_col,
    }

    missing_cols = required_cols - set(per_image_df.columns)

    if missing_cols:
        raise ValueError(
            f"per_image_df is missing required columns: "
            f"{sorted(missing_cols)}"
        )

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(
            "alternative must be 'two-sided', 'greater', or 'less'."
        )

    model_a_df = (
        per_image_df.loc[
            per_image_df["model"] == model_a,
            ["image_id", "true_label", score_col],
        ]
        .rename(columns={score_col: "score_model_a"})
    )

    model_b_df = (
        per_image_df.loc[
            per_image_df["model"] == model_b,
            ["image_id", "true_label", score_col],
        ]
        .rename(columns={score_col: "score_model_b"})
    )

    matched_df = model_a_df.merge(
        model_b_df,
        on=["image_id", "true_label"],
        how="inner",
        validate="one_to_one",
    )

    matched_df = matched_df.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(
        subset=["score_model_a", "score_model_b"]
    ).reset_index(drop=True)

    if len(matched_df) < 2:
        raise ValueError(
            "At least two matched images with finite scores are required."
        )

    matched_df["difference_a_minus_b"] = (
        matched_df["score_model_a"]
        - matched_df["score_model_b"]
    )

    scores_a = matched_df["score_model_a"].to_numpy(
        dtype=np.float64
    )
    scores_b = matched_df["score_model_b"].to_numpy(
        dtype=np.float64
    )
    differences = matched_df["difference_a_minus_b"].to_numpy(
        dtype=np.float64
    )

    # Paired t-test.
    t_result = ttest_rel(
        scores_a,
        scores_b,
        alternative=alternative,
        nan_policy="omit",
    )

    n_images = len(differences)
    mean_difference = float(np.mean(differences))
    std_difference = float(np.std(differences, ddof=1))
    sem_difference = std_difference / np.sqrt(n_images)

    # Paired standardized effect size: Cohen's dz.
    cohens_dz = (
        mean_difference / std_difference
        if std_difference > 0
        else np.nan
    )

    # Shapiro-Wilk applies to the paired differences, not each model
    # distribution separately.
    if 3 <= n_images <= 5000:
        shapiro_result = shapiro(differences)
        shapiro_statistic = float(shapiro_result.statistic)
        shapiro_p_value = float(shapiro_result.pvalue)
    else:
        shapiro_statistic = np.nan
        shapiro_p_value = np.nan

    summary = {
        "model_a": model_a,
        "model_b": model_b,
        "score": score_col,
        "alternative": alternative,
        "n_matched_images": n_images,
        "mean_model_a": float(np.mean(scores_a)),
        "mean_model_b": float(np.mean(scores_b)),
        "mean_difference_a_minus_b": mean_difference,
        "median_difference_a_minus_b": float(
            np.median(differences)
        ),
        "std_difference": std_difference,
        "sem_difference": sem_difference,
        "t_statistic": float(t_result.statistic),
        "t_test_p_value": float(t_result.pvalue),
        "cohens_dz": float(cohens_dz),
        "shapiro_difference_statistic": shapiro_statistic,
        "shapiro_difference_p_value": shapiro_p_value,
    }

    if run_wilcoxon:
        try:
            wilcoxon_result = wilcoxon(
                scores_a,
                scores_b,
                alternative=alternative,
                zero_method="wilcox",
            )

            summary["wilcoxon_statistic"] = float(
                wilcoxon_result.statistic
            )
            summary["wilcoxon_p_value"] = float(
                wilcoxon_result.pvalue
            )

        except ValueError:
            summary["wilcoxon_statistic"] = np.nan
            summary["wilcoxon_p_value"] = np.nan

    test_summary_df = pd.DataFrame([summary])

    return matched_df, test_summary_df


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. paired ttest across emotions, accounting for class imbalance (images are inversely weighted to class size)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def class_balanced_paired_ttest(
    per_image_df: pd.DataFrame,
    model_a: Hashable,
    model_b: Hashable,
    score_col: str = "distance_difference",
    alternative: str = "two-sided",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare two models while giving every emotion class equal weight.

    Procedure
    ---------
    1. Match the same images across models.
    2. Calculate the paired difference for every image:
           model_a - model_b
    3. Average paired differences separately within each emotion.
    4. Perform a one-sample t-test on the emotion-level mean differences.

    Thus, each emotion contributes one observation to the t-test,
    regardless of its number of images.

    Positive differences indicate greater separability for model_a.

    Parameters
    ----------
    per_image_df:
        DataFrame containing one row per image and model.

        Required columns:
            model
            image_id
            true_label
            score_col

    model_a, model_b:
        Models to compare.

    score_col:
        Recommended:
            "distance_difference"

    alternative:
        "two-sided", "greater", or "less".

    Returns
    -------
    matched_df:
        Matched image-level scores and paired differences.

    class_effects_df:
        One row per emotion, containing its mean paired difference.

    summary_df:
        Class-balanced t-test results.
    """
    required_cols = {
        "model",
        "image_id",
        "true_label",
        score_col,
    }

    missing = required_cols - set(per_image_df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(
            "alternative must be 'two-sided', 'greater', or 'less'."
        )

    model_a_df = (
        per_image_df.loc[
            per_image_df["model"] == model_a,
            ["image_id", "true_label", score_col],
        ]
        .rename(columns={score_col: "score_model_a"})
    )

    model_b_df = (
        per_image_df.loc[
            per_image_df["model"] == model_b,
            ["image_id", "true_label", score_col],
        ]
        .rename(columns={score_col: "score_model_b"})
    )

    matched_df = model_a_df.merge(
        model_b_df,
        on=["image_id", "true_label"],
        how="inner",
        validate="one_to_one",
    )

    matched_df = (
        matched_df
        .replace([np.inf, -np.inf], np.nan)
        .dropna(
            subset=["score_model_a", "score_model_b"]
        )
        .reset_index(drop=True)
    )

    if matched_df.empty:
        raise ValueError(
            "No matched images were found between the models."
        )

    matched_df["difference_a_minus_b"] = (
        matched_df["score_model_a"]
        - matched_df["score_model_b"]
    )

    # One equally weighted observation per emotion.
    class_effects_df = (
        matched_df
        .groupby("true_label", as_index=False)
        .agg(
            mean_model_a=("score_model_a", "mean"),
            mean_model_b=("score_model_b", "mean"),
            mean_difference_a_minus_b=(
                "difference_a_minus_b",
                "mean",
            ),
            std_difference_within_class=(
                "difference_a_minus_b",
                "std",
            ),
            n_images=("image_id", "nunique"),
        )
    )

    class_differences = class_effects_df[
        "mean_difference_a_minus_b"
    ].to_numpy(dtype=np.float64)

    class_differences = class_differences[
        np.isfinite(class_differences)
    ]

    if len(class_differences) < 2:
        raise ValueError(
            "At least two emotion classes are required."
        )

    test_result = ttest_1samp(
        class_differences,
        popmean=0.0,
        alternative=alternative,
    )

    n_classes = len(class_differences)
    mean_difference = float(class_differences.mean())
    std_difference = float(
        class_differences.std(ddof=1)
    )
    sem_difference = (
        std_difference / np.sqrt(n_classes)
    )

    # Standardized effect across emotion classes.
    cohens_d = (
        mean_difference / std_difference
        if std_difference > 0
        else np.nan
    )

    summary_df = pd.DataFrame([{
        "model_a": model_a,
        "model_b": model_b,
        "score": score_col,
        "alternative": alternative,
        "macro_mean_model_a": float(
            class_effects_df["mean_model_a"].mean()
        ),
        "macro_mean_model_b": float(
            class_effects_df["mean_model_b"].mean()
        ),
        "macro_difference_a_minus_b": mean_difference,
        "std_difference_across_emotions": std_difference,
        "sem_difference_across_emotions": sem_difference,
        "t_statistic": float(test_result.statistic),
        "p_value": float(test_result.pvalue),
        "cohens_d_across_emotions": float(cohens_d),
        "n_emotions": n_classes,
        "n_matched_images": matched_df["image_id"].nunique(),
    }])

    return matched_df, class_effects_df, summary_df

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3. paired ttest separately for each emotion-pair
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def adjust_pvalues(
    p_values: np.ndarray,
    method: str = "holm",
) -> np.ndarray:
    """
    Correct p-values without statsmodels.

    Supported:
        "holm"
        "bonferroni"
        "fdr_bh"
        "none"
    """
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)

    valid_mask = np.isfinite(p_values)
    valid_p = p_values[valid_mask]

    if valid_p.size == 0:
        return adjusted

    method = method.lower()
    n_tests = len(valid_p)

    if method == "none":
        adjusted_valid = valid_p.copy()

    elif method == "bonferroni":
        adjusted_valid = np.minimum(
            valid_p * n_tests,
            1.0,
        )

    elif method == "holm":
        order = np.argsort(valid_p)
        sorted_p = valid_p[order]

        sorted_adjusted = np.empty(
            n_tests,
            dtype=float,
        )

        running_max = 0.0

        for rank, p_value in enumerate(sorted_p):
            corrected = (
                n_tests - rank
            ) * p_value

            running_max = max(
                running_max,
                corrected,
            )

            sorted_adjusted[rank] = min(
                running_max,
                1.0,
            )

        adjusted_valid = np.empty(
            n_tests,
            dtype=float,
        )
        adjusted_valid[order] = sorted_adjusted

    elif method == "fdr_bh":
        order = np.argsort(valid_p)
        sorted_p = valid_p[order]

        sorted_adjusted = np.empty(
            n_tests,
            dtype=float,
        )

        running_min = 1.0

        for rank in range(
            n_tests - 1,
            -1,
            -1,
        ):
            corrected = (
                sorted_p[rank]
                * n_tests
                / (rank + 1)
            )

            running_min = min(
                running_min,
                corrected,
            )

            sorted_adjusted[rank] = min(
                running_min,
                1.0,
            )

        adjusted_valid = np.empty(
            n_tests,
            dtype=float,
        )
        adjusted_valid[order] = sorted_adjusted

    else:
        raise ValueError(
            "method must be 'holm', 'bonferroni', "
            "'fdr_bh', or 'none'."
        )

    adjusted[valid_mask] = adjusted_valid

    return adjusted


def make_directed_pairwise_distances(
    pairwise_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert each unordered image pair into two directed rows.

    Original:
        image A -- image B

    Directed:
        anchor=A, comparison=B
        anchor=B, comparison=A
    """
    required_cols = {
        "model",
        "image_id_a",
        "image_id_b",
        "label_a",
        "label_b",
        "wasserstein",
    }

    missing = required_cols - set(
        pairwise_df.columns
    )

    if missing:
        raise ValueError(
            f"pairwise_df is missing columns: "
            f"{sorted(missing)}"
        )

    side_a = (
        pairwise_df
        .rename(
            columns={
                "image_id_a": "image_id",
                "image_id_b": "comparison_image_id",
                "label_a": "true_label",
                "label_b": "comparison_label",
            }
        )
        [
            [
                "model",
                "image_id",
                "comparison_image_id",
                "true_label",
                "comparison_label",
                "wasserstein",
            ]
        ]
    )

    side_b = (
        pairwise_df
        .rename(
            columns={
                "image_id_b": "image_id",
                "image_id_a": "comparison_image_id",
                "label_b": "true_label",
                "label_a": "comparison_label",
            }
        )
        [
            [
                "model",
                "image_id",
                "comparison_image_id",
                "true_label",
                "comparison_label",
                "wasserstein",
            ]
        ]
    )

    return pd.concat(
        [side_a, side_b],
        ignore_index=True,
    )


def compute_per_image_pair_separability(
    pairwise_df: pd.DataFrame,
    emotions: Sequence[Hashable] | None = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Compute one separability score per image, model, and emotion pair.

    For emotion pair A-B:

    For an image from A:
        within = mean distance to other A images
        between = mean distance to B images

    For an image from B:
        within = mean distance to other B images
        between = mean distance to A images

    Scores:
        difference = between - within
        ratio      = between / within

    Higher scores indicate greater separability.
    """
    directed_df = make_directed_pairwise_distances(
        pairwise_df
    )

    if emotions is None:
        emotions = list(
            pd.unique(
                pd.concat(
                    [
                        directed_df["true_label"],
                        directed_df["comparison_label"],
                    ],
                    ignore_index=True,
                )
            )
        )
    else:
        emotions = list(emotions)

    output_rows = []

    for emotion_a, emotion_b in combinations(
        emotions,
        2,
    ):
        pair_emotions = {
            emotion_a,
            emotion_b,
        }

        pair_df = directed_df[
            directed_df["true_label"].isin(
                pair_emotions
            )
            & directed_df["comparison_label"].isin(
                pair_emotions
            )
        ].copy()

        if pair_df.empty:
            continue

        for (
            model,
            image_id,
            true_label,
        ), image_df in pair_df.groupby(
            [
                "model",
                "image_id",
                "true_label",
            ],
            sort=False,
        ):
            if true_label not in pair_emotions:
                continue

            other_emotion = (
                emotion_b
                if true_label == emotion_a
                else emotion_a
            )

            within_distances = image_df.loc[
                image_df["comparison_label"]
                == true_label,
                "wasserstein",
            ].to_numpy(dtype=float)

            between_distances = image_df.loc[
                image_df["comparison_label"]
                == other_emotion,
                "wasserstein",
            ].to_numpy(dtype=float)

            within_distances = within_distances[
                np.isfinite(within_distances)
            ]

            between_distances = between_distances[
                np.isfinite(between_distances)
            ]

            if len(within_distances) == 0:
                continue

            if len(between_distances) == 0:
                continue

            within_mean = float(
                np.mean(within_distances)
            )

            between_mean = float(
                np.mean(between_distances)
            )

            output_rows.append({
                "model": model,
                "image_id": image_id,
                "true_label": true_label,
                "emotion_a": emotion_a,
                "emotion_b": emotion_b,
                "within_distance": within_mean,
                "between_distance": between_mean,
                "distance_difference": (
                    between_mean - within_mean
                ),
                "distance_ratio": (
                    between_mean
                    / max(within_mean, eps)
                ),
                "n_within_images": len(
                    within_distances
                ),
                "n_between_images": len(
                    between_distances
                ),
            })

    result_df = pd.DataFrame(output_rows)

    if result_df.empty:
        raise ValueError(
            "No emotion-pair separability scores "
            "could be calculated."
        )

    return result_df


def paired_ttests_by_emotion_pair(
    pairwise_df: pd.DataFrame,
    model_a: Hashable,
    model_b: Hashable,
    emotions: Sequence[Hashable] | None = None,
    score_col: str = "distance_difference",
    alternative: str = "two-sided",
    correction: str = "holm",
    alpha: float = 0.05,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Compare two models separately for every emotion pair.

    For each image and emotion pair, separability is:

        between-emotion distance
        minus
        within-emotion distance

    The same images are matched between models and tested using a
    paired t-test.

    Parameters
    ----------
    pairwise_df:
        Output from compute_pairwise_image_distances().

    model_a, model_b:
        Models to compare.

    emotions:
        Emotion classes to include.

    score_col:
        Recommended:
            "distance_difference"

        Alternative:
            "distance_ratio"

    alternative:
        "two-sided", "greater", or "less".

    correction:
        "holm", "bonferroni", "fdr_bh", or "none".

    Returns
    -------
    per_image_pair_df:
        One separability score per image/model/emotion-pair.

    matched_scores_df:
        Matched model scores for each image and emotion pair.

    test_results_df:
        One t-test result per emotion pair.
    """
    if alternative not in {
        "two-sided",
        "greater",
        "less",
    }:
        raise ValueError(
            "alternative must be 'two-sided', "
            "'greater', or 'less'."
        )

    per_image_pair_df = (
        compute_per_image_pair_separability(
            pairwise_df=pairwise_df,
            emotions=emotions,
        )
    )

    if score_col not in per_image_pair_df.columns:
        raise ValueError(
            f"Unknown score column: {score_col!r}"
        )

    model_a_df = (
        per_image_pair_df.loc[
            per_image_pair_df["model"] == model_a,
            [
                "image_id",
                "true_label",
                "emotion_a",
                "emotion_b",
                score_col,
            ],
        ]
        .rename(
            columns={
                score_col: "score_model_a"
            }
        )
    )

    model_b_df = (
        per_image_pair_df.loc[
            per_image_pair_df["model"] == model_b,
            [
                "image_id",
                "true_label",
                "emotion_a",
                "emotion_b",
                score_col,
            ],
        ]
        .rename(
            columns={
                score_col: "score_model_b"
            }
        )
    )

    matched_scores_df = model_a_df.merge(
        model_b_df,
        on=[
            "image_id",
            "true_label",
            "emotion_a",
            "emotion_b",
        ],
        how="inner",
        validate="one_to_one",
    )

    matched_scores_df = (
        matched_scores_df
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                "score_model_a",
                "score_model_b",
            ]
        )
        .reset_index(drop=True)
    )

    matched_scores_df[
        "difference_a_minus_b"
    ] = (
        matched_scores_df["score_model_a"]
        - matched_scores_df["score_model_b"]
    )

    result_rows = []

    for (
        emotion_a,
        emotion_b,
    ), group_df in matched_scores_df.groupby(
        ["emotion_a", "emotion_b"],
        sort=False,
    ):
        scores_a = group_df[
            "score_model_a"
        ].to_numpy(dtype=float)

        scores_b = group_df[
            "score_model_b"
        ].to_numpy(dtype=float)

        differences = scores_a - scores_b
        n_images = len(differences)

        if n_images < 2:
            continue

        test_result = ttest_rel(
            scores_a,
            scores_b,
            alternative=alternative,
            nan_policy="omit",
        )

        mean_difference = float(
            np.mean(differences)
        )

        std_difference = float(
            np.std(differences, ddof=1)
        )

        cohens_dz = (
            mean_difference / std_difference
            if std_difference > 0
            else np.nan
        )

        # Optional class-side summaries.
        side_a_scores = group_df.loc[
            group_df["true_label"] == emotion_a,
            "difference_a_minus_b",
        ]

        side_b_scores = group_df.loc[
            group_df["true_label"] == emotion_b,
            "difference_a_minus_b",
        ]

        result_rows.append({
            "emotion_a": emotion_a,
            "emotion_b": emotion_b,
            "model_a": model_a,
            "model_b": model_b,
            "score": score_col,
            "n_images": n_images,
            "n_images_emotion_a": int(
                (
                    group_df["true_label"]
                    == emotion_a
                ).sum()
            ),
            "n_images_emotion_b": int(
                (
                    group_df["true_label"]
                    == emotion_b
                ).sum()
            ),
            "mean_score_model_a": float(
                np.mean(scores_a)
            ),
            "mean_score_model_b": float(
                np.mean(scores_b)
            ),
            "mean_difference_a_minus_b": (
                mean_difference
            ),
            "median_difference_a_minus_b": float(
                np.median(differences)
            ),
            "mean_difference_for_emotion_a_images": (
                float(side_a_scores.mean())
                if len(side_a_scores) > 0
                else np.nan
            ),
            "mean_difference_for_emotion_b_images": (
                float(side_b_scores.mean())
                if len(side_b_scores) > 0
                else np.nan
            ),
            "std_difference": std_difference,
            "sem_difference": (
                std_difference / np.sqrt(n_images)
            ),
            "t_statistic": float(
                test_result.statistic
            ),
            "p_value": float(
                test_result.pvalue
            ),
            "cohens_dz": float(cohens_dz),
        })

    test_results_df = pd.DataFrame(
        result_rows
    )

    if test_results_df.empty:
        raise ValueError(
            "No emotion-pair tests could be performed."
        )

    test_results_df[
        "p_value_corrected"
    ] = adjust_pvalues(
        test_results_df["p_value"].to_numpy(),
        method=correction,
    )

    test_results_df[
        "significant_corrected"
    ] = (
        test_results_df["p_value_corrected"]
        < alpha
    )

    test_results_df["direction"] = np.select(
        [
            test_results_df[
                "mean_difference_a_minus_b"
            ] > 0,
            test_results_df[
                "mean_difference_a_minus_b"
            ] < 0,
        ],
        [
            f"{model_a} more separable",
            f"{model_b} more separable",
        ],
        default="equal mean separability",
    )

    test_results_df["correction"] = correction
    test_results_df["alpha"] = alpha

    test_results_df = (
        test_results_df
        .sort_values(
            "p_value_corrected",
            na_position="last",
        )
        .reset_index(drop=True)
    )

    return (
        per_image_pair_df,
        matched_scores_df,
        test_results_df,
    )

# ============================================================
# Class-balanced per-image separability
# ============================================================

def compute_class_balanced_per_image_separability(
    pairwise_df: pd.DataFrame,
    emotions: Sequence[Hashable] | None = None,
    eps: float = 1e-12,
    require_all_other_emotions: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute class-balanced separability for every image and model.

    For an anchor image from emotion y:

        within_distance =
            mean distance to all other images from emotion y

        between_distance_balanced =
            first compute the mean distance to each other emotion separately,
            then average those emotion-specific means equally

        distance_difference_balanced =
            between_distance_balanced - within_distance

        distance_ratio_balanced =
            between_distance_balanced / within_distance

    Therefore, every other emotion contributes equally, regardless of its
    number of images.

    Parameters
    ----------
    pairwise_df:
        Pairwise image-distance dataframe returned by
        compute_pairwise_image_distances().

    emotions:
        Full emotion list.

        If None, all labels found in pairwise_df are used.

    require_all_other_emotions:
        If True, an image must have valid distances to every other requested
        emotion.

    Returns
    -------
    per_image_df:
        One class-balanced separability score per image and model.

    per_target_class_df:
        Intermediate dataframe with the mean distance from every anchor image
        to every target emotion.
    """
    directed_df = make_directed_pairwise_distances(
        pairwise_df
    )

    if emotions is None:
        emotions = list(
            pd.unique(
                pd.concat(
                    [
                        directed_df["true_label"],
                        directed_df["comparison_label"],
                    ],
                    ignore_index=True,
                )
            )
        )
    else:
        emotions = list(emotions)

    directed_df = directed_df[
        directed_df["true_label"].isin(emotions)
        & directed_df["comparison_label"].isin(emotions)
    ].copy()

    # For every anchor image, compute its mean distance to each emotion.
    per_target_class_df = (
        directed_df
        .groupby(
            [
                "model",
                "image_id",
                "true_label",
                "comparison_label",
            ],
            as_index=False,
        )
        .agg(
            mean_distance_to_class=(
                "wasserstein",
                "mean",
            ),
            median_distance_to_class=(
                "wasserstein",
                "median",
            ),
            n_images_in_comparison_class=(
                "wasserstein",
                "size",
            ),
        )
    )

    output_rows = []

    for (
        model,
        image_id,
        true_label,
    ), image_df in per_target_class_df.groupby(
        [
            "model",
            "image_id",
            "true_label",
        ],
        sort=False,
    ):
        within_rows = image_df[
            image_df["comparison_label"]
            == true_label
        ]

        between_rows = image_df[
            image_df["comparison_label"]
            != true_label
        ]

        if len(within_rows) != 1:
            # This occurs when there is no second image in the anchor's
            # own emotion class.
            continue

        expected_other_emotions = [
            emotion
            for emotion in emotions
            if emotion != true_label
        ]

        available_other_emotions = set(
            between_rows["comparison_label"]
        )

        missing_other_emotions = [
            emotion
            for emotion in expected_other_emotions
            if emotion not in available_other_emotions
        ]

        if (
            require_all_other_emotions
            and missing_other_emotions
        ):
            continue

        within_distance = float(
            within_rows[
                "mean_distance_to_class"
            ].iloc[0]
        )

        # Each other emotion contributes one mean, so all other emotions
        # receive equal weight.
        between_distance_balanced = float(
            between_rows[
                "mean_distance_to_class"
            ].mean()
        )

        output_rows.append({
            "model": model,
            "image_id": image_id,
            "true_label": true_label,
            "within_distance": within_distance,
            "between_distance_balanced": (
                between_distance_balanced
            ),
            "distance_difference_balanced": (
                between_distance_balanced
                - within_distance
            ),
            "distance_ratio_balanced": (
                between_distance_balanced
                / max(within_distance, eps)
            ),
            "n_other_emotions": int(
                between_rows[
                    "comparison_label"
                ].nunique()
            ),
            "expected_other_emotions": (
                len(expected_other_emotions)
            ),
            "missing_other_emotions": (
                missing_other_emotions
            ),
            "n_within_images": int(
                within_rows[
                    "n_images_in_comparison_class"
                ].iloc[0]
            ),
        })

    per_image_df = pd.DataFrame(output_rows)

    if per_image_df.empty:
        raise ValueError(
            "No class-balanced per-image scores could be calculated."
        )

    per_image_df["complete_other_emotion_set"] = (
        per_image_df["n_other_emotions"]
        == per_image_df["expected_other_emotions"]
    )

    return per_image_df, per_target_class_df


# ============================================================
# Seven paired t-tests: one per anchor emotion
# ============================================================
def paired_ttests_by_emotion_class_balanced(
    pairwise_df: pd.DataFrame,
    model_a: Hashable,
    model_b: Hashable,
    emotions: Sequence[Hashable] | None = None,
    score_col: str = "distance_difference_balanced",
    alternative: str = "two-sided",
    correction: str = "holm",
    alpha: float = 0.05,
    confidence_level: float = 0.95,
    require_all_other_emotions: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Compare two models separately for each anchor emotion.

    Example for fear
    ----------------
    For every fear image and model:

        within =
            mean distance to other fear images

        between-balanced =
            mean of:
                mean distance to anger images
                mean distance to disgust images
                mean distance to happy images
                mean distance to sad images
                mean distance to surprise images
                mean distance to neutral images

        separability =
            between-balanced - within

    Then the same fear images are compared between models using a paired
    t-test.

    Parameters
    ----------
    pairwise_df:
        Pairwise image-distance dataframe.

    model_a, model_b:
        Models to compare.

    emotions:
        Emotions to include and output order.

    score_col:
        Recommended:
            "distance_difference_balanced"

        Alternative:
            "distance_ratio_balanced"

    alternative:
        "two-sided", "greater", or "less".

    correction:
        Multiple-comparison correction across the seven emotion tests.

    Returns
    -------
    per_image_balanced_df:
        Class-balanced score for every image and model.

    matched_scores_df:
        Matched scores for the selected models.

    emotion_tests_df:
        One paired t-test result per anchor emotion.
    """
    if alternative not in {
        "two-sided",
        "greater",
        "less",
    }:
        raise ValueError(
            "alternative must be 'two-sided', "
            "'greater', or 'less'."
        )

    (
        per_image_balanced_df,
        per_target_class_df,
    ) = compute_class_balanced_per_image_separability(
        pairwise_df=pairwise_df,
        emotions=emotions,
        require_all_other_emotions=(
            require_all_other_emotions
        ),
    )

    if score_col not in per_image_balanced_df.columns:
        raise ValueError(
            f"Unknown score column: {score_col!r}"
        )

    model_a_df = (
        per_image_balanced_df.loc[
            per_image_balanced_df["model"]
            == model_a,
            [
                "image_id",
                "true_label",
                score_col,
            ],
        ]
        .rename(
            columns={
                score_col: "score_model_a"
            }
        )
    )

    model_b_df = (
        per_image_balanced_df.loc[
            per_image_balanced_df["model"]
            == model_b,
            [
                "image_id",
                "true_label",
                score_col,
            ],
        ]
        .rename(
            columns={
                score_col: "score_model_b"
            }
        )
    )

    matched_scores_df = model_a_df.merge(
        model_b_df,
        on=[
            "image_id",
            "true_label",
        ],
        how="inner",
        validate="one_to_one",
    )

    matched_scores_df = (
        matched_scores_df
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                "score_model_a",
                "score_model_b",
            ]
        )
        .reset_index(drop=True)
    )

    if matched_scores_df.empty:
        raise ValueError(
            "No matched images were found between the models."
        )

    matched_scores_df[
        "difference_a_minus_b"
    ] = (
        matched_scores_df["score_model_a"]
        - matched_scores_df["score_model_b"]
    )

    if emotions is None:
        emotions = list(
            pd.unique(
                matched_scores_df["true_label"]
            )
        )
    else:
        emotions = list(emotions)

    result_rows = []

    alpha_ci = 1.0 - confidence_level

    for emotion in emotions:
        emotion_df = matched_scores_df[
            matched_scores_df["true_label"]
            == emotion
        ].copy()

        n_images = len(emotion_df)

        if n_images < 2:
            result_rows.append({
                "emotion": emotion,
                "model_a": model_a,
                "model_b": model_b,
                "score": score_col,
                "n_images": n_images,
                "mean_model_a": np.nan,
                "mean_model_b": np.nan,
                "mean_difference_a_minus_b": np.nan,
                "t_statistic": np.nan,
                "p_value": np.nan,
                "cohens_dz": np.nan,
            })
            continue

        scores_a = emotion_df[
            "score_model_a"
        ].to_numpy(dtype=float)

        scores_b = emotion_df[
            "score_model_b"
        ].to_numpy(dtype=float)

        differences = scores_a - scores_b

        test_result = ttest_rel(
            scores_a,
            scores_b,
            alternative=alternative,
            nan_policy="omit",
        )

        mean_difference = float(
            np.mean(differences)
        )

        std_difference = float(
            np.std(
                differences,
                ddof=1,
            )
        )

        sem_difference = (
            std_difference
            / np.sqrt(n_images)
        )

        critical_t = t.ppf(
            1.0 - alpha_ci / 2.0,
            df=n_images - 1,
        )

        ci_lower = (
            mean_difference
            - critical_t * sem_difference
        )

        ci_upper = (
            mean_difference
            + critical_t * sem_difference
        )

        cohens_dz = (
            mean_difference / std_difference
            if std_difference > 0
            else np.nan
        )

        result_rows.append({
            "emotion": emotion,
            "model_a": model_a,
            "model_b": model_b,
            "score": score_col,
            "alternative": alternative,
            "n_images": n_images,
            "mean_model_a": float(
                np.mean(scores_a)
            ),
            "mean_model_b": float(
                np.mean(scores_b)
            ),
            "mean_difference_a_minus_b": (
                mean_difference
            ),
            "median_difference_a_minus_b": float(
                np.median(differences)
            ),
            "std_difference": std_difference,
            "sem_difference": sem_difference,
            "confidence_level": confidence_level,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "t_statistic": float(
                test_result.statistic
            ),
            "degrees_of_freedom": n_images - 1,
            "p_value": float(
                test_result.pvalue
            ),
            "cohens_dz": float(cohens_dz),
        })

    emotion_tests_df = pd.DataFrame(
        result_rows
    )

    emotion_tests_df[
        "p_value_corrected"
    ] = adjust_pvalues(
        emotion_tests_df["p_value"].to_numpy(),
        method=correction,
    )

    emotion_tests_df[
        "significant_corrected"
    ] = (
        emotion_tests_df[
            "p_value_corrected"
        ]
        < alpha
    )

    emotion_tests_df["direction"] = np.select(
        [
            emotion_tests_df[
                "mean_difference_a_minus_b"
            ] > 0,
            emotion_tests_df[
                "mean_difference_a_minus_b"
            ] < 0,
        ],
        [
            f"{model_a} more separable",
            f"{model_b} more separable",
        ],
        default="equal mean separability",
    )

    emotion_tests_df["correction"] = correction
    emotion_tests_df["alpha"] = alpha

    emotion_tests_df["emotion"] = pd.Categorical(
        emotion_tests_df["emotion"],
        categories=emotions,
        ordered=True,
    )

    emotion_tests_df = (
        emotion_tests_df
        .sort_values("emotion")
        .reset_index(drop=True)
    )

    return (
        per_image_balanced_df,
        matched_scores_df,
        emotion_tests_df,
    )
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Usage
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
img_subset = 'CORRECT_PRED'  # 'CORRECT_PRED' / 'ALL_PRED' / 'ALL_TRUE'
results = pd.read_pickle(f'attention_maps/image_attention_separability_sigma8_{img_subset}/attention_map_separability_results.pkl')

models_to_compare = [
    "RAF_blur8_pretrained0-8_freeze_fix_projs",
    "RAF_blur8_freeze",
]

# matched_df, test_summary_df = (
#     paired_image_level_separability_test(
#         per_image_df=results["per_image"],
#         model_a=models_to_compare[0],
#         model_b=models_to_compare[1],
#         score_col="distance_difference",
#         alternative="two-sided",
#         run_wilcoxon=True,
#     )
# )
#
# print(test_summary_df.to_string(index=False))
#
# matched_balanced_df, class_effects_df, balanced_test_df = (
#     class_balanced_paired_ttest(
#         per_image_df=results["per_image"],
#         model_a=models_to_compare[0],
#         model_b=models_to_compare[1],
#         score_col="distance_difference",
#         alternative="two-sided",
#     )
# )
#
# print(balanced_test_df.to_string(index=False))
# print(class_effects_df.to_string(index=False))

#
# per_image_pair_df, matched_pair_scores_df, pair_tests_df, = (
#     paired_ttests_by_emotion_pair(
#         pairwise_df=results["pairwise_distances"],
#         model_a=models_to_compare[0],
#         model_b=models_to_compare[1],
#         emotions=EMOTIONS,
#         alternative="two-sided",
#         correction="holm",
#         alpha=0.05,
#     )
# )

(
    per_image_balanced_df,
    matched_balanced_scores_df,
    emotion_tests_df,
) = paired_ttests_by_emotion_class_balanced(
    pairwise_df=results["pairwise_distances"],
    model_a=models_to_compare[0],
    model_b=models_to_compare[1],
    emotions=EMOTIONS,
    score_col="distance_difference_balanced",
    alternative="two-sided",
    correction="holm",
    alpha=0.05,
    confidence_level=0.95,
    require_all_other_emotions=True,
)
save_dir = Path(f"attention_maps/emotion_separability_tests/{img_subset}/comp_image_separability_data")
pickle_filename = "comp_image_separability_results_per_emotion.pkl"

comp_results = {
    "per_image_balanced": per_image_balanced_df,
    "matched_balanced_scores": matched_balanced_scores_df,
    "emotion_tests": emotion_tests_df,
}

if save_results:
    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    pd.to_pickle(
        comp_results,
        save_dir / pickle_filename,
    )

print('/')
