"""
26/6/26
Assumes dataframes "attention_comparisons/wasserstein_{pair}_true_label.csv" from code calc_wasserstein_dist.py, for all
'pair's in 'dists2comp' below.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel
from pathlib import Path


def filter_all_three_correct(
    pair_a_df,
    pair_b_df,
    pair_cols=("image_id", "target_label"),
):
    """
    Keep only paired observations where all three relevant predictions are correct:

        1. mixed model on low-res image
        2. low-resolution-only model on low-res image
        3. mixed model on high-res image

    Assumed structure:

    between_models_low_df:
        correct_a = mixed low correct
        correct_b = lowres low correct

    within_mixed_low_high_df:
        correct_a = mixed low correct
        correct_b = mixed high correct
    """

    df_a = pair_a_df.copy()
    df_b = pair_b_df.copy()

    # All correct in between-model comparison:
    # mixed low and lowres low
    between_all_correct = df_a[
        (df_a["correct_a"] == True) &
        (df_a["correct_b"] == True)
    ].copy()

    # All correct in within-mixed comparison:
    # mixed low and mixed high
    within_all_correct = df_b[
        (df_b["correct_a"] == True) &
        (df_b["correct_b"] == True)
    ].copy()

    # Keep only image/target pairs that survive both filters.
    valid_pairs = (
        between_all_correct[list(pair_cols)]
        .drop_duplicates()
        .merge(
            within_all_correct[list(pair_cols)].drop_duplicates(),
            on=list(pair_cols),
            how="inner",
        )
    )

    filtered_a = df_a.merge(
        valid_pairs,
        on=list(pair_cols),
        how="inner",
    )

    filtered_b = df_b.merge(
        valid_pairs,
        on=list(pair_cols),
        how="inner",
    )

    return filtered_a, filtered_b, valid_pairs


def prepare_paired_wasserstein_comparison(
    pair_a_df: pd.DataFrame,
    pair_b_df: pd.DataFrame,
    pair_cols=("image_id", "target_label"),
    distance_col="wasserstein_distance",
    suffix_a="_a",
    suffix_b="_b",
):
    """
    Pair two Wasserstein result dataframes.

    Parameters
    ----------
    pair_a_df:
        DataFrame comparing:
            mixed model on low-res images
            vs
            low-resolution-only model on low-res images

    pair_b_df:
        DataFrame comparing:
            mixed model on low-res images
            vs
            mixed model on high-res images

    pair_cols:
        Columns used to pair rows across the two analyses.
        Usually:
            ("image_id", "target_label")

        If you have several blur levels or conditions, use:
            ("image_id", "sigma", "target_label")

        But for low-vs-high comparisons, sigma may not match across dfs,
        so usually image_id + target_label is enough.

    distance_col:
        Name of Wasserstein distance column.

    Returns
    -------
    paired_df:
        DataFrame with one row per paired comparison.
    """
    df_a = pair_a_df.copy()
    df_b = pair_b_df.copy()

    keep_a = list(pair_cols) + [distance_col]
    keep_b = list(pair_cols) + [distance_col]

    # Add useful columns if they exist.
    optional_cols = [
        "true_label",
        "target_label",
        "correct_a",
        "correct_b",
        "pred_label_a",
        "pred_label_b",
        "sigma",
    ]

    for col in optional_cols:
        if col in df_a.columns and col not in keep_a:
            keep_a.append(col)

    for col in optional_cols:
        if col in df_b.columns and col not in keep_b:
            keep_b.append(col)

    df_a = df_a[keep_a].copy()
    df_b = df_b[keep_b].copy()

    paired = df_a.merge(
        df_b,
        on=list(pair_cols),
        how="inner",
        suffixes=(suffix_a, suffix_b),
    )

    dist_col_a = distance_col + suffix_a
    dist_col_b = distance_col + suffix_b

    paired["delta_a_minus_b"] = (
        paired[dist_col_a] - paired[dist_col_b]
    )

    paired["abs_delta"] = paired["delta_a_minus_b"].abs()

    return paired


def bootstrap_ci(
    values,
    statistic=np.mean,
    n_boot=10_000,
    ci=95,
    random_state=0,
):
    """
    Bootstrap confidence interval for a paired statistic.
    """
    rng = np.random.default_rng(random_state)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    n = len(values)

    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats.append(statistic(sample))

    alpha = 100 - ci
    lower = np.percentile(boot_stats, alpha / 2)
    upper = np.percentile(boot_stats, 100 - alpha / 2)

    return lower, upper


def permutation_sign_flip_test(
    deltas,
    alternative="less",
    n_perm=100_000,
    random_state=0,
):
    """
    Paired permutation test using sign flipping.

    Null hypothesis:
        The paired differences are symmetrically centered around 0.

    Parameters
    ----------
    deltas:
        Paired differences.

    alternative:
        "less":
            tests whether mean(delta) < 0
            meaning within-mixed low/high maps are more similar.

        "greater":
            tests whether mean(delta) > 0
            meaning within-mixed low/high maps are less similar.

        "two-sided":
            tests whether mean(delta) != 0.

    Returns
    -------
    p_value
    """
    rng = np.random.default_rng(random_state)

    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[~np.isnan(deltas)]

    observed = np.mean(deltas)

    signs = rng.choice([-1, 1], size=(n_perm, len(deltas)))
    permuted = signs * deltas[None, :]
    permuted_stats = permuted.mean(axis=1)

    if alternative == "less":
        p = np.mean(permuted_stats <= observed)
    elif alternative == "greater":
        p = np.mean(permuted_stats >= observed)
    elif alternative == "two-sided":
        p = np.mean(np.abs(permuted_stats) >= abs(observed))
    else:
        raise ValueError("alternative must be 'less', 'greater', or 'two-sided'")

    return float(p)


def paired_wasserstein_tests(
    paired_df: pd.DataFrame,
    delta_col="delta_a_minus_b",
    alternative="less",
    n_boot=10_000,
    n_perm=100_000,
    random_state=0,
):
    """
    Run paired tests on Wasserstein distance differences.

    delta = within_mixed_low_high - between_models_low

    alternative="less":
        Tests whether delta < 0.
        This means mixed low/high maps are MORE similar to each other.

    alternative="greater":
        Tests whether delta > 0.
        This means mixed low/high maps are LESS similar to each other.

    alternative="two-sided":
        Tests whether the two distances differ.
    """
    deltas = paired_df[delta_col].to_numpy(dtype=float)
    deltas = deltas[~np.isnan(deltas)]

    n = len(deltas)

    if n == 0:
        raise ValueError("No paired observations available.")

    mean_delta = np.mean(deltas)
    median_delta = np.median(deltas)
    sd_delta = np.std(deltas, ddof=1)

    # Cohen's dz for paired designs
    cohens_dz = mean_delta / sd_delta if sd_delta > 0 else np.nan

    # Wilcoxon signed-rank test
    wilcoxon_result = wilcoxon(
        deltas,
        alternative=alternative,
        zero_method="wilcox",
    )

    # Paired t-test against 0
    ttest_result = ttest_rel(
        paired_df[delta_col],
        np.zeros(len(paired_df)),
        alternative=alternative,
        nan_policy="omit",
    )

    mean_ci = bootstrap_ci(
        deltas,
        statistic=np.mean,
        n_boot=n_boot,
        random_state=random_state,
    )

    median_ci = bootstrap_ci(
        deltas,
        statistic=np.median,
        n_boot=n_boot,
        random_state=random_state,
    )

    permutation_p = permutation_sign_flip_test(
        deltas,
        alternative=alternative,
        n_perm=n_perm,
        random_state=random_state,
    )

    results = {
        "n_pairs": n,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "sd_delta": sd_delta,
        "cohens_dz": cohens_dz,

        "mean_delta_ci_low": mean_ci[0],
        "mean_delta_ci_high": mean_ci[1],

        "median_delta_ci_low": median_ci[0],
        "median_delta_ci_high": median_ci[1],

        "wilcoxon_statistic": wilcoxon_result.statistic,
        "wilcoxon_p": wilcoxon_result.pvalue,

        "paired_t_statistic": ttest_result.statistic,
        "paired_t_p": ttest_result.pvalue,

        "permutation_p": permutation_p,

        "alternative": alternative,
    }

    return pd.DataFrame([results])


# dists2comp = ['RAFblur8pretrained08freezefixprojs_vs_RAFblur8freeze',  # between models on low-res images
#               'RAFblur8pretrained08freezefixprojs_vs_RAFblur0pretrained08freeze']  # within mixed model, between image resolutions.

# Comparisons with high-res model, using native-resolution maps:
dists2comp = ['RAFblur8pretrained08freezefixprojs_vs_RAFblur8freeze_native',  # between models (variable vs unifrom) on low-res images
              'RAFblur0pretrained08freeze_vs_RAFblur0freeze_native']  # between models (variable vs unifrom) on high-res images.

work_dir = 'attention_comparisons/comp_high_low_native'
target_mode = 'true_label'

pair_a_df = pd.read_csv(f"{work_dir}/wasserstein_{dists2comp[0]}_{target_mode}.csv")
pair_b_df = pd.read_csv(f"{work_dir}/wasserstein_{dists2comp[1]}_{target_mode}.csv")

all3_df_a, all3_df_b, valid_pairs = filter_all_three_correct(
    pair_a_df=pair_a_df,
    pair_b_df=pair_b_df,
    pair_cols=("image_id", "target_label"),
)

print("Between-model rows before:", len(pair_a_df))
print("Within-mixed rows before:", len(pair_b_df))
print("Between-model rows after:", len(all3_df_a))
print("Within-mixed rows after:", len(all3_df_b))
print("Valid image/target pairs:", len(valid_pairs))


paired_df_all3_correct = prepare_paired_wasserstein_comparison(
    pair_a_df=all3_df_a,
    pair_b_df=all3_df_b,
    pair_cols=("image_id", "target_label"),
    distance_col="wasserstein_distance",
)

test_results = paired_wasserstein_tests(
    paired_df_all3_correct,
    alternative="two-sided",
    n_boot=10_000,
    n_perm=100_000,
    random_state=0,
)

test_results

Path(work_dir).mkdir(exist_ok=True)

paired_df_all3_correct.to_csv(
    f"{work_dir}/paired_wasserstein_comparison_all3_correct.csv",
    index=False,
)

test_results.to_csv(
    f"{work_dir}/paired_wasserstein_test_results_all3_correct.csv",
    index=False,
)
