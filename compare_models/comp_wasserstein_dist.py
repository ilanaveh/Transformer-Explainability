"""Paired comparison of two distance tables; delta always means A minus B."""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_1samp
from pathlib import Path


def _validate_pairs(df, pair_cols):
    missing = set(pair_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing pairing columns: {sorted(missing)}")
    if df[list(pair_cols)].isna().any().any() or df.duplicated(list(pair_cols)).any():
        raise ValueError("Missing or duplicate pairing keys; select conditions explicitly.")


def filter_all_predictions_correct(
    pair_a_df,
    pair_b_df,
    pair_cols=("image_id", "target_label"),
):
    """Keep keys where both predictions in EACH distance comparison are correct.

    This can involve three or four distinct predictions, depending on inputs.
    """

    df_a = pair_a_df.copy()
    df_b = pair_b_df.copy()
    for frame in (df_a, df_b):
        _validate_pairs(frame, pair_cols)
        for col in ("correct_a", "correct_b"):
            if not frame[col].map(lambda x: isinstance(x, (bool, np.bool_))).all():
                raise ValueError(f"{col} must contain boolean values.")

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

    filtered_a = between_all_correct.merge(
        valid_pairs,
        on=list(pair_cols),
        how="inner",
    )

    filtered_b = within_all_correct.merge(
        valid_pairs,
        on=list(pair_cols),
        how="inner",
    )

    return filtered_a, filtered_b, valid_pairs


# Compatibility alias for existing notebooks; no assumption of three predictions.
filter_all_three_correct = filter_all_predictions_correct


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
    pair_a_df, pair_b_df:
        Two distance tables from independently configured map comparisons.
        Their delta is always distance A minus distance B.

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

    for frame in (df_a, df_b):
        _validate_pairs(frame, pair_cols)
        if not np.isfinite(frame[distance_col].to_numpy(dtype=float)).all():
            raise ValueError("Distances must be finite.")
        if (frame[distance_col] < 0).any():
            raise ValueError("Distances must be nonnegative.")
    for col in ("distance_scale", "transport_height", "transport_width"):
        if (col in df_a) != (col in df_b):
            raise ValueError(f"Both inputs must specify {col}, or neither (legacy data).")
        if col in df_a:
            values_a = df_a[col].drop_duplicates().tolist()
            values_b = df_b[col].drop_duplicates().tolist()
            if len(values_a) != 1 or values_a != values_b:
                raise ValueError(f"Incompatible {col} values across distance tables.")

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
        "model_a",
        "model_b",
        "sigma_a",
        "sigma_b",
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
        validate="one_to_one",
    )

    if paired.empty:
        raise ValueError("No matched distance observations.")

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

    if not values.size or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be nonempty and finite.")
    if n_boot < 1 or not 0 < ci < 100:
        raise ValueError("n_boot must be positive and ci must lie between 0 and 100.")

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
    """Monte Carlo sign-flip test of mean(A - B).

    less means distance A is smaller; greater means distance A is larger.
    Uses bounded batches and a plus-one correction to avoid zero p-values.
    """
    if alternative not in ("less", "greater", "two-sided"):
        raise ValueError("alternative must be less, greater, or two-sided")
    deltas = np.asarray(deltas, dtype=float)
    if not deltas.size or not np.isfinite(deltas).all():
        raise ValueError("Differences must be nonempty and finite.")
    if n_perm < 1:
        raise ValueError("n_perm must be positive.")
    rng = np.random.default_rng(random_state)
    observed = deltas.mean()
    count = 0
    batch_size = max(1, min(1000, 1_000_000 // len(deltas)))
    for start in range(0, n_perm, batch_size):
        signs = rng.choice([-1, 1], size=(min(batch_size, n_perm - start), len(deltas)))
        stats = (signs * deltas).mean(axis=1)
        if alternative == "less":
            count += np.count_nonzero(stats <= observed)
        elif alternative == "greater":
            count += np.count_nonzero(stats >= observed)
        else:
            count += np.count_nonzero(np.abs(stats) >= abs(observed))
    return float((count + 1) / (n_perm + 1))


def paired_wasserstein_tests(
    paired_df: pd.DataFrame,
    delta_col="delta_a_minus_b",
    alternative="two-sided",
    n_boot=10_000,
    n_perm=100_000,
    random_state=0,
):
    """Test paired delta = distance A - distance B.

    less: A is smaller; greater: A is larger; two-sided: they differ.
    """
    deltas = paired_df[delta_col].to_numpy(dtype=float)
    if not np.isfinite(deltas).all():
        raise ValueError("Differences must be finite.")

    n = len(deltas)

    if n < 2:
        raise ValueError("At least two paired observations are required.")

    mean_delta = np.mean(deltas)
    median_delta = np.median(deltas)
    sd_delta = np.std(deltas, ddof=1)

    # Cohen's dz for paired designs
    cohens_dz = mean_delta / sd_delta if sd_delta > 0 else np.nan

    from types import SimpleNamespace
    if np.all(deltas == 0):
        wilcoxon_result = SimpleNamespace(statistic=0.0, pvalue=1.0)
        ttest_result = SimpleNamespace(statistic=0.0, pvalue=1.0)
    else:
        wilcoxon_result = wilcoxon(deltas, alternative=alternative, zero_method="wilcox")
        ttest_result = ttest_1samp(deltas, 0, alternative=alternative)

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare two paired Wasserstein distance tables; delta = A - B.")
    parser.add_argument("distance_a", type=Path)
    parser.add_argument("distance_b", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--all-correct", action="store_true", help="Require every prediction in both comparisons to be correct")
    parser.add_argument("--pair-cols", nargs="+", default=["image_id", "target_label"])
    parser.add_argument("--alternative", choices=["less", "greater", "two-sided"], default="two-sided")
    parser.add_argument("--n-perm", type=int, default=100_000)
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()
    a, b = pd.read_csv(args.distance_a), pd.read_csv(args.distance_b)
    if args.all_correct:
        a, b, _ = filter_all_predictions_correct(a, b, args.pair_cols)
    paired = prepare_paired_wasserstein_comparison(a, b, args.pair_cols)
    tests = paired_wasserstein_tests(paired, alternative=args.alternative,
                                      n_perm=args.n_perm, n_boot=args.n_boot)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "paired_distances.csv", index=False)
    tests.to_csv(args.output_dir / "paired_tests.csv", index=False)
    import json
    provenance = {
        "distance_a": str(args.distance_a.resolve()),
        "distance_b": str(args.distance_b.resolve()),
        "pair_cols": args.pair_cols,
        "all_correct": args.all_correct,
        "delta_definition": "distance A minus distance B",
    }
    (args.output_dir / "paired_distances.metadata.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
