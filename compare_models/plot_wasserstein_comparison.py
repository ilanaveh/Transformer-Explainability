"""
26/5/26
Assumes attention_comparisons/paired_wasserstein_comparison_all3_correct.csv saved from comp_wasserstein_dist.py:
    'wasserstein_distance_a' - col with dists for RAFblur8pretrained08freezefixprojs_vs_RAFblur8freeze
    'wasserstein_distance_b' - col with dists for RAFblur8pretrained08freezefixprojs_vs_RAFblur0pretrained08freeze
    'delta_a_minus_b' - col with deltas: wasserstein_distance_a - wasserstein_distance_b
                        aka: mixed-vs-low-models minus mixed-model-low-vs-high-images
                        aka: positive = maps between models are more different.

28/6/26
Assumes attention_comparisons/comp_high_low_native/paired_wasserstein_comparison_all3_correct.csv saved from comp_wasserstein_dist.py:
    'wasserstein_distance_a' - col with dists for RAFblur8pretrained08freezefixprojs_vs_RAFblur8freeze_native
    'wasserstein_distance_b' - col with dists for RAFblur0pretrained08freeze_vs_RAFblur0freeze_native
    'delta_a_minus_b' - col with deltas: wasserstein_distance_a - wasserstein_distance_b
                        aka: mixed-vs-low-models minus mixed-vs-high-models
                        aka: positive = maps between low models are more different.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from pathlib import Path


def bootstrap_ci(values, statistic=np.mean, n_boot=10_000, ci=95, random_state=0):
    rng = np.random.default_rng(random_state)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_stats.append(statistic(sample))

    alpha = 100 - ci
    return (
        np.percentile(boot_stats, alpha / 2),
        np.percentile(boot_stats, 100 - alpha / 2),
    )


def plot_paired_wasserstein_all3_correct(
    paired_df,
    dist_a_col="wasserstein_distance_between_models_low",
    dist_b_col="wasserstein_distance_within_mixed_low_high",
    label_a="between_models_low",
    label_b="within_mixed_low_vs_high",
    output_path=None,
    title="Attention-map similarity in images correctly classified by all models",
    jitter=0.035,
    random_state=0,
):
    """
    Paper-style paired plot.

    Left:
        Mixed model low-res vs high-res maps

    Right:
        Mixed model low-res vs low-resolution-only model low-res maps

    Lower Wasserstein = more similar attention maps.
    """
    rng = np.random.default_rng(random_state)
    df = paired_df.dropna(subset=[dist_a_col, dist_b_col]).copy()

    dist_a = df[dist_a_col].to_numpy(dtype=float)
    dist_b = df[dist_b_col].to_numpy(dtype=float)
    delta = dist_b - dist_a

    n = len(df)

    # One-sided Wilcoxon:
    # alternative='less' tests whether within < between; 'two-sided' tests whether abs difference > 0.
    w = wilcoxon(dist_b, dist_a, alternative="two-sided")

    x_a = np.zeros(n) + rng.normal(0, jitter, size=n)
    x_b = np.ones(n) + rng.normal(0, jitter, size=n)

    fig, ax = plt.subplots(figsize=(5.2, 4.8))

    # Paired lines
    for i in range(n):
        ax.plot(
            [x_a[i], x_b[i]],
            [dist_a[i], dist_b[i]],
            linewidth=0.7,
            alpha=0.25,
        )

    # Individual points
    ax.scatter(
        x_a,
        dist_a,
        s=18,
        alpha=0.55,
        label="Individual image/target pairs",
    )

    ax.scatter(
        x_b,
        dist_b,
        s=18,
        alpha=0.55,
    )

    # Means and bootstrap CIs
    mean_a = np.mean(dist_a)
    mean_b = np.mean(dist_b)

    ci_a = bootstrap_ci(dist_a, statistic=np.mean, random_state=random_state)
    ci_b = bootstrap_ci(dist_b, statistic=np.mean, random_state=random_state + 1)

    ax.errorbar(
        [0],
        [mean_a],
        yerr=[[mean_a - ci_a[0]], [ci_a[1] - mean_a]],
        fmt="o",
        markersize=8,
        capsize=5,
        linewidth=2,
        label="Mean ± 95% bootstrap CI",
    )

    ax.errorbar(
        [1],
        [mean_b],
        yerr=[[mean_b - ci_b[0]], [ci_b[1] - mean_b]],
        fmt="o",
        markersize=8,
        capsize=5,
        linewidth=2,
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            label_a,
            label_b
            # "Mixed model\nlow vs high",
            # "Mixed vs low-res model\non low images",
        ]
    )

    ax.set_ylabel("Wasserstein distance")
    ax.set_title(title)

    subtitle = (
        f"n = {n} paired image/target maps; "
        f"Wilcoxon one-sided p = {w.pvalue:.3g}; "
        f"mean Δ = {np.mean(delta):.4f}"
    )

    ax.text(
        0.5,
        1.03,
        subtitle,
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=9,
    )

    ax.axhline(0, linewidth=0.8, alpha=0.4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, loc="best", fontsize=8)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

    return {
        "n": n,
        f"mean_{label_a}": mean_a,
        f"mean_{label_b}": mean_b,
        "mean_delta_within_minus_between": np.mean(delta),
        "median_delta_within_minus_between": np.median(delta),
        "wilcoxon_statistic": w.statistic,
        "wilcoxon_p": w.pvalue,
        "percent_within_more_similar": 100 * np.mean(delta < 0),
    }


def plot_delta_hist(
    df,
    delta_col="delta_a_minus_b",
    output_path="figures/delta_wasserstein_hist_paper.png",
    bin_width=None,
    n_bins_each_side=20,
    title=None,
    xlabel=None,
    figsize=(4.8, 3.6),
    dpi=600,
    show_mean=False,
    show_median=False,
    show_ann=False
):
    """
    histogram for paired Wasserstein-distance deltas.

    delta < 0:
        condition A has lower Wasserstein distance, meaning more similar maps.

    delta > 0:
        condition B has lower Wasserstein distance, meaning more similar maps.

    Ensures that 0 is exactly a bin boundary.
    """
    x = df[delta_col].dropna().to_numpy(dtype=float)

    if len(x) == 0:
        raise ValueError("No non-NaN values found.")

    mean_x = np.mean(x)
    median_x = np.median(x)
    ci_low, ci_high = bootstrap_ci(x, statistic=np.mean)

    percent_negative = 100 * np.mean(x < 0)
    n = len(x)

    # --------------------------------------------------------
    # Create bins with 0 exactly as a bin edge
    # --------------------------------------------------------
    if bin_width is not None:
        left_edges = np.arange(0, x.min() - bin_width, -bin_width)[::-1]
        right_edges = np.arange(0, x.max() + bin_width, bin_width)
        bins = np.unique(np.concatenate([left_edges, right_edges]))
    else:
        max_abs = np.max(np.abs(x))
        bins = np.linspace(
            -max_abs,
            max_abs,
            2 * n_bins_each_side + 1,
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        x,
        bins=bins,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )

    if show_mean:
        ax.axvline(
            mean_x,
            linewidth=1.6,
            label=f"Mean Δ = {mean_x:.3f}",
        )

    if show_median:
        ax.axvline(
            median_x,
            linestyle=":",
            linewidth=1.6,
            label=f"Median Δ = {median_x:.3f}",
        )

    if title is None:
        title = "Paired difference in attention-map Wasserstein distance"

    if xlabel is None:
        xlabel = (
            "Δ Wasserstein distance\n"
            "(mixed–vs-uniform-lowres minus mixed–vs-uniform-hires)"
        )

    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Number of images", fontsize=10)

    annotation = (
        f"n = {n}\n"
        f"95% CI mean Δ: [{ci_low:.3f}, {ci_high:.3f}]\n"
        f"{percent_negative:.1f}% Δ < 0"
    )

    if show_ann:
        ax.text(
            0.98,
            0.96,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                linewidth=0.5,
                alpha=0.9,
            ),
        )

    ax.legend(
        frameon=False,
        fontsize=8,
        loc="upper left",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")

    plt.show()

    return {
        "n": n,
        "mean_delta": mean_x,
        "median_delta": median_x,
        "mean_ci_low": ci_low,
        "mean_ci_high": ci_high,
        "percent_delta_less_than_zero": percent_negative,
        "output_path_png": str(output_path),
        "output_path_pdf": str(output_path.with_suffix(".pdf")),
    }


if __name__ == "__main__":
    paired_df_all3_correct = pd.read_csv('attention_comparisons/comp_high_low_native/paired_wasserstein_comparison_all3_correct.csv')

    plot_stats = plot_paired_wasserstein_all3_correct(
        paired_df_all3_correct,
        dist_a_col='wasserstein_distance_a',
        dist_b_col='wasserstein_distance_b',
        label_a='mixed_vs_uniform_lowres',
        label_b='mixed_vs_uniform_hires',
        output_path="figures/paired_wasserstein_all3_correct_high_vs_low.png",
    )

    plot_stats

    stats = plot_delta_hist(
        paired_df_all3_correct,
        delta_col="delta_a_minus_b",
        output_path="figures/delta_wasserstein_all3_correct_high_vs_low.png",
        bin_width=0.02,  # adjust to your Wasserstein scale
    )

    stats