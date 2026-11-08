"""
31/5/26
Moved the visualization code from 'build_prototypes.py' to here.
"""
# ============================================================
# 7. Visualize prototypes
# ============================================================
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from create_attn_maps import ensure_2d_map, resize_map_numpy, normalize_relevance_map, normalize_relevance_map, EMOTIONS
import pandas as pd
import ast
import torch


class_n = {
    'Angry': 162,
    'Disgust': 160,
    'Fear': 74,
    'Sad': 478,
    'Happy': 1185,
    'Surprise': 329,
    'Neutral': 680,
}

POOLING = True


def minmax_normalize_map(
    relevance: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Min-max normalization for visualization only.

    Do not use this for prototype averaging unless you specifically want
    visual contrast rather than probability-like relevance allocation.
    """
    relevance = relevance.astype(np.float32)
    min_val = relevance.min()
    max_val = relevance.max()

    if max_val - min_val < eps:
        return np.zeros_like(relevance, dtype=np.float32)

    return (relevance - min_val) / (max_val - min_val)


def plot_prototype_grid(
    prototypes: dict,
    model_sigma_rows: list[tuple[str, int | float]],
    emotions: list[str],
    output_path: str | Path | None = None,
    figsize_per_cell: float = 2.5,
    use_minmax_for_display: bool = True,
    title: str | None = None,
    show_on_mean_face: bool = False,
    alpha: float = 0.45,
    cmap: str = "jet",
    subset: str = "CORRECT",
    interpolation_mode: str = 'bilinear',  # 'bilinear' for smoothing, 'nearest' for keeping patch-resolution (relevant for native prototypes)
):
    """
    Plot attention prototypes in a grid.

    Rows:
        Explicit model/sigma pairs, e.g.
        [
            ("mixed_pretrain", 0),
            ("mixed_pretrain", 8),
            ("lowres_pretrain", 8),
        ]

    Columns:
        Emotion classes.

    Expected prototype keys:
        (model, sigma, emotion)
    """
    n_rows = len(model_sigma_rows)
    n_cols = len(emotions)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_cell * n_cols, figsize_per_cell * n_rows),
        squeeze=False,
    )

    for r, (model, sigma) in enumerate(model_sigma_rows):
        for c, emotion in enumerate(emotions):
            ax = axes[r, c]

            key = (model, sigma, emotion)

            if key in prototypes:
                proto = prototypes[key]["prototype"]
                n = prototypes[key]["n"]

                display_proto = (
                    minmax_normalize_map(proto)
                    if use_minmax_for_display
                    else proto
                )

                if show_on_mean_face:
                    face = prototypes[key]['mean_face'].copy()
                    ax.imshow(np.clip(face, 0, 1))

                    if display_proto.shape != face.shape[:2]:
                        display_proto = torch.nn.functional.interpolate(
                            torch.from_numpy(display_proto).reshape(1, 1, 14, 14),
                            scale_factor=face.shape[0]/display_proto.shape[0],
                            mode=interpolation_mode).reshape(face.shape[:2]).numpy()

                    alpha_map = make_alpha_map(
                        display_proto,
                        min_alpha=0.2,
                        max_alpha=0.65,
                        gamma=1.5,
                    )

                    ax.imshow(display_proto, cmap=cmap, alpha=alpha_map)  # transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
                else:
                    ax.imshow(display_proto)
                if subset == 'CORRECT':
                    ax.set_title(f"{emotion}\n({int(n/class_n[emotion]*100)}% correct)", fontsize=9)
                else:
                    ax.set_title(emotion, fontsize=12)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "missing",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                )

            # ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])

            # Row label: make both model and sigma explicit.
            if c == 0:
                row_label = f"{model}\nσ = {sigma}"
                ax.set_ylabel(
                    row_label,
                    fontsize=10,
                    rotation=0,
                    labelpad=55,
                    va="center",
                    ha="right",
                )

    if title is not None:
        fig.suptitle(title, fontsize=13, y=1.02)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        # plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


def make_alpha_map(
    rel: np.ndarray,
    min_alpha: float = 0.0,
    max_alpha: float = 0.65,
    gamma: float = 1.5,
) -> np.ndarray:
    """
    Create a smooth value-dependent alpha map.

    Low relevance values become transparent.
    High relevance values become opaque.

    Parameters
    ----------
    rel:
        Normalized relevance map in range [0, 1].

    min_alpha:
        Minimum opacity for low values. Use 0.0 to make dark-blue areas transparent.

    max_alpha:
        Maximum opacity for high values.

    gamma:
        Controls how quickly opacity increases.
            gamma > 1: more conservative; only high values become opaque.
            gamma = 1: linear.
            gamma < 1: low/mid values become visible sooner.

    threshold:
        Values below this are fully transparent.
    """
    rel = np.asarray(rel, dtype=np.float32)
    alpha = np.clip(rel, 0, 1)

    # Smooth nonlinear ramp.
    alpha = alpha ** gamma

    # Scale to requested alpha range.
    alpha = min_alpha + alpha * (max_alpha - min_alpha)

    # Force very low values to be fully transparent.
    # alpha[rel <= threshold] = 0.0

    return alpha


def load_saved_prototypes(
    prototype_dir: str | Path,
    summary_filename: str = "prototype_summary.csv",
) -> dict:
    """
    Load previously saved prototypes from disk.

    Expects files created by save_prototypes(), including:
        prototype_summary.csv
        prototype__<key>.npy files

    Returns
    -------
    prototypes:
        Same dictionary structure returned by build_attention_prototypes().

        Example:
        {
            ("mixed_pretrain", 8, "happy"): {
                "prototype": np.ndarray,
                "n": int,
                "map_paths": [],
            }
        }
    """
    prototype_dir = Path(prototype_dir)
    summary_path = prototype_dir / summary_filename

    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find summary file: {summary_path}")

    summary_df = pd.read_csv(summary_path)

    required_cols = {"key", "prototype_path", "n"}
    missing_cols = required_cols - set(summary_df.columns)

    if missing_cols:
        raise ValueError(
            f"Summary file is missing required columns: {missing_cols}"
        )

    prototypes = {}

    for _, row in summary_df.iterrows():
        key = ast.literal_eval(row["key"])
        prototype_path = Path(row["prototype_path"])
        mean_face_path = Path(row["mean_face_path"])

        # Allow paths saved as either absolute or relative.
        if not prototype_path.exists():
            prototype_path = prototype_dir / prototype_path.name
            mean_face_path = prototype_dir / mean_face_path.name

        if not prototype_path.exists():
            raise FileNotFoundError(
                f"Could not find prototype file: {prototype_path}"
            )

        prototypes[key] = {
            "prototype": np.load(prototype_path),
            "mean_face": np.load(mean_face_path) if mean_face_path.exists() else None,
            "n": int(row["n"]),
            "map_paths": [],
        }

    return prototypes


def get_attention_prototypes(
    prototype_dir: str | Path,
    force_rebuild: bool = False,
) -> dict:
    """
    Load saved prototypes if they exist, otherwise build and save them.

    Parameters
    ----------
    force_rebuild:
        If True, ignore saved prototypes and rebuild from individual maps.
    """
    prototype_dir = Path(prototype_dir)
    summary_path = prototype_dir / "prototype_summary.csv"

    if summary_path.exists() and not force_rebuild:
        print(f"Loading saved prototypes from: {prototype_dir}")
        return load_saved_prototypes(prototype_dir)


# ============================================================
# 8. Prototype similarity metrics
# ============================================================

def flatten_prob_map(
    relevance: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Flatten map and convert to probability vector.
    """
    relevance = np.maximum(relevance.astype(np.float64), 0)
    relevance = relevance + eps
    relevance = relevance / relevance.sum()
    return relevance.ravel()


def prototype_similarity(
    proto_a: np.ndarray,
    proto_b: np.ndarray,
) -> dict:
    """
    Compute similarity / distance metrics between two prototypes.

    Uses:
        cosine similarity
        Pearson correlation
        Jensen-Shannon distance
    """
    from scipy.spatial.distance import cosine, jensenshannon

    a = flatten_prob_map(proto_a)
    b = flatten_prob_map(proto_b)

    cosine_similarity = 1.0 - cosine(a, b)
    pearson_correlation = np.corrcoef(a, b)[0, 1]
    js_distance = jensenshannon(a, b)

    return {
        "cosine_similarity": float(cosine_similarity),
        "pearson_correlation": float(pearson_correlation),
        "js_distance": float(js_distance),
    }


def compare_models_by_emotion(
    prototypes: dict,
    model_a: str,
    model_b: str,
    sigmas: list[int | float],
    emotions: list[str] = EMOTIONS,
) -> pd.DataFrame:
    """
    Compare prototype similarity between two models for each sigma and emotion.
    """
    rows = []

    for sigma in sigmas:
        for emotion in emotions:
            key_a = (model_a, sigma, emotion)
            key_b = (model_b, sigma, emotion)

            if key_a not in prototypes or key_b not in prototypes:
                continue

            proto_a = prototypes[key_a]["prototype"]
            proto_b = prototypes[key_b]["prototype"]

            metrics = prototype_similarity(proto_a, proto_b)

            rows.append({
                "sigma": sigma,
                "emotion": emotion,
                "model_a": model_a,
                "model_b": model_b,
                "n_a": prototypes[key_a]["n"],
                "n_b": prototypes[key_b]["n"],
                **metrics,
            })

    return pd.DataFrame(rows)


def compare_blur_stability(
    prototypes: dict,
    models: list[str],
    reference_sigma: int | float,
    test_sigmas: list[int | float],
    emotions: list[str] = EMOTIONS,
) -> pd.DataFrame:
    """
    Compare each sigma's prototype to a reference sigma, usually sigma=0.

    This measures whether each model preserves its emotion-specific relevance
    allocation as blur increases.
    """
    rows = []

    for model in models:
        for emotion in emotions:
            ref_key = (model, reference_sigma, emotion)

            if ref_key not in prototypes:
                continue

            ref_proto = prototypes[ref_key]["prototype"]

            for sigma in test_sigmas:
                test_key = (model, sigma, emotion)

                if test_key not in prototypes:
                    continue

                test_proto = prototypes[test_key]["prototype"]
                metrics = prototype_similarity(ref_proto, test_proto)

                rows.append({
                    "model": model,
                    "emotion": emotion,
                    "reference_sigma": reference_sigma,
                    "sigma": sigma,
                    "n_ref": prototypes[ref_key]["n"],
                    "n_test": prototypes[test_key]["n"],
                    **metrics,
                })

    return pd.DataFrame(rows)


def emotion_prototype_separability(
    prototypes: dict,
    models: list[str],
    sigmas: list[int | float],
    emotions: list[str] = EMOTIONS,
) -> pd.DataFrame:
    """
    Compute pairwise similarity between emotion prototypes within each model/sigma.

    Lower similarity between different emotions implies more class-specific
    spatial relevance allocation.
    """
    rows = []

    for model in models:
        for sigma in sigmas:
            for i, emotion_a in enumerate(emotions):
                for emotion_b in emotions[i + 1:]:
                    key_a = (model, sigma, emotion_a)
                    key_b = (model, sigma, emotion_b)

                    if key_a not in prototypes or key_b not in prototypes:
                        continue

                    proto_a = prototypes[key_a]["prototype"]
                    proto_b = prototypes[key_b]["prototype"]

                    metrics = prototype_similarity(proto_a, proto_b)

                    rows.append({
                        "model": model,
                        "sigma": sigma,
                        "emotion_a": emotion_a,
                        "emotion_b": emotion_b,
                        "n_a": prototypes[key_a]["n"],
                        "n_b": prototypes[key_b]["n"],
                        **metrics,
                    })

    return pd.DataFrame(rows)


def summarize_prototype_results(
    stability_df: pd.DataFrame,
    separability_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize stability and separability results.
    """
    stability_summary = (
        stability_df
        .groupby(["model", "sigma"])
        .agg(
            mean_stability_cosine=("cosine_similarity", "mean"),
            mean_stability_js=("js_distance", "mean"),
            mean_stability_corr=("pearson_correlation", "mean"),
        )
        .reset_index()
    )

    separability_summary = (
        separability_df
        .groupby(["model", "sigma"])
        .agg(
            mean_between_emotion_cosine=("cosine_similarity", "mean"),
            mean_between_emotion_js=("js_distance", "mean"),
            mean_between_emotion_corr=("pearson_correlation", "mean"),
        )
        .reset_index()
    )

    return stability_summary, separability_summary


# ------------------------------------------------------------
# Step 2: Build true-label prototypes from correct trials
# ------------------------------------------------------------

pool_dir = '/WITH_POOLING' if POOLING else ''

prototypes_true_correct = get_attention_prototypes(
    prototype_dir=f'attention_prototypes/NATIVE_RESOLUTION{pool_dir}/true_label_correct'
)
# prototypes_true_correct = get_attention_prototypes(
#     prototype_dir='attention_prototypes/true_label_correct'
# )
#
# prototypes_pred_wrong = get_attention_prototypes(
#     prototype_dir='attention_prototypes/pred_label_wrong'
# )
#
# prototypes_true_wrong = get_attention_prototypes(
#     prototype_dir='attention_prototypes/true_label_wrong'
# )

prototypes_pred_all = get_attention_prototypes(
    prototype_dir=f'attention_prototypes/NATIVE_RESOLUTION{pool_dir}/pred_label_all'
)

model_sigma_rows = [
    # ('RAF_blur0_pretrained0-8_freeze', 0),
    ('RAF_blur0_freeze', 0),
    ('RAF_blur0_pretrained0-8_freeze', 0),
    ('RAF_blur8_pretrained0-8_freeze_fix_projs', 8),
    ('RAF_blur8_freeze', 8),
    # ('RAF_blur16_pretrained0-8_freeze', 16),
    # ('RAF_blur16_pretrained8_freeze', 16),
]

plot_prototype_grid(
    prototypes=prototypes_true_correct,
    model_sigma_rows=model_sigma_rows,
    emotions=EMOTIONS,
    output_path=f"figures/prototypes_true_label_correct{'_with_pool' if POOLING else ''}.png",
    title="Emotion-specific attention prototypes in correct trials",
    show_on_mean_face=True
)

# plot_prototype_grid(
#     prototypes=prototypes_true_wrong,
#     model_sigma_rows=model_sigma_rows,
#     emotions=EMOTIONS,
#     output_path="figures/prototypes_true_label_wrong.png",
#     title="Emotion-specific attention prototypes for true labels in wrong trials",
#     show_on_mean_face=True,
# )
#
# plot_prototype_grid(
#     prototypes=prototypes_pred_wrong,
#     model_sigma_rows=model_sigma_rows,
#     emotions=EMOTIONS,
#     output_path="figures/prototypes_pred_label_wrong.png",
#     title="Emotion-specific attention prototypes for predicted labels in wrong trials",
#     show_on_mean_face=True,
# )

plot_prototype_grid(
    prototypes=prototypes_pred_all,
    model_sigma_rows=model_sigma_rows,
    emotions=EMOTIONS,
    output_path=f"figures/prototypes_pred_label_all{'_with_pool' if POOLING else ''}.png",
    title="Emotion-specific attention prototypes for predicted labels in all trials",
    show_on_mean_face=True,
    subset='ALL',
)

plt.show(block=True)


