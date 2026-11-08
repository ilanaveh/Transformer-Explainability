from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch


# =============================================================================
# Configuration
# =============================================================================

METADATA_PATH = Path("attention_maps/NATIVE_RESOLUTION/metadata_all_models_native.csv")

MODELS = [
    "RAF_blur8_freeze",
    "RAF_blur8_pretrained0-8_freeze_fix_projs",
]

LABELS = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happy",
    "Sad",
    "Angry",
    "Neutral",
]

MODE = 'correct_vs_wrong'  # 'all_combs' - creates full confusion-matrix / 'correct_vs_wrong' - pools together all misclassified images.
TARGET_LABEL = 'pred_label'  # 'pred_label' / 'true_label'
AGG_WRONG_BY = 'pred_label'  # 'pred_label' / 'true_label'

OUTPUT_DIR = Path("figures")

FIGSIZE = (18, 18) if MODE == 'all_combs' else (18, 10)
DPI = 300
CMAP = "jet"

# When True, each individual map is independently min-max normalized before
# averaging. Usually leave False if the saved maps are already normalized.
NORMALIZE_EACH_MAP = False

# Use the same color limits for all 49 panels within each model figure.
USE_SHARED_COLOR_SCALE = False

INTERPOLATION = 'bilinear'  # 'bilinear' for smoothing, 'nearest' for keeping patch-resolution (relevant for native prototypes)
# =============================================================================
# Loading and preprocessing
# =============================================================================

def load_attention_map(map_path):
    """
    Load a saved .npy attention/relevance map and return a 2D float array.
    """
    map_path = Path(map_path).expanduser()

    # Resolve relative map paths relative to the metadata file.
    # if not map_path.is_absolute():
    #     map_path = METADATA_PATH.parent / map_path

    if not map_path.exists():
        raise FileNotFoundError(f"Map file not found: {map_path}")

    if map_path.suffix.lower() != ".npy":
        raise ValueError(f"Expected a .npy file, got: {map_path}")

    attention_map = np.load(map_path)
    attention_map = np.asarray(attention_map, dtype=float)
    attention_map = np.squeeze(attention_map)

    if attention_map.ndim != 2:
        raise ValueError(
            f"Expected a 2D map, but {map_path} has shape "
            f"{attention_map.shape}"
        )

    return attention_map


def normalize_map(attention_map, eps=1e-12):
    """
    Min-max normalize one map to [0, 1].
    """
    minimum = np.nanmin(attention_map)
    maximum = np.nanmax(attention_map)

    if maximum - minimum < eps:
        return np.zeros_like(attention_map, dtype=float)

    return (attention_map - minimum) / (maximum - minimum)


def load_face_image(
    path: str | Path,
    output_size: tuple[int, int] = (224, 224),
    grayscale: bool = False,
) -> np.ndarray:
    """
    Load and resize a face image.

    Returns
    -------
    image:
        If grayscale=False: [H, W, 3], float32, range [0, 1]
        If grayscale=True:  [H, W], float32, range [0, 1]
    """
    path = Path(path)

    if grayscale:
        img = Image.open(path).convert("L")
    else:
        img = Image.open(path).convert("RGB")

    out_h, out_w = output_size
    img = img.resize((out_w, out_h))

    arr = np.asarray(img).astype(np.float32) / 255.0

    return arr


def compute_average_map_and_face(map_paths, face_ids, use_median=False):
    """
    Load and average all valid maps in one true-label/predicted-label cell.

    Returns
    -------
    mean_map : np.ndarray or None
        Mean map across successfully loaded images.
    n_used : int
        Number of maps included in the mean.
    """
    loaded_maps = []
    loaded_images = []
    expected_shape = None

    for map_path, face_id in zip(map_paths, face_ids):
        attention_map = load_attention_map(map_path)

        if NORMALIZE_EACH_MAP:
            attention_map = normalize_map(attention_map)

        if expected_shape is None:
            expected_shape = attention_map.shape

        if attention_map.shape != expected_shape:
            raise ValueError(
                f"Map shape {attention_map.shape} does not match "
                f"expected shape {expected_shape}"
            )

        loaded_maps.append(attention_map)

        face_image = load_face_image(
            f'/home/projects/bagon/ilanaveh/data/RAF-DB/basic/Image/aligned_224/{face_id}.jpg',
            output_size=(112, 112),
            grayscale=False,)
        loaded_images.append(face_image)

    if not loaded_maps:
        return None, None, 0

    stacked_maps = np.stack(loaded_maps, axis=0)
    mean_map = np.nanmedian(stacked_maps, axis=0) if use_median else np.nanmean(stacked_maps, axis=0)

    stacked_images = np.stack(loaded_images, axis=0)
    mean_face = stacked_images.mean(axis=0)

    return mean_map, mean_face, len(loaded_maps)


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


# =============================================================================
# Figure creation
# =============================================================================

def create_model_figure(metadata, model_name, labels):
    """
    Create a 7 x 7 mean-map grid for one model.

    Columns:
        true_label

    Rows:
        pred_label


    Only rows satisfying target_label == TARGET_LABEL are included.

    """
    model_df = metadata.loc[
        metadata["model"].astype(str) == str(model_name)
    ].copy()

    if model_df.empty:
        raise ValueError(f"No eligible rows found for model: {model_name}")

    mean_maps = {}
    mean_faces = {}
    counts = {}

    if MODE == 'all_combs':
        for true_label in labels:
            for pred_label in labels:
                cell_key = (pred_label, true_label)
                cell_df = model_df.loc[
                    (model_df["true_label"].astype(str) == str(true_label))
                    & (model_df["pred_label"].astype(str) == str(pred_label))
                ]

                mean_map, mean_face, n_used = compute_average_map_and_face(
                    cell_df["map_path"], cell_df["image_id"]
                )

                mean_maps[cell_key] = mean_map
                counts[cell_key] = n_used
                mean_faces[cell_key] = mean_face

    elif MODE == 'correct_vs_wrong':
        if AGG_WRONG_BY == 'true_label':
            for true_label in labels:
                all_wrongs_df = pd.DataFrame()
                for pred_label in labels:
                    if pred_label == true_label:
                        cell_key = ('correct', true_label)

                        cell_df = model_df.loc[
                            (model_df["true_label"].astype(str) == str(true_label))
                            & (model_df["pred_label"].astype(str) == str(pred_label))
                            ]

                        mean_map, mean_face, n_used = compute_average_map_and_face(
                            cell_df["map_path"], cell_df["image_id"]
                        )

                        mean_maps[cell_key] = mean_map
                        counts[cell_key] = n_used
                        mean_faces[cell_key] = mean_face

                    else:
                        all_wrongs_df = pd.concat([
                                all_wrongs_df,
                                model_df.loc[
                                    (model_df["true_label"].astype(str) == str(true_label))
                                    & (model_df["pred_label"].astype(str) == str(pred_label))
                                ]
                        ]

                        )
                cell_key = ('wrong', true_label)
                mean_map, mean_face, n_used = compute_average_map_and_face(
                    all_wrongs_df["map_path"], all_wrongs_df["image_id"]
                )

                mean_maps[cell_key] = mean_map
                counts[cell_key] = n_used
                mean_faces[cell_key] = mean_face
        elif AGG_WRONG_BY == 'pred_label':
            for pred_label in labels:
                all_wrongs_df = pd.DataFrame()
                for true_label in labels:
                    if pred_label == true_label:
                        cell_key = ('correct', true_label)

                        cell_df = model_df.loc[
                            (model_df["true_label"].astype(str) == str(true_label))
                            & (model_df["pred_label"].astype(str) == str(pred_label))
                            ]

                        mean_map, mean_face, n_used = compute_average_map_and_face(
                            cell_df["map_path"], cell_df["image_id"]
                        )

                        mean_maps[cell_key] = mean_map
                        counts[cell_key] = n_used
                        mean_faces[cell_key] = mean_face

                    else:
                        all_wrongs_df = pd.concat([
                            all_wrongs_df,
                            model_df.loc[
                                (model_df["true_label"].astype(str) == str(true_label))
                                & (model_df["pred_label"].astype(str) == str(pred_label))
                                ]
                        ]

                        )
                cell_key = ('wrong', pred_label)
                mean_map, mean_face, n_used = compute_average_map_and_face(
                    all_wrongs_df["map_path"], all_wrongs_df["image_id"]
                )

                mean_maps[cell_key] = mean_map
                counts[cell_key] = n_used
                mean_faces[cell_key] = mean_face

    populated_maps = [
        attention_map
        for attention_map in mean_maps.values()
        if attention_map is not None
    ]

    if not populated_maps:
        raise ValueError(
            f"No attention maps could be loaded for model: {model_name}"
        )

    if USE_SHARED_COLOR_SCALE:
        vmin = min(
            np.nanmin(attention_map)
            for attention_map in populated_maps
        )
        vmax = max(
            np.nanmax(attention_map)
            for attention_map in populated_maps
        )
    else:
        vmin = None
        vmax = None

    n_labels = len(labels)

    fig, axes = plt.subplots(
        nrows=n_labels if MODE == 'all_combs' else 2,
        ncols=n_labels,
        figsize=FIGSIZE,
        squeeze=False,
    )

    last_image = None

    rows = labels if MODE == 'all_combs' else ['correct', 'wrong']

    for row_idx, row_label in enumerate(rows):  # row_label is pred_label if MODE='all_combs', or correct/wrong if MODE='correct_vs_wrong'
        for col_idx, true_label in enumerate(labels):
            ax = axes[row_idx, col_idx]

            cell_key = (row_label, true_label)
            mean_map = mean_maps[cell_key]
            mean_face = mean_faces[cell_key]
            n_used = counts[cell_key]

            if mean_map is None:
                ax.set_facecolor("0.92")
                ax.text(
                    0.5,
                    0.5,
                    "No images",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
            else:
                ax.imshow(np.clip(mean_face, 0, 1))

                if mean_map.shape != mean_face.shape[:2]:
                    display_proto = torch.nn.functional.interpolate(
                        torch.from_numpy(mean_map).reshape(1, 1, 14, 14),
                        scale_factor=mean_face.shape[0] / mean_map.shape[0],
                        mode=INTERPOLATION).reshape(mean_face.shape[:2]).numpy()

                alpha_map = make_alpha_map(
                    display_proto,
                    min_alpha=0.3,
                    max_alpha=1,
                    gamma=1.5,
                )
                last_image = ax.imshow(
                    display_proto,
                    cmap=CMAP,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=alpha_map
                )

            ax.set_title(f"n = {n_used}", fontsize=9, pad=3)
            ax.set_xticks([])
            ax.set_yticks([])

            # Emphasize the correctly classified cells.
            if row_label == true_label:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2.5)
                    spine.set_edgecolor("black")
            else:
                for spine in ax.spines.values():
                    spine.set_visible(False)

            # Column labels: true labels
            if row_idx == 0:
                ax.set_xlabel(
                    str(true_label),
                    fontsize=10,
                    fontweight="bold",
                    labelpad=8,
                )
                ax.xaxis.set_label_position("top")

            # Row labels: predicted labels
            if col_idx == 0:
                ax.set_ylabel(
                    str(row_label),
                    fontsize=10,
                    fontweight="bold",
                    labelpad=8,
                )

    fig.suptitle(
        f'{model_name} (Label: {TARGET_LABEL})',
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    fig.text(
        0.5,
        0.012,
        AGG_WRONG_BY,
        ha="center",
        fontsize=13,
        fontweight="bold",
    )

    if MODE == 'all_combs':
        fig.text(
            0.008,
            0.5,
            "Predicted label",
            va="center",
            rotation=90,
            fontsize=13,
            fontweight="bold",
        )

    if USE_SHARED_COLOR_SCALE and last_image is not None:
        fig.subplots_adjust(
            left=0.07,
            right=0.91,
            bottom=0.055,
            top=0.92,
            wspace=0.08,
            hspace=0.18,
        )

        colorbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
        colorbar = fig.colorbar(last_image, cax=colorbar_ax)
        colorbar.set_label("Mean relevance", fontsize=11)

    else:
        fig.subplots_adjust(
            left=0.07,
            right=0.98,
            bottom=0.055,
            top=0.92,
            wspace=0.08,
            hspace=0.18,
        )

    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(METADATA_PATH)

    required_columns = [
        "image_id",
        "model",
        "sigma",
        "true_label",
        "pred_label",
        "target_label",
        "target_class_idx",
        "correct",
        "pred_confidence",
        "true_confidence",
        "map_path",
        "normalized_before_saving",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in metadata.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing columns: {missing_columns}\n"
            f"Available columns: {metadata.columns.tolist()}"
        )

    # Remove rows without a saved map path.
    metadata = metadata.dropna(subset=["map_path"]).copy()

    # Critical filter:
    # retain only maps generated using the model's predicted class as target.
    metadata = metadata.loc[
        metadata["target_label"].astype(str)
        == metadata[TARGET_LABEL].astype(str)
    ].copy()

    print(
        f"Rows remaining after target_label == TARGET_LABEL filtering: "
        f"{len(metadata)}"
    )

    for model_name in MODELS:
        model_row_count = (
            metadata["model"].astype(str) == str(model_name)
        ).sum()

        print(
            f"\nCreating figure for {model_name}: "
            f"{model_row_count} eligible metadata rows"
        )

        fig = create_model_figure(
            metadata=metadata,
            model_name=model_name,
            labels=LABELS
        )

        png_path = (
            OUTPUT_DIR
            / f"prototypes_{MODE}_{TARGET_LABEL}_agg_wrong_by_{AGG_WRONG_BY}_{model_name}.png"
        )

        fig.savefig(
            png_path,
            dpi=DPI,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()