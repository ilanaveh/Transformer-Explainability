"""
27/5/26
Assumes file "attention_maps(/NATIVE_RESOLUTION)/metadata_all_models(_native).csv" exists (created by
create_attn_maps.py).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm

from create_attn_maps import ensure_2d_map, resize_map_numpy, normalize_relevance_map


POOLING = True


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
    img = img.resize((out_w, out_h), Image.BILINEAR)

    arr = np.asarray(img).astype(np.float32) / 255.0

    return arr


def load_map(path: str | Path) -> np.ndarray:
    """
    Load saved map from .npy.
    """
    relevance = np.load(path)
    return ensure_2d_map(relevance)


def preprocess_saved_map(
    path: str | Path,
    output_size: tuple[int, int] = (224, 224),
    normalize: bool = True,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Load, resize, and normalize a saved attention/relevance map.

    Even if maps were normalized before saving, normalizing again is harmless
    and keeps this pipeline robust.
    """
    relevance = load_map(path)
    relevance = resize_map_numpy(relevance, output_size)

    if normalize:
        relevance = normalize_relevance_map(
            relevance,
            clip_negative=clip_negative,
        )

    return relevance


def build_attention_prototypes_with_mean_faces(
    df: pd.DataFrame,
    # output_size: tuple[int, int] = (224, 224),
    proto_output_size: tuple[int, int] = (224, 224),
    face_output_size: tuple[int, int] = (224, 224),
    only_correct: bool = False,
    only_wrong: bool = False,
    target_mode: str = "true_label",
    group_cols: tuple[str, ...] = ("model", "sigma", "target_label"),
    image_col: str = "image_path",
    map_col: str = "map_path",
    grayscale_faces: bool = False,
) -> dict:
    """
    Build attention/relevance prototypes and corresponding mean face images.

    Each prototype gets:
        - prototype: average relevance map
        - mean_face: average of all face images used for that prototype
        - n: number of samples
        - map_paths
        - image_paths

    Important:
        The mean face is computed over the same rows used to compute the prototype.
    """
    work = df.copy()

    if image_col not in work.columns:
        raise ValueError(f"DataFrame must contain image column: {image_col}")

    if only_correct:
        work = work[work["correct"] == True]
    elif only_wrong:
        work = work[work["correct"] == False]

    if target_mode == "true_label":
        work = work[work["target_label"] == work["true_label"]]
    elif target_mode == "pred_label":
        work = work[work["target_label"] == work["pred_label"]]
    elif target_mode == "all":
        pass
    else:
        raise ValueError(
            "target_mode must be one of: 'true_label', 'pred_label', 'all'"
        )

    prototypes = {}
    for group_key, group_df in tqdm(
            work.groupby(list(group_cols)),
            desc="Building prototypes",
    ):

        maps = []
        images = []
        map_paths = []
        image_paths = []

        for _, row in group_df.iterrows():
            try:
                relevance = preprocess_saved_map(
                    row[map_col],
                    output_size=proto_output_size,
                    normalize=True,
                    clip_negative=True,
                )

                face = load_face_image(
                    f'/home/projects/bagon/ilanaveh/data/RAF-DB/basic/Image/aligned_224/{row[image_col]}.jpg',
                    output_size=face_output_size,
                    grayscale=grayscale_faces,
                )

                maps.append(relevance)
                images.append(face)
                map_paths.append(row[map_col])
                image_paths.append(row[image_col])

            except Exception as exc:
                print(
                    f"Skipping map={row.get(map_col)} image={row.get(image_col)} "
                    f"because: {exc}"
                )

        if len(maps) == 0:
            continue

        stacked_maps = np.stack(maps, axis=0)
        prototype = stacked_maps.mean(axis=0)
        prototype = normalize_relevance_map(prototype)

        stacked_images = np.stack(images, axis=0)
        mean_face = stacked_images.mean(axis=0)

        prototypes[group_key] = {
            "prototype": prototype,
            "mean_face": mean_face,
            "n": len(maps),
            "map_paths": map_paths,
            "image_paths": image_paths,
        }

    return prototypes


# ============================================================
# 6. Save prototypes
# ============================================================
def save_prototypes(
    prototypes: dict,
    output_dir: str | Path,
    suf: str = '',
) -> pd.DataFrame:
    """
    Save prototype maps as .npy files and write a summary CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for key, value in prototypes.items():
        key_str = "__".join(str(x) for x in key)
        key_str = (key_str + f'_{suf}') if suf is not None else key_str

        prototype_path = output_dir / f"prototype__{key_str}.npy"

        np.save(
            prototype_path,
            value["prototype"].astype(np.float32),
        )

        rows.append({
            "key": key,
            "prototype_path": str(prototype_path),
            "n": value["n"],
        })

    summary_df = pd.DataFrame(rows)

    summary_path = output_dir / "prototype_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return summary_df


def save_prototypes_with_mean_faces(
    prototypes: dict,
    output_dir: str | Path,
):
    """
    Save:
        - prototype relevance maps as .npy
        - mean face images as .npy
        - overlay visualizations as .png
        - summary csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prototype_map_dir = output_dir / "prototype_maps"
    mean_face_dir = output_dir / "mean_faces"

    prototype_map_dir.mkdir(exist_ok=True)
    mean_face_dir.mkdir(exist_ok=True)

    rows = []

    for key, value in prototypes.items():
        key_str = "__".join(str(x) for x in key)

        prototype_path = prototype_map_dir / f"prototype__{key_str}.npy"
        mean_face_path = mean_face_dir / f"mean_face__{key_str}.npy"

        np.save(prototype_path, value["prototype"])

        if "mean_face" in value and value["mean_face"] is not None:
            np.save(mean_face_path, value["mean_face"])
        else:
            mean_face_path = None

        rows.append({
            "key": key,
            "prototype_path": str(prototype_path),
            "mean_face_path": str(mean_face_path) if mean_face_path is not None else None,
            "n": value["n"],
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "prototype_summary.csv", index=False)

    return summary_df
# ------------------------------------------------------------
# Step 4: Compare model prototypes by emotion
# ------------------------------------------------------------
"""
model_similarity_df = compare_models_by_emotion(
    prototypes=prototypes_true_correct,
    model_a="mixed_pretrain",
    model_b="lowres_pretrain",
    sigmas=[0, 2, 4, 6, 8],
    emotions=EMOTIONS,
)

model_similarity_df.to_csv(
    "attention_prototypes/model_similarity_true_correct.csv",
    index=False,
)

print(model_similarity_df)
"""


# ------------------------------------------------------------
# Step 5: Compare stability across blur
# ------------------------------------------------------------

"""
stability_df = compare_blur_stability(
    prototypes=prototypes_true_correct,
    models=["mixed_pretrain", "lowres_pretrain"],
    reference_sigma=0,
    test_sigmas=[2, 4, 6, 8],
    emotions=EMOTIONS,
)

stability_df.to_csv(
    "attention_prototypes/blur_stability_true_correct.csv",
    index=False,
)

print(stability_df)
"""


# ------------------------------------------------------------
# Step 6: Compare emotion separability
# ------------------------------------------------------------

"""
separability_df = emotion_prototype_separability(
    prototypes=prototypes_true_correct,
    models=["mixed_pretrain", "lowres_pretrain"],
    sigmas=[0, 2, 4, 6, 8],
    emotions=EMOTIONS,
)

separability_df.to_csv(
    "attention_prototypes/emotion_separability_true_correct.csv",
    index=False,
)

print(separability_df)
"""
# ------------------------------------------------------------
# Step 7: Summaries
# ------------------------------------------------------------

"""
stability_summary, separability_summary = summarize_prototype_results(
    stability_df=stability_df,
    separability_df=separability_df,
)

stability_summary.to_csv(
    "attention_prototypes/blur_stability_summary.csv",
    index=False,
)

separability_summary.to_csv(
    "attention_prototypes/emotion_separability_summary.csv",
    index=False,
)

print("Stability summary")
print(stability_summary)

print("Separability summary")
print(separability_summary)
"""

if __name__ == "__main__":
    print("Building prototypes from individual maps...")
    pool_dir = '/WITH_POOLING' if POOLING else ''
    df = pd.read_csv(f"attention_maps/NATIVE_RESOLUTION{pool_dir}/metadata_all_models_native.csv")
    prototype_dir = f'attention_prototypes/NATIVE_RESOLUTION{pool_dir}'
    # output_size = (112, 112)
    proto_output_size = (14, 14)
    face_output_size = (112, 112)

    prototypes_true_correct = build_attention_prototypes_with_mean_faces(
        df=df,
        proto_output_size=proto_output_size,
        face_output_size=face_output_size,
        only_correct=True,
        only_wrong=False,
        target_mode="true_label",
        group_cols=("model", "sigma", "target_label"),
        image_col='image_id'
    )
    #
    # prototypes_true_wrong = build_attention_prototypes_with_mean_faces(
    #     df=df,
    #     output_size=output_size,
    #     only_correct=False,
    #     only_wrong=True,
    #     target_mode="true_label",
    #     group_cols=("model", "sigma", "target_label"),
    #     image_col='image_id'
    # )
    #
    # prototypes_pred_wrong = build_attention_prototypes_with_mean_faces(
    #     df=df,
    #     output_size=output_size,
    #     only_correct=False,
    #     only_wrong=True,
    #     target_mode="pred_label",
    #     group_cols=("model", "sigma", "target_label"),
    #     image_col='image_id'
    # )

    prototypes_pred_all = build_attention_prototypes_with_mean_faces(
        df=df,
        proto_output_size=proto_output_size,
        face_output_size=face_output_size,
        only_correct=False,
        only_wrong=False,
        target_mode="pred_label",
        group_cols=("model", "sigma", "target_label"),
        image_col='image_id'
    )

    # save_prototypes(
    #     prototypes=prototypes_true_correct,
    #     output_dir=os.path.join(prototype_dir, 'true_label_correct'),
    #     # suf='14_14'
    # )

    save_prototypes_with_mean_faces(
        prototypes=prototypes_true_correct,
        output_dir=os.path.join(prototype_dir, 'true_label_correct'),
    )

    # save_prototypes_with_mean_faces(
    #     prototypes=prototypes_true_wrong,
    #     output_dir=os.path.join(prototype_dir, 'true_label_wrong'),
    # )

    # save_prototypes_with_mean_faces(
    #     prototypes=prototypes_pred_wrong,
    #     output_dir=os.path.join(prototype_dir, 'pred_label_wrong'),
    # )

    save_prototypes_with_mean_faces(
        prototypes=prototypes_pred_all,
        output_dir=os.path.join(prototype_dir, 'pred_label_all'),
    )

    print(f"Saved prototypes to: {prototype_dir}")

