"""Compare matched image relevance maps; run with --help for CLI usage."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# 1. Map utilities
# ============================================================

if __package__:
    from .wasserstein import (load_attention_map, normalize_map, resize_map,
                              prepare_map, make_2d_cost_matrix, wasserstein_distance_2d)
else:
    from wasserstein import (load_attention_map, normalize_map, resize_map,
                             prepare_map, make_2d_cost_matrix, wasserstein_distance_2d)


def preprocess_map_for_wasserstein(path, output_size=None, clip_negative=True):
    return prepare_map(load_attention_map(path), output_size, clip_negative)


def compare_models_wasserstein(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    output_size: tuple[int, int] | None = None,
    target_mode: str = "true_label",
    require_same_prediction_status: bool = False,
    clip_negative: bool = True,
    normalize_cost: bool = True,
    sigma_a: float | None = None,
    sigma_b: float | None = None,
    map_root: str | Path | None = None,
) -> pd.DataFrame:
    """Compare maps paired by image, target, and sigma.

    To compare different conditions, supply BOTH sigma_a and sigma_b;
    pairing then uses image and target after condition selection.
    Duplicate keys, invalid maps, and empty selections raise errors.
    Native resolution is preserved unless output_size is supplied.
    Relative paths are resolved against map_root (cwd by default).
    pred_label mode only pairs models predicting the same target class.
    Normalized distances use diagonal units, in [0, 1].
    """
    required_cols = {
        "image_id",
        "model",
        "sigma",
        "true_label",
        "pred_label",
        "target_label",
        "correct",
        "map_path",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    work = df.copy()

    if target_mode == "true_label":
        work = work[work["target_label"] == work["true_label"]]
    elif target_mode == "pred_label":
        work = work[work["target_label"] == work["pred_label"]]
    elif target_mode == "all":
        pass
    else:
        raise ValueError("target_mode must be one of: 'true_label', 'pred_label', 'all'")

    df_a = work[work["model"] == model_a].copy()
    df_b = work[work["model"] == model_b].copy()

    if (sigma_a is None) != (sigma_b is None):
        raise ValueError("Supply both sigma_a and sigma_b, or neither.")
    pair_cols = ["image_id", "target_label"]
    if sigma_a is not None:
        df_a = df_a[df_a["sigma"] == sigma_a]
        df_b = df_b[df_b["sigma"] == sigma_b]
    else:
        pair_cols.append("sigma")
    for name, frame in (("A", df_a), ("B", df_b)):
        if frame.empty:
            raise ValueError(f"No maps selected for model {name}.")
        if frame[pair_cols].isna().any().any() or frame.duplicated(pair_cols).any():
            raise ValueError(f"Missing or duplicate pairing keys for model {name}: {pair_cols}")
        if not frame["correct"].map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError("correct must contain boolean values.")
    paired = df_a.merge(df_b, on=pair_cols, suffixes=("_a", "_b"),
                           how="inner", validate="one_to_one")
    if require_same_prediction_status:
        paired = paired[paired["correct_a"] == paired["correct_b"]]
    if paired.empty:
        raise ValueError("No matched maps remain after selection.")
    if not (paired["true_label_a"] == paired["true_label_b"]).all():
        raise ValueError("Matched images have conflicting true labels.")
    map_root = Path(map_root) if map_root is not None else Path.cwd()

    rows = []

    for _, row in tqdm(
        paired.iterrows(),
        total=len(paired),
        desc=f"Wasserstein: {model_a} vs {model_b}",
    ):
        try:
            map_a = preprocess_map_for_wasserstein(
                map_root / row["map_path_a"],
                output_size=output_size,
                clip_negative=clip_negative,
            )

            map_b = preprocess_map_for_wasserstein(
                map_root / row["map_path_b"],
                output_size=output_size,
                clip_negative=clip_negative,
            )

            emd = wasserstein_distance_2d(
                map_a,
                map_b,
                normalize_cost=normalize_cost,
            )

            rows.append({
                "image_id": row["image_id"],
                "target_label": row["target_label"],

                "model_a": model_a,
                "model_b": model_b,

                "sigma_a": row.get("sigma_a", row.get("sigma")),
                "sigma_b": row.get("sigma_b", row.get("sigma")),

                "true_label_a": row["true_label_a"],
                "true_label_b": row["true_label_b"],

                "pred_label_a": row["pred_label_a"],
                "pred_label_b": row["pred_label_b"],

                "correct_a": row["correct_a"],
                "correct_b": row["correct_b"],

                "pred_confidence_a": row.get("pred_confidence_a", np.nan),
                "pred_confidence_b": row.get("pred_confidence_b", np.nan),

                "true_confidence_a": row.get("true_confidence_a", np.nan),
                "true_confidence_b": row.get("true_confidence_b", np.nan),

                "wasserstein_distance": emd,
                "distance_scale": "diagonal" if normalize_cost else "grid_pixels",
                "transport_height": map_a.shape[0],
                "transport_width": map_a.shape[1],

                "map_path_a": map_root / row["map_path_a"],
                "map_path_b": map_root / row["map_path_b"],
            })

        except (ValueError, OSError) as exc:
            raise ValueError(
                f"Cannot compare image_id={row['image_id']}, "
                f"target={row['target_label']}: {exc}"
            ) from exc

    return pd.DataFrame(rows)



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calculate matched spatial Wasserstein-1 distances.")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--sigma-a", type=float)
    parser.add_argument("--sigma-b", type=float)
    parser.add_argument("--size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--target-mode", choices=["true_label", "pred_label", "all"], default="true_label")
    parser.add_argument("--map-root", type=Path, help="Root of relative map paths; default: metadata directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_models_wasserstein(
        pd.read_csv(args.metadata), args.model_a, args.model_b,
        output_size=args.size, target_mode=args.target_mode,
        sigma_a=args.sigma_a, sigma_b=args.sigma_b,
        map_root=args.map_root or args.metadata.resolve().parent,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} distances to {args.output}")


if __name__ == "__main__":
    main()
