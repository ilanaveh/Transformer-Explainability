"""Edit these options, then run python -m compare_models.calc_wasserstein_dist.

Relative paths below are relative to this file. Model names must match metadata.
"""

CONFIG = {
    "metadata": "attention_maps/NATIVE_RESOLUTION/WITH_POOLING/metadata_all_models_native.csv",
    "model_a": "MODEL_A",
    "model_b": "MODEL_B",
    # Use two explicit sigmas, or None for both to match all equal sigmas.
    "sigma_a": None,
    "sigma_b": None,
    "target_mode": "true_label",  # true_label, pred_label, or all
    "size": None,  # Native resolution, or (14, 14) for a common grid.
    # Root against which map_path entries in the metadata are resolved.
    "map_root": ".",
    # Each run creates a unique subdirectory with distances and configuration copies.
    "output_dir": "attention_comparisons/wasserstein_runs",
}
