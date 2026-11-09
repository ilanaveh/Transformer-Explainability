"""Edit these options, then run python -m compare_models.calc_wasserstein_dist.

Relative paths below are relative to this file. Model names must match metadata.
"""

CONFIG = {
    "metadata": "attention_maps/NATIVE_RESOLUTION/WITH_POOLING/metadata_all_models_native.csv",
    "model_a": "RAF_blur8_freeze",
    "model_b": "RAF_blur8_pretrained0-8_freeze_fix_projs",
    # Use two explicit sigmas, or None for both to match all equal sigmas.
    "sigma_a": 8,
    "sigma_b": 8,
    "target_mode": "pred_label",  # true_label, pred_label, or all
    "size": None,  # Native resolution, or (14, 14) for a common grid.
    # Root against which map_path entries in the metadata are resolved.
    "map_root": ".",
    # Each run creates a unique subdirectory with distances and configuration copies.
    "output_dir": "attention_comparisons/wasserstein_runs",
}
