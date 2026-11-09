"""Edit these options, then run python -m compare_models.calc_wasserstein_dist.

Relative paths below are relative to this file. Model names must match metadata.
"""

_COMMON = {
    "metadata": "attention_maps/NATIVE_RESOLUTION/WITH_POOLING/metadata_all_models_native.csv",
    # Use two explicit sigmas, or None for both to match all equal sigmas.
    "size": None,  # Native resolution, or (14, 14) for a common grid.
    # Root against which map_path entries in the metadata are resolved.
    "map_root": ".",
    # Each config writes to output_dir / config_name; reruns replace its results.
    "output_dir": "attention_comparisons/wasserstein_runs",
}

# Add configurations by copying an entry and giving it a descriptive, unique name.
# pred_label compares each model's own predicted-class map, including disagreements.
CONFIGS = {
    "mixed_vs_low_blur8_pred_label": {
        **_COMMON,
        "model_a": "RAF_blur8_freeze",
        "model_b": "RAF_blur8_pretrained0-8_freeze_fix_projs",
        "sigma_a": 8,
        "sigma_b": 8,
        "target_mode": "pred_label",
    },
    "mixed_vs_low_blur8_true_label": {
        **_COMMON,
        "model_a": "RAF_blur8_freeze",
        "model_b": "RAF_blur8_pretrained0-8_freeze_fix_projs",
        "sigma_a": 8,
        "sigma_b": 8,
        "target_mode": "true_label",
    },
    "mixed_blur8_vs_blur0_pred_label": {
        **_COMMON,
        "model_a": "RAF_blur0_pretrained0-8_freeze",
        "model_b": "RAF_blur8_pretrained0-8_freeze_fix_projs",
        "sigma_a": 0,
        "sigma_b": 8,
        "target_mode": "pred_label",
    },
}

DEFAULT_CONFIG = "mixed_vs_low_blur8_pred_label"
