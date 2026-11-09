# Attention-map distances

The standalone analysis works with saved `.npy` maps. It does not require
APViT, model checkpoints, CUDA, or the model-generation dependencies.
From the repository root, install into a Python 3.9 environment:

```powershell
.\.venv\Scripts\python -m pip install -r compare_models/requirements.txt
```

## Compare two arrays

```python
import numpy as np
from compare_models.wasserstein import prepare_map, wasserstein_distance_2d

a = prepare_map(np.load("map_a.npy"))
b = prepare_map(np.load("map_b.npy"))
distance = wasserstein_distance_2d(a, b)
```

Maps must have matching spatial shapes. To explicitly resample different
shapes, pass the same `output_size=(14, 14)` to both `prepare_map` calls.
The default preserves native resolution; use a small transport grid for
large maps because the dense cost matrix has `(height * width) ** 2` entries.

The metric is exact spatial Wasserstein-1 **on the prepared grid**, using
Euclidean costs divided by that grid's diagonal. Valid distances lie in
`[0, 1]`: identical distributions have distance 0; opposite-corner point
masses have distance 1. It measures spatial allocation, not total relevance.
Negative values are clipped before resizing and normalization; set
`clip_negative=False` to reject them instead. Empty, nonfinite, and all-zero
maps are errors. Resampling changes the distributions and can change results.

## Compare models using metadata

For configured runs, edit `compare_models/wasserstein_config.py`, then run
from the repository root:

```powershell
.\.venv\Scripts\python -m compare_models.calc_wasserstein_dist
```

The `CONFIG` dictionary selects the two model names, sigma for each model,
target mode (`true_label`, `pred_label`, or `all`), transport size, metadata
CSV, map directory, and output directory. Paths are relative to the config
file. `map_root` resolves the metadata's `map_path` entries; it should match
the directory those paths were recorded relative to. Set both sigmas to
`None` to compare matching conditions, or specify both to select conditions.

Each run creates a unique UTC timestamp directory under `output_dir` with
`distances.csv`, `distances.config.json` (resolved paths and effective settings),
and a copy of `wasserstein_config.py`. You can select another configuration
with `--config path/to/wasserstein_config.py`. Configuration mode does not
accept additional command-line overrides. The explicit CLI below remains
available and also saves a `.config.json` alongside its output CSV.

The metadata CSV must contain:

`image_id, model, sigma, true_label, pred_label, target_label, correct, map_path`

`correct` must contain boolean values. Each image/target/condition must be
unique per model. Without condition selectors, the script pairs by
`image_id + target_label + sigma`. Only matched observations are analyzed.
Missing models, empty matches, duplicate keys, conflicting true labels,
missing maps, and invalid maps raise errors rather than silently skipping rows.

```powershell
.\.venv\Scripts\python -m compare_models.calc_wasserstein_dist metadata.csv --model-a MODEL_A --model-b MODEL_B --map-root compare_models --output results/distances.csv
```

By default relative map paths are resolved against the metadata CSV's
directory. Use `--map-root` for CSVs whose paths were recorded relative to a
different working directory (the original experiments used `compare_models`).

To compare two explicit blur conditions, supply both selectors:

```powershell
.\.venv\Scripts\python -m compare_models.calc_wasserstein_dist metadata.csv --model-a MODEL_A --model-b MODEL_B --sigma-a 8 --sigma-b 0 --map-root compare_models --output results/cross_blur.csv
```

Use `--size 14 14` to explicitly resample both maps. Default `--target-mode
true_label` explains the true class, even for incorrect predictions.
`pred_label` selects each model's predicted-class map but only matches pairs
with the same target class. `all` compares every matching target class.
Multiple target maps per image are dependent observations; select one target
per image before applying image-level paired statistical tests.

Output includes image, target, model names, `sigma_a`, `sigma_b`, correctness,
map paths, `wasserstein_distance`, `distance_scale`, and transport dimensions.

## Compare two sets of distances

```powershell
.\.venv\Scripts\python -m compare_models.comp_wasserstein_dist results/low.csv results/high.csv --all-correct --output-dir results/paired
```

This writes `paired_distances.csv` and `paired_tests.csv`. Delta is always
**distance table A minus distance table B**. `--alternative less` tests
whether A is smaller; `greater` tests whether A is larger; the default is
`two-sided`. The correctness filter requires both predictions in each table
to be correct, regardless of whether this involves three or four distinct
predictions. The old `filter_all_three_correct` name remains an API alias.

If tables contain multiple conditions, filter them first or pass suitable
`--pair-cols` (for example `image_id target_label sigma_a sigma_b`).
Duplicate pairing keys are rejected. New-format tables must agree on scale
and transport size. Legacy tables without these fields are accepted together;
their comparability must be established from the generating experiment.
Permutation testing uses bounded batches and returns nonzero Monte Carlo
p-values. All-zero paired differences return p=1.

## Other analyses and existing outputs

| Script | Question |
| --- | --- |
| `create_attn_maps.py` | Generate class-specific relevance maps from models; requires the original APViT setup. |
| `build_prototypes.py` | Average maps by model, blur condition, and target emotion. |
| `calc_prototype_separability.py` | How far apart are emotion prototypes within a model? |
| `calc_image_separability.py` | How do within-emotion and between-emotion image distances compare? |
| `plot_wasserstein_comparison.py` | Plot saved paired distances; legacy runnable example is guarded against import execution. |

The distance, statistical comparison, separability, and paired plotting
modules can be imported without starting experiments. The separability
scripts now use the same diagonal-normalized cost as the direct comparison.
Their `normalize_coordinates=False` option uses raw grid distances.

**Historical CSVs and prototypes are left unchanged.** Earlier separability
results used axis-normalized costs (maximum `sqrt(2)`); on square grids the
new distances equal those old distances divided by `sqrt(2)` when preparation
is otherwise identical. Recompute analyses before mixing old and new results.
Image pairwise caches have a metric version so old scales cannot be reused.
The checkout contains saved prototypes and distance results but not the
per-image source maps or metadata needed to regenerate the original runs.

Run regression checks with:

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```
