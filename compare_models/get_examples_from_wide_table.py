"""
10/09/26
Use wide table with prediction info from mixed and low models (saved from create_overlayed_figures.py), and choose
example images with low/high confidence from each model.
Use the output examples for plotting attention-maps via plot_diff_map_single.py.

"""

import pandas as pd


POOLING = True
pool_dir = '/WITH_POOLING' if POOLING else ''
MIXED = "RAF_blur8_pretrained0-8_freeze_fix_projs"
LOW = "RAF_blur8_freeze"

df = pd.read_csv(f'attention_maps/NATIVE_RESOLUTION{pool_dir}/overlayed_figures/wide_pred_table.csv')

configs = {
    'mixed_wins':
        dict(models_correct=[MIXED],
             models_incorrect=[LOW],
             confidence_preferences=[
                 (f"{MIXED}_true_confidence", "high"),
                 (f"{LOW}_true_confidence", "low"),
             ],),
    'low_wins':
        dict(models_correct=[LOW],
             models_incorrect=[MIXED],
             confidence_preferences=[
                 (f"{LOW}_true_confidence", "high"),
                 (f"{MIXED}_true_confidence", "low"),
             ],),
    'both_correct':
        dict(models_correct=[LOW, MIXED],
             confidence_preferences=[
                 (f"{LOW}_true_confidence", "high"),
                 (f"{MIXED}_true_confidence", "high"),
             ],),
    'both_wrong':
        dict(models_incorrect=[LOW, MIXED],
             confidence_preferences=[
                 (f"{LOW}_true_confidence", "low"),
                 (f"{MIXED}_true_confidence", "low"),
             ],),
}


def select_images(
    df,
    n=5,
    models_correct=None,
    models_incorrect=None,
    confidence_preferences=None,
    exclude_classes=None,
    include_classes=None,
    n_per_class=None,
    ranking="rank_sum",
    return_rows=False,
):
    """
    Select images according to model accuracy, class, and confidence criteria.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
            image_id
            true_label
            <model>_pred_label
            <model>_pred_confidence
            <model>_true_confidence

    n : int or None
        Total number of images to return.
        Used only when n_per_class is None.

    models_correct : list[str], optional
        Models that MUST classify the image correctly.

    models_incorrect : list[str], optional
        Models that MUST classify the image incorrectly.

    confidence_preferences : list[tuple], optional
        Each tuple:
            (column_name, "high" or "low")

        Example:
            [
                ("model_A_pred_confidence", "high"),
                ("model_B_true_confidence", "low"),
            ]

    exclude_classes : list, optional
        true_label values to exclude.

    include_classes : list, optional
        If provided, ONLY these true_label values are considered.

    n_per_class : int or dict, optional
        Controls how many images to select from each true class.

        Examples
        --------
        Same number from every included/available class:
            n_per_class=3

        Different number for different classes:
            n_per_class={
                "Angry": 3,
                "Sad": 2,
                "Fear": 4,
            }

        If n_per_class is provided, selection is performed independently
        within each class and `n` is ignored.

    ranking : {"rank_sum", "lexicographic"}
        "rank_sum":
            Equal-weight rank combination across confidence criteria.

        "lexicographic":
            Sort first by criterion 1, then criterion 2, etc.

    return_rows : bool
        False -> return list of image_ids
        True  -> return selected DataFrame rows

    Returns
    -------
    list or pd.DataFrame
    """

    models_correct = models_correct or []
    models_incorrect = models_incorrect or []
    confidence_preferences = confidence_preferences or []
    exclude_classes = exclude_classes or []

    selected = df.copy()

    # =========================================================
    # Accuracy constraints
    # =========================================================
    for model in models_correct:
        selected = selected[
            selected[f"{model}_pred_label"] == selected["true_label"]
        ]

    for model in models_incorrect:
        selected = selected[
            selected[f"{model}_pred_label"] != selected["true_label"]
        ]

    # =========================================================
    # Confidence ranking
    # =========================================================
    if confidence_preferences:

        for column, preference in confidence_preferences:
            if column not in selected.columns:
                raise ValueError(f"Column not found: {column}")

            if preference not in {"high", "low"}:
                raise ValueError(
                    f"Preference must be 'high' or 'low', "
                    f"got: {preference}"
                )

        if ranking == "rank_sum":
            rank_cols = []

            for i, (column, preference) in enumerate(
                confidence_preferences
            ):
                rank_col = f"_rank_{i}"

                # rank = 1 always means "best"
                selected[rank_col] = selected[column].rank(
                    ascending=(preference == "low"),
                    method="average"
                )

                rank_cols.append(rank_col)

            selected["_selection_score"] = (
                selected[rank_cols].mean(axis=1)
            )

            selected = selected.sort_values("_selection_score")
            selected = selected.drop(columns=rank_cols)

        elif ranking == "lexicographic":

            sort_columns = [
                column
                for column, _ in confidence_preferences
            ]

            ascending = [
                preference == "low"
                for _, preference in confidence_preferences
            ]

            selected = selected.sort_values(
                sort_columns,
                ascending=ascending
            )

        else:
            raise ValueError(
                "ranking must be 'rank_sum' or 'lexicographic'"
            )

    # =========================================================
    # Class constraints
    # =========================================================
    if include_classes is not None:
        selected = selected[
            selected["true_label"].isin(include_classes)
        ]

    if exclude_classes:
        selected = selected[
            ~selected["true_label"].isin(exclude_classes)
        ]

    # =========================================================
    # Select N overall OR N per class
    # =========================================================
    if n_per_class is None:

        if n is not None:
            selected = selected.head(n)

    else:

        if isinstance(n_per_class, int):
            # Same number from every class that remains
            selected = (
                selected
                .groupby("true_label", sort=False, group_keys=False)
                .head(n_per_class)
            )

        elif isinstance(n_per_class, dict):
            parts = []

            for class_name, class_n in n_per_class.items():
                class_rows = selected[
                    selected["true_label"] == class_name
                ].head(class_n)

                parts.append(class_rows)

            if parts:
                selected = pd.concat(parts, ignore_index=False)
            else:
                selected = selected.iloc[0:0]

        else:
            raise TypeError(
                "n_per_class must be None, int, or dict"
            )

    # =========================================================
    # Return
    # =========================================================
    if return_rows:
        return selected

    return selected["image_id"].tolist()

img_id_dict = {}
for ttl, cfg in configs.items():
    # print(ttl)
    img_id_dict[ttl] = select_images(
        df,
        n=20,
        **cfg,
        # include_classes=["Surprise", "Angry", ],
        # exclude_classes=["Happy", "Neutral"],
        # n_per_class={cls: 3 for cls in ['Surprise', 'Angry', 'Happy', 'Neutral', 'Sad', 'Disgust', 'Fear']},
        return_rows=True,
    )
    # print(image_ids)

print(img_id_dict)
