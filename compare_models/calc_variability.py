"""
03/08/26
Similar to comp_image_separability, but calculate variability (mean pairwise wasserstein distances):
    - between all image-pairs for each model (distances to images between & within the category)
        - average per model
        - average per category
    - between image-pairs within each category

Assumes cache files saved from 'calc_image_separability.py':
    attention_maps/
        - image_attention_separability_sigma8_ALL_PRED/pairwise_model_cache/pairwise_distances__{model}.pkl
        - image_attention_separability_sigma8_ALL_TRUE/pairwise_model_cache/pairwise_distances__{model}.pkl
        - image_attention_separability_sigma8_CORRECT_PRED/pairwise_model_cache/pairwise_distances__{model}.pkl
"""
import pandas as pd

models = ['RAF_blur8_freeze', 'RAF_blur8_pretrained0-8_freeze_fix_projs']
mean_dists = {mdl: None for mdl in models}
cache_pths = {mdl:
                  f'attention_maps/image_attention_separability_sigma8_CORRECT_PRED/pairwise_model_cache'
                  f'/pairwise_distances__{mdl}.pkl'
              for mdl in models}

for mdl in models:
    cached_df = pd.read_pickle(cache_pths[mdl])
    mean_dists[mdl] = cached_df.loc[cached_df['same_emotion']].groupby('label_a')['wasserstein'].mean()

print(mean_dists)
