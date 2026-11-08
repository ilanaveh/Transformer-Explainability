"""
2/9/26
Plot the difference map of a single face-image (similar to plot_diff_maps.py, but single image, not mean over subset).
+ show overlayed attention maps of each of the models separately - loaded from
    attention_maps/NATIVE_RESOLUTION{pool_dir}/overlayed_figures, maps were originally created & saved by
    attention_maps/create_overlayed_figures.py.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from plot_prototypes_confusion import make_alpha_map


POOLING = True
pool_dir = '/WITH_POOLING' if POOLING else ''
add_diff = False
target_label = 'pred'  # 'pred' / 'true'
n_imgs_per_class = 3  # according to faces_ids lists given by get_examples_from_wide_table.py.
n_classes = 7

# face_id = 'test_0970'  # mixed wins, high diff in confidence, true: surprised, pred: happy
# face_id = 'test_1281'  # mixed wins, high diff in confidence, true: Angry, pred:
# face_id = 'test_1807'  # uniform wins, high diff in confidence, true: surprised, pred: disgust
# face_id = 'test_2234'  # Both correct with high confidence, true: surprised
# face_id = 'test_1643'  # Both correct with high confidence, true: Angry
# face_id = 'test_0517'  # one of the paper examples

# faces_ids = ['test_0970', 'test_0517']
# faces_ids = {
#     'mixed_wins':
#         ['test_0411', 'test_1128', 'test_1697', 'test_0435', 'test_1155', 'test_0970', 'test_0765', 'test_0189', 'test_1809', 'test_0187', 'test_0004', 'test_1345', 'test_2550', 'test_1233', 'test_2803', 'test_0045', 'test_1234', 'test_0470', 'test_1281', 'test_2396', 'test_2746', 'test_1939', 'test_2585', 'test_1132', 'test_0244', 'test_2303', 'test_2381', 'test_2214', 'test_2373', 'test_2284', 'test_2378', 'test_2324', 'test_2293', 'test_2273'],
#     'low_wins':
#         ['test_1841', 'test_0279', 'test_1807', 'test_0524', 'test_2152', 'test_2444', 'test_2135', 'test_1413', 'test_2678', 'test_2958', 'test_0293', 'test_2517', 'test_2476', 'test_0644', 'test_1174', 'test_2272', 'test_1756', 'test_1250', 'test_0853', 'test_0998', 'test_2358', 'test_1997', 'test_0960', 'test_0095', 'test_0347', 'test_2227', 'test_2246', 'test_1891', 'test_0774', 'test_2034', 'test_2249', 'test_0217', 'test_0290', 'test_2238', 'test_2301'],
#     'both_correct':
#         ['test_0236', 'test_1497', 'test_0886', 'test_1438', 'test_0568', 'test_2234', 'test_1503', 'test_1146', 'test_0839', 'test_0994', 'test_1643', 'test_0551', 'test_2788', 'test_2424', 'test_2791', 'test_2285', 'test_1059', 'test_2899', 'test_2407', 'test_2278', 'test_2289', 'test_1838', 'test_1782', 'test_1365', 'test_2294', 'test_2282', 'test_0276', 'test_0476', 'test_0005', 'test_1599', 'test_1094', 'test_2319', 'test_2371', 'test_1935', 'test_2320'],
#     'both_wrong':
#         ['test_2243', 'test_2296', 'test_2252', 'test_2254', 'test_2548', 'test_2021', 'test_2300', 'test_1487', 'test_1077', 'test_1162', 'test_1914', 'test_1988', 'test_0679', 'test_1035', 'test_2370', 'test_0466', 'test_1026', 'test_0091', 'test_0147', 'test_0465', 'test_1293', 'test_1737', 'test_2985', 'test_2056', 'test_1973', 'test_2154', 'test_1625', 'test_0582', 'test_1844', 'test_1041', 'test_2656', 'test_2826', 'test_1278', 'test_1904', 'test_2980'],
# }
faces_ids = {
    'mixed_wins':
        ['test_0189', 'test_0316', 'test_1145', 'test_0244', 'test_1281', 'test_0793', 'test_0411', 'test_1128', 'test_0435', 'test_2550', 'test_2814', 'test_2746', 'test_1809', 'test_1233', 'test_0765', 'test_2324', 'test_2381', 'test_2373', 'test_2303', 'test_2259', 'test_2256'],
    'low_wins':
        ['test_1807', 'test_0441', 'test_0293', 'test_1756', 'test_1413', 'test_0347', 'test_0279', 'test_1841', 'test_2053', 'test_2476', 'test_2517', 'test_3039', 'test_1174', 'test_1450', 'test_1607', 'test_0960', 'test_2034', 'test_2358', 'test_2272', 'test_2268', 'test_2238'],
    'both_correct':
        ['test_1503', 'test_2234', 'test_2146', 'test_1643', 'test_0551', 'test_1059', 'test_0886', 'test_1497', 'test_0236', 'test_2791', 'test_2788', 'test_2569', 'test_1365', 'test_1086', 'test_0276', 'test_2371', 'test_1599', 'test_1935', 'test_2289', 'test_2278', 'test_2285'],
    'both_wrong':
        ['test_1077', 'test_0679', 'test_0147', 'test_1487', 'test_1914', 'test_0091', 'test_1930', 'test_1594', 'test_0649', 'test_2548', 'test_2985', 'test_2826', 'test_2021', 'test_1162', 'test_1737', 'test_1988', 'test_1035', 'test_2370', 'test_2243', 'test_2252', 'test_2300']
}

assert all([len(faces_ids[x])/n_classes == n_imgs_per_class for x in faces_ids.keys()])
# face_id = 'test_2050'
# face_id = 'test_0136'

out_h, out_w = (112, 112)
CMAP = "jet"
INTERPOLATION = 'bilinear'  # 'bilinear' for smoothing, 'nearest' for keeping patch-resolution (relevant for native prototypes)
max_alpha = 0.65
models = ['RAF_blur8_pretrained0-8_freeze_fix_projs', 'RAF_blur8_freeze']
# format_suf = '__gray__thresh0.2'  # according to suffix given by create_overlayed_figures.py.
# format_suf = '__thresh0.2__faceBlurred8'  # according to suffix given by create_overlayed_figures.py.
format_suf = '__gray__thresh0.2__faceBlurred8'  # according to suffix given by create_overlayed_figures.py.
# models = ['RAF_blur0_freeze', 'RAF_blur8_freeze']

home_pth = Path('/home/projects/bagon/ilanaveh')
metadata_pth = f'attention_maps/NATIVE_RESOLUTION{pool_dir}/metadata_all_models_native.csv'

metadata = pd.read_csv(metadata_pth)

for ttl, id_list in faces_ids.items():
    save_dir = f'attention_maps/NATIVE_RESOLUTION{pool_dir}/overlayed_figures/compare/{ttl}/show_attn_{target_label}_class'
    os.makedirs(save_dir, exist_ok=True)
    for idx, face_id in enumerate(id_list):
        img_save_dir = f'{save_dir}/no{np.mod(idx, n_imgs_per_class) + 1}'  # subdir according to image ranking
        os.makedirs(img_save_dir, exist_ok=True)
        fig_save_pth = f'{img_save_dir}/{face_id}_compare{format_suf.replace("__", "_")}.png'
        # if os.path.isfile(fig_save_pth):
        #     continue

        face_pth = home_pth / f'data/RAF-DB/basic/Image/aligned_224/{face_id}.jpg'
        diff_map_pth = f'difference_maps/RAF_blur8_pretrained0-8_freeze_fix_projs_minus_RAF_blur8_freeze_pred_label/maps/{face_id}.npy'
        img_row = metadata[metadata['image_id'] == face_id]
        true_label = img_row['true_label'].iloc[0]
        pred_labels = [img_row.loc[img_row['model'] == mdl, 'pred_label'].iloc[0] for mdl in models]
        sigmas = [img_row.loc[img_row['model'] == mdl, 'sigma'].iloc[0] for mdl in models]

        face_img = Image.open(face_pth).convert("RGB")
        face_img = face_img.resize((out_w, out_h), Image.BILINEAR)
        face_arr = np.asarray(face_img).astype(np.float32) / 255.0

        diff_attn_map = np.load(diff_map_pth)
        diff_attn_map = np.asarray(diff_attn_map, dtype=float)
        diff_attn_map = np.squeeze(diff_attn_map)
        if diff_attn_map.shape != face_arr.shape[:2]:
            display_map = torch.nn.functional.interpolate(
                torch.from_numpy(diff_attn_map).reshape(1, 1, 14, 14),
                scale_factor=face_arr.shape[0] / diff_attn_map.shape[0],
                mode=INTERPOLATION).reshape(face_arr.shape[:2]).numpy()

        ncols = (len(models) + 1) if add_diff else len(models)
        fig, axes = plt.subplots(1, ncols)

        for i, ax in enumerate(axes[:len(models)]):
            overlayed = plt.imread(f'attention_maps/NATIVE_RESOLUTION{pool_dir}/overlayed_figures/{models[i]}/{face_id}__true-{true_label}__pred-{pred_labels[i]}__target-{target_label if true_label!=pred_labels[i] else "pred"}__sigma-{sigmas[i]}{format_suf}.png')
            # overlayed = plt.imread(f'attention_maps/NATIVE_RESOLUTION{pool_dir}/overlayed_figures/{models[i]}/{face_id}__true-{true_label}__pred-{pred_labels[i]}__target-{"true" if i==1 else "pred"}__sigma-8.png')
            ax.imshow(overlayed)
            ax.axis('off')
            ax.set_title(f"{models[i].replace('RAF_', '').split('_freeze')[0]}\nPred: {pred_labels[i]}")

        if add_diff:
            ax = axes[len(models)]
            ax.imshow(np.clip(face_arr, 0, 1), cmap='gray')
            alpha_map = max_alpha * (np.abs(display_map) - np.min(np.abs(display_map))) / (np.max(np.abs(display_map)) - np.min(np.abs(display_map)))
            overlayed = ax.imshow(
                display_map,
                cmap=CMAP,
                alpha=alpha_map
            )
            ax.axis('off')

        fig.suptitle(f'True: {true_label}')

        plt.savefig(fig_save_pth)
        plt.close()
        # plt.show(block=True)
