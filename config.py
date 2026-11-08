"""
configs for viz_attn_apvit.py
"""

viz_attn_config = {
    1: {
        'choose_model': 'apvit',  # 'deit' / 'apvit'
        'model_names': ['RAF_blur8_pretrained0-8_freeze_fix_projs', 'RAF_blur8_freeze'],
        'test_blurs': [8],
        'save_figs': True,
        'split_by_emo': True,
        'emos2plot': ['Angry'],  # Choose from keys in 'ims': ['Happy', 'Surprise', 'Angry']
        'im_dir': 'samples/test_images_apvit',
        'ims_type': 'mixed_wins',  # choose from 'mixed_wins' / 'uniform_wins'
        'cols2show': {'img': ['original', 'trans'],
                      'RAF_blur8_freeze': ['pred_map', 'true_map'],
                      'RAF_blur8_pretrained0-8_freeze_fix_projs': ['pred_map']
                      },
    },
    2: {
        # Show attn-map on high-res image of mixed model (winner)
        'choose_model': 'apvit',  # 'deit' / 'apvit'
        'model_names': ['RAF_blur8_pretrained0-8_freeze_fix_projs', 'RAF_blur8_freeze'],
        'test_blurs': [0, 8],
        'save_figs': True,
        'split_by_emo': True,
        'emos2plot': ['Angry', 'Happy', 'Surprise', 'Sad'],  # Choose from keys in 'ims': ['Happy', 'Surprise', 'Angry']
        'im_dir': 'samples/test_images_apvit',
        'ims_type': 'mixed_wins',  # choose from 'mixed_wins' / 'uniform_wins'
        'cols2show': {
            'img': ['original', 'trans'],
            'RAF_blur8_freeze': ['pred_map', 'true_map'],
            'RAF_blur8_pretrained0-8_freeze_fix_projs': ['pred_map', 'pred_map_HR']
                      },
    },
    3: {
        # Show attn-map on high-res image of mixed model (winner), BUT use model trained on high-res for hr viz (pretrinaed on mixed).
        'choose_model': 'apvit',  # 'deit' / 'apvit'
        'model_names': ['RAF_blur8_pretrained0-8_freeze_fix_projs', 'RAF_blur8_freeze'],
        'test_blurs': [0, 8],
        'save_figs': True,
        'split_by_emo': True,
        'emos2plot': ['Angry', 'Happy', 'Surprise', 'Sad'],  # Choose from keys in 'ims': ['Happy', 'Surprise', 'Angry']
        'im_dir': 'samples/test_images_apvit',
        'ims_type': 'mixed_wins',  # choose from 'mixed_wins' / 'uniform_wins'
        'cols2show': {
            'img': ['original', 'trans'],
            'RAF_blur8_freeze': ['pred_map', 'true_map'],
            'RAF_blur8_pretrained0-8_freeze_fix_projs': ['pred_map', 'pred_map_HR']
                      },
        'models_for_hr_viz': {'RAF_blur8_pretrained0-8_freeze_fix_projs': 'RAF_blur0_pretrained0-8_freeze'},
    },
    4: {
        # Show attn-maps of mixed on low-res; mixed on high-res & low on low-res, for images correctly classified by all models.
        'choose_model': 'apvit',  # 'deit' / 'apvit'
        'model_names': ['RAF_blur8_pretrained0-8_freeze_fix_projs', 'RAF_blur8_freeze'],
        'test_blurs': [0, 8],
        'save_figs': True,
        'split_by_emo': True,
        'emos2plot': ['Angry', 'Happy'],  # Choose from keys in 'ims': ['Happy', 'Surprise', 'Angry', 'Sad']
        'im_dir': 'samples/test_images_apvit',
        'ims_type': 'all3_win',  # choose from 'mixed_wins' / 'uniform_wins' / 'all3_win'
        'cols2show': {
            'img': ['original', 'trans'],
            'RAF_blur8_freeze': ['pred_map', 'true_map'],
            'RAF_blur8_pretrained0-8_freeze_fix_projs': ['pred_map', 'pred_map_HR']
                      },
        'models_for_hr_viz': {'RAF_blur8_pretrained0-8_freeze_fix_projs': 'RAF_blur0_pretrained0-8_freeze'},
    },
    5: {
        # nimstim image
        'choose_model': 'apvit',  # 'deit' / 'apvit'
        'model_names': ['RAF_blur8_pretrained0-8_freeze_fix_projs', 'RAF_blur8_freeze'],
        'test_blurs': [8],
        'save_figs': True,
        'split_by_emo': True,
        'emos2plot': ['Female', 'Male'],
        'im_dir': 'samples/test_images_nimstim',
        'use_img_lbl_dict': True,
        'ims_type': 'nimstim',  # choose from 'mixed_wins' / 'uniform_wins' / 'all3_win'
        'cols2show': {
            'img': ['original', 'trans'],
            'RAF_blur8_freeze': ['pred_map', 'true_map'],
            'RAF_blur8_pretrained0-8_freeze_fix_projs': ['pred_map', 'true_map']
                      },
    },
    6: {
        # Examples from RAF-DB that are similar to nimstim (open mouths)
        'choose_model': 'apvit',  # 'deit' / 'apvit'
        'model_names': ['RAF_blur8_pretrained0-8_freeze_fix_projs', 'RAF_blur8_freeze'],
        'test_blurs': [8],
        'save_figs': True,
        'split_by_emo': True,
        'emos2plot': ['Surprise', 'Sad', 'Disgust', 'Fear'],
        'im_dir': '/home/projects/bagon/ilanaveh/data/RAF-DB/basic/Image/aligned_224',
        'ims_type': 'examples_apvit',  # choose from 'mixed_wins' / 'uniform_wins' / 'all3_win'
        'cols2show': {
            'img': ['original', 'trans'],
            'RAF_blur8_freeze': ['pred_map', 'true_map'],
            'RAF_blur8_pretrained0-8_freeze_fix_projs': ['pred_map', 'true_map']
                      },
    }

}
