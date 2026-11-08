
img_lbl_dict = {
    'test_0038_112.jpg': 4,
    'test_1261_112.jpg': 5,
    'test_1697_112.jpg': 4,
    'test_1261.jpg': 5,
    'test_0411.jpg': 4,
    'test_0189.jpg': 5,
    'test_0130.jpg': 4,
    'test_1155.jpg': 4,
    'test_0793.jpg': 0,
    'test_2119.jpg': 3,
    'test_1128.jpg': 4,
    'test_2481.jpg': 6,
    'f_ang_comp_224.png': 0,
    'f_dis_comp_224.png': 1,
    'f_fea_comp_224.png': 2,
    'f_hap_comp_224.png': 4,
    'f_sad_comp_224.png': 3,
    'f_sup_comp_224.png': 5,
    'ChatGPT Image May 28, 2026, 12_04_53 PM.png': 1,
    'ChatGPT Image May 28, 2026, 11_37_42 AM.png': 0,
    'm_ang_comp_224.png': 0,
    'm_dis_comp_224.png': 1,
    'm_fea_comp_224.png': 2,
    'm_hap_comp_224.png': 4,
    'm_sad_comp_224.png': 3,
    'm_sup_comp_224.png': 5,
    'f_ang_comp_crop_224.png': 0,
    'f_dis_comp_crop_224.png': 1,
    'f_fea_comp_crop_224.png': 2,
    'f_hap_comp_crop_224.png': 4,
    'f_sad_comp_crop_224.png': 3,
    'f_sup_comp_crop_224.png': 5,
    'm_ang_comp_crop_224.png': 0,
    'm_dis_comp_crop_224.png': 1,
    'm_fea_comp_crop_224.png': 2,
    'm_hap_comp_crop_224.png': 4,
    'm_sad_comp_crop_224.png': 3,
    'm_sup_comp_crop_224.png': 5,
    '01F_AN_O_square_224.png': 0,
    '01F_DI_O_square_224.png': 1,
    '01F_FE_O_square_224.png': 2,
}

ims_fail = [  # images that uniform model failed and mixed model succeeded
        '1128',  # Happy
        '1155',  # Happy
        '0130',  # Happy
        '0189',  # Surprise
        '0793',  # Angry
        '2119',  # Sad
        '2481',  # Neutral
    ]

ims = {'mixed_wins': {
    'Happy': [
        '1128',
        '1155',
        '0130',
        '2004',
        '1596',
        '0815',
    ],
    'Surprise': [
        '0189',
        '0903',
        '2207',
        '2014',
        '1822',
        '1400',
    ],
    'Angry': [
            '1939',
            '1263',
            '1294',
            '0793',
            '0190',
            '1281',
    ],
    'Sad': [
            '1156',
            '0765',
            '2050',
            '2119',
            '1961',
            '2138',
    ]
        },
    'uniform_wins': {
        'Happy': [
            '0364',
            '0769',
            '0345',
            '0811',
            '1635',
            '2027',
        ],
        'Surprise': [
            '0441',
            '1807',
            '1250',
            '0293',
            '1744',
            '2149',
        ],
        'Angry': [
            '1756',
            '0953',
            '1061',
            '1217',
            '1309',
            '0347'
        ],
        'Sad': [
            '1450',
            '0738',
            '1052',
            '1607',
            '0290',
            '1241',
        ]
    },
    'all3_win': {
        'Angry': [
            '1976',  # high delta (between models > within mixed)
            '1007',  # high delta
            '1946',  # high delta
            '0136',  # low delta (within mixed > between models)
            '1760',  # low delta
            '0228'  # low delta
        ],
        'Happy': [
            '0494',  # high delta (between models > within mixed)
            '2090',  # high delta
            '1483',  # high delta
            '0421',  # low delta (within mixed > between models)
            '1401',  # low delta
            '0239',  # low delta
        ],
        'Sad': [],
        'Surprise': [],
    },
    'nimstim': {
        'Female': [
            # 'f_ang_comp_crop_224.png',
            # 'f_dis_comp_crop_224.png',
            # 'f_fea_comp_crop_224.png',
            # 'f_hap_comp_crop_224.png',
            # 'f_sad_comp_crop_224.png',
            # 'f_sup_comp_crop_224.png',
            # 'ChatGPT Image May 28, 2026, 12_04_53 PM.png',
            # 'ChatGPT Image May 28, 2026, 11_37_42 AM.png',
            '01F_AN_O_square_224.png',
            '01F_FE_O_square_224.png',
            '01F_DI_O_square_224.png',
        ],
        'Male': [
            # 'm_ang_comp_crop_224.png',
            # 'm_dis_comp_crop_224.png',
            # 'm_fea_comp_crop_224.png',
            # 'm_hap_comp_crop_224.png',
            # 'm_sad_comp_crop_224.png',
            # 'm_sup_comp_crop_224.png',
        ]
    },
    'examples_apvit': {  # try to find examples similar to nimstim - open mouths.
        'Surprise': ['0008', '0094', '0997'],
        'Sad': ['0134', '0315', '1221', '1645'],
        'Disgust': ['0011', '2366', '0280'],
        'Fear': ['2267', '2275', '1477', '2422'],
    }
}
