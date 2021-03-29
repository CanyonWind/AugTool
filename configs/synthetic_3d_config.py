# Dataset settings
dataset_type = 'Synthetic3D'
aug_times = 10
data = dict(
    root='./data',
    output_dir='./outputs',
    shuffle_load=True,
    batch_size=4,
)

# Augmentation settings
photo_metric_distortion_for_all = [False]
pipeline = 'default'  # choice of ('default', 'RL_searched')
default_pipeline = [
    # sequence & magnitudes of the applied augmentations
    dict(type='Brightness', apply_prob=0.2, value_range=(0.1, 1.9), apply_all=photo_metric_distortion_for_all),
    dict(type='Color', apply_prob=0.2, value_range=(0.1, 1.0), apply_all=photo_metric_distortion_for_all),
    dict(type='Contrast', apply_prob=0.2, value_range=(0.1, 1.9), apply_all=photo_metric_distortion_for_all),
    dict(type='AutoContrast', apply_prob=0.2, apply_all=photo_metric_distortion_for_all),
    dict(type='Invert', apply_prob=0.05, apply_all=photo_metric_distortion_for_all),
    dict(type='Equalize', apply_prob=0.05, apply_all=photo_metric_distortion_for_all),
    dict(type='Solarize', apply_prob=0.2, value_range=(0, 256), apply_all=photo_metric_distortion_for_all),
    dict(type='Posterize', apply_prob=0.2, value_range=(4, 8), apply_all=photo_metric_distortion_for_all),
    dict(type='Sharpness', apply_prob=1.0, value_range=(0.1, 1.9), apply_all=photo_metric_distortion_for_all),
    dict(type='Mirror', apply_prob=0.5),
    dict(type='Flip', apply_prob=0.2),
    dict(type='Rotate', apply_prob=1.0, value_range=(-30, 30)),
    dict(type='ShearX', apply_prob=1.0, value_range=(-0.25, 0.25)),
    dict(type='ShearY', apply_prob=1.0, value_range=(-0.25, 0.25)),
    dict(type='TranslateX', apply_prob=1.0, value_range=(-0.30, 0.30)),
    dict(type='TranslateY', apply_prob=1.0, value_range=(-0.30, 0.30)),
    dict(type='RandomResizedCrop', scale=(0.5, 2.0), padding_mode='constant'),
]

# Auto-augmentation trained RL model to explore the optimal augmentation policy combinations.
# Check https://arxiv.org/pdf/1805.09501.pdf for more details.
RL_searched_pipeline = [
    [dict(type='Posterize', apply_prob=0.4, value_range=(8, 8), apply_all=photo_metric_distortion_for_all),
     dict(type='Rotate', apply_prob=0.6, value_range=(9, 9))],
    [dict(type='Solarize', apply_prob=0.6, value_range=(5, 5), apply_all=photo_metric_distortion_for_all),
     dict(type='AutoContrast', apply_prob=0.6, apply_all=photo_metric_distortion_for_all)],
    [dict(type='Equalize', apply_prob=0.8, apply_all=photo_metric_distortion_for_all),
     dict(type='Equalize', apply_prob=0.6, apply_all=photo_metric_distortion_for_all)],
    [dict(type='Posterize', apply_prob=0.6, value_range=(7, 7), apply_all=photo_metric_distortion_for_all),
     dict(type='Posterize', apply_prob=0.6, value_range=(6, 6), apply_all=photo_metric_distortion_for_all)],
    [dict(type='Equalize', apply_prob=0.4, apply_all=photo_metric_distortion_for_all),
     dict(type='Solarize', apply_prob=0.2, value_range=(4, 4), apply_all=photo_metric_distortion_for_all)],
    # 19 sub-policies more. Not listing all here for better clarity.
]
