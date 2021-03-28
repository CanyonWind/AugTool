# dataset settings
dataset_type = 'Synthetic3D'
data = dict(
    root='../data',
    output_dir='../outputs',
    shuffle_load=True,
    img_norm_cfg=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375])
)

aug_times = 10
apply_all = True

aug_pipeline = [
    dict(type='Rotate', fill_color=128, value_range=(-45, 45)),
]
