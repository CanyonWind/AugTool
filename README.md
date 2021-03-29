# Augmentation tool
A simple augmentation tool to load, augment and save multi-source input data. This repo contains 17 augmentation 
operations,  and supports [auto-aug](https://arxiv.org/pdf/1805.09501.pdf) searched augmentation policies.
![aug_tool](./resource/augmentation.png)

## Usage
```sh
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

python src/augment.py \
    --source-dirs ./data/depth ./data/rgb ./data/normal \
    --output-dir ./outputs \
    --count 10 \
    --config ./configs/synthetic_3d_config.py \
    --pipeline default \
    --photo-distort-all --shuffle-load
    

# Alternatively, you can just run
sh ./scripts/do_augmentation.sh
```

## Features
-  Modularized structure to facilitate configurable pipeline. 
-  17 augmentation operations implemented

        'Rotate': transform.Rotate,
        'ShearX': transform.ShearX,
        'ShearY': transform.ShearY,
        'TranslateX': transform.TranslateX,
        'TranslateY': transform.TranslateY,
        'AutoContrast': transform.AutoContrast,
        'Invert': transform.Invert,
        'Equalize': transform.Equalize,
        'Mirror': transform.Mirror,
        'Flip': transform.Flip,
        'Solarize': transform.Solarize,
        'Posterize': transform.Posterize,
        'Contrast': transform.Contrast,
        'Color': transform.Color,
        'Brightness': transform.Brightness,
        'Sharpness': transform.Sharpness,
        'RandomResizedCrop': transform.RandomResizedCrop

- Support [Auto Augmentation](https://arxiv.org/pdf/1805.09501.pdf) searched policies when `--pipeline RL_searched` specified. 
- Load and process batch-wise data. Allow data shuffling before loading when `--shuffle-load` specified.
- Photo metric distortions, like `Contrast`, `Color`, `Solarize`,  can be turned on/off for `depth` and `normal` data with `--photo-distort-all` specified or not. Default setting is to only do photo metric distortions on `rgb` data and apply geometry distortions across all sources.
