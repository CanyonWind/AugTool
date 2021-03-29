# A simple augmentation tool
A simple augmentation tool to load, augment and save multi-source input data. This repo contains 17 augmentation 
operations,  and supports [auto-aug](https://arxiv.org/pdf/1805.09501.pdf) searched augmentation policies.
![aug_tool](./resource/augmentation.png)

# Usage
```sh
python3 -m venv env
source env/bin/activate
pip install -r requirements

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

# Features
A few features of this augmentation tool.

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

- 