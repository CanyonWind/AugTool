# do augmentation with photo metric distortions applied on all sources.
python src/augment.py \
    --source-dirs ./data/rgb ./data/depth ./data/normal \
    --output-dir ./outputs \
    --count 10 \
    --config ./configs/synthetic_3d_config.py \
    --pipeline default \
    --batch-size 4 --shuffle-load
