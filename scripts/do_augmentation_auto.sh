# do augmentation with auto-aug policies
python src/augment.py \
    --source-dirs ./data/rgb ./data/depth ./data/normal \
    --output-dir ./outputs \
    --count 10 \
    --config ./configs/synthetic_3d_config.py \
    --pipeline RL_searched \
    --batch-size 4 --shuffle-load
