python src/augment.py \
    --source-dirs ./data/depth ./data/rgb ./data/normal \
    --output-dir ./outputs \
    --count 10 \
    --config ./configs/synthetic_3d_config.py \
    --pipeline default \
    --photo-distort-all --shuffle-load
