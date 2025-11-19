#!/bin/bash
# Generate risk labels for train/val split

# Train: scenes 0-7 (8 scenes)
echo "Generating TRAIN risk labels..."
python tools/create_risk_labels.py \
    --dataroot data/nuscenes \
    --version v1.0-mini \
    --output_dir data/emergence_risk_v5_temp_train \
    --scenes scene-0061 scene-0103 scene-0553 scene-0655 scene-0757 scene-0796 scene-0916 scene-1077

# Val: scenes 8-9 (2 scenes)
echo ""
echo "Generating VAL risk labels..."
python tools/create_risk_labels.py \
    --dataroot data/nuscenes \
    --version v1.0-mini \
    --output_dir data/emergence_risk_v5_temp_val \
    --scenes scene-1094 scene-1100

# Move files to final location
echo ""
echo "Moving files to data/emergence_risk_v5/..."
cp data/emergence_risk_v5_temp_train/risk_labels_train.pkl data/emergence_risk_v5/risk_labels_train_real.pkl
cp data/emergence_risk_v5_temp_val/risk_labels_train.pkl data/emergence_risk_v5/risk_labels_val_real.pkl
cp data/emergence_risk_v5_temp_train/risk_config.json data/emergence_risk_v5/

echo ""
echo "✅ Done! Files created:"
echo "  - data/emergence_risk_v5/risk_labels_train_real.pkl (8 scenes)"
echo "  - data/emergence_risk_v5/risk_labels_val_real.pkl (2 scenes)"
echo ""
echo "To use these files, update your config:"
echo "  risk_labels_path='data/emergence_risk_v5/risk_labels_train_real.pkl'"
echo "  risk_labels_path='data/emergence_risk_v5/risk_labels_val_real.pkl'"
