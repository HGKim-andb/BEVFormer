#!/bin/bash
# Subsample full dataset risk labels to 20%

echo "Subsampling risk labels to 20%..."
echo ""

# Train set
python tools/subsample_risk_labels.py \
    --input data/emergence_risk_v5_full/risk_labels_train.pkl \
    --output data/emergence_risk_v5_full/risk_labels_train_20pct.pkl \
    --ratio 0.2 \
    --seed 42

echo ""
echo "=========================================="
echo ""

# Val set
python tools/subsample_risk_labels.py \
    --input data/emergence_risk_v5_full/risk_labels_val.pkl \
    --output data/emergence_risk_v5_full/risk_labels_val_20pct.pkl \
    --ratio 0.2 \
    --seed 42

echo ""
echo "=========================================="
echo "All subsampling complete!"
echo "=========================================="
echo ""
echo "New files created:"
echo "  - data/emergence_risk_v5_full/risk_labels_train_20pct.pkl"
echo "  - data/emergence_risk_v5_full/risk_labels_val_20pct.pkl"
echo ""
echo "To use these in training, update your config:"
echo "  risk_labels_path='data/emergence_risk_v5_full/risk_labels_train_20pct.pkl'"
