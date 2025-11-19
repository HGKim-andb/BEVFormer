# Risk-Guided BEVFormer

BEVFormer extended with risk prediction capability for autonomous driving safety assessment.

## Overview

This project extends BEVFormer to simultaneously predict:
1. **3D Object Detection** - Standard BEVFormer detection output
2. **BEV Risk Maps** - Risk assessment of occluded/uncertain regions

The risk prediction uses pre-computed risk labels based on occlusion analysis (V5 algorithm) and can optionally guide the attention mechanism for improved detection performance.

## Architecture

```
Multi-Camera Images
       ↓
Image Backbone (ResNet-50/101)
       ↓
BEV Transformer Encoder
       ↓
    BEV Features [B, H*W, C]
       ↓
       ├─→ Detection Head → 3D Bboxes
       │
       └─→ Risk Prediction Head → Risk Map [B, 1, 200, 200]
```

### Key Components

1. **RiskPredictionHead** ([risk_head.py](projects/mmdet3d_plugin/bevformer/dense_heads/risk_head.py))
   - Converts BEV features to risk maps
   - Input: `[B, 256, 50, 50]` BEV features
   - Output: `[B, 1, 200, 200]` risk maps (values in [0, 1])
   - Loss: Combined MSE + MAE with focal weighting

2. **RiskGuidedAttentionHead** ([risk_head.py](projects/mmdet3d_plugin/bevformer/dense_heads/risk_head.py))
   - Extends RiskPredictionHead with attention guidance
   - Generates attention weights based on predicted risk
   - Can guide spatial or channel attention

3. **BEVFormerRisk** ([bevformer_risk.py](projects/mmdet3d_plugin/bevformer/detectors/bevformer_risk.py))
   - Main detector class
   - Multi-task training: detection + risk prediction
   - Configurable risk loss weight

4. **NuScenesRiskDataset** ([nuscenes_risk_dataset.py](projects/mmdet3d_plugin/datasets/nuscenes_risk_dataset.py))
   - Extends NuScenes dataset to load risk labels
   - Automatic filtering by risk threshold
   - Built-in risk evaluation metrics

## Installation

### Prerequisites

```bash
# Same as BEVFormer
# - Python 3.8+
# - PyTorch 1.9+
# - CUDA 11.1+
# - mmdet3d 0.17.1
# - mmcv-full 1.4.0
# - mmdet 2.14.0
```

### Install Additional Dependencies

```bash
pip install scikit-learn scipy matplotlib
```

## Data Preparation

### 1. Download nuScenes Dataset

Follow the [official BEVFormer instructions](https://github.com/fundamentalvision/BEVFormer) to download and prepare nuScenes data.

### 2. Generate Risk Labels

```bash
# For v1.0-mini (testing)
python tools/create_risk_labels.py \
    --dataroot data/nuscenes \
    --version v1.0-mini \
    --output_dir data/emergence_risk_v5

# For v1.0-trainval (full dataset)
python tools/create_risk_labels.py \
    --dataroot data/nuscenes \
    --version v1.0-trainval \
    --output_dir data/emergence_risk_v5_full
```

Expected output structure:
```
data/
├── emergence_risk_v5/              # Mini dataset
│   ├── risk_labels_train.pkl       (~100MB)
│   └── risk_config.json
└── emergence_risk_v5_full/         # Full dataset
    ├── risk_labels_train.pkl       (~15GB)
    ├── risk_labels_val.pkl
    └── risk_config.json
```

See [Risk_Label_Specification.md](docs/Risk_Label_Specification.md) for detailed risk label format.

## Validation & Testing

### Run All Tests

```bash
# Model architecture tests
python validation/test_model.py

# Data pipeline tests
python validation/test_data.py

# Integration tests
python validation/integration_test.py
```

### Individual Test Modules

```python
# Test risk head shape and forward pass
python validation/test_model.py

# Test dataset loading
python validation/test_data.py

# Test visualization
python validation/visualize.py

# Test evaluation metrics
python validation/evaluate.py
```

## Training

### Basic Training (Risk Prediction Only)

```bash
./tools/dist_train.sh \
    projects/configs/bevformer/bevformer_risk_tiny.py \
    8 \
    --work-dir work_dirs/bevformer_risk_tiny
```

### Training with Risk-Guided Attention

Modify config:
```python
model = dict(
    type='BEVFormerRiskAttention',  # Change model type
    risk_head=dict(
        type='RiskGuidedAttentionHead',  # Use attention head
        attention_type='spatial',         # or 'channel', 'both'
        ...
    ),
    use_risk_guidance=True,  # Enable guidance
    ...
)
```

Then train:
```bash
./tools/dist_train.sh \
    projects/configs/bevformer/bevformer_risk_attention_tiny.py \
    8 \
    --work-dir work_dirs/bevformer_risk_attention_tiny
```

## Evaluation

### Detection + Risk Evaluation

```bash
./tools/dist_test.sh \
    projects/configs/bevformer/bevformer_risk_tiny.py \
    work_dirs/bevformer_risk_tiny/latest.pth \
    8 \
    --eval bbox risk
```

### Risk Metrics Only

```python
from validation.evaluate import RiskEvaluator

evaluator = RiskEvaluator(thresholds=[0.3, 0.5, 0.7])

# Add predictions
for pred_risk, gt_risk in predictions:
    evaluator.add_batch(pred_risk, gt_risk)

# Compute metrics
metrics = evaluator.compute_metrics()
evaluator.print_metrics(metrics)
```

## Visualization

```python
from validation.visualize import RiskVisualizer

visualizer = RiskVisualizer(output_dir='visualizations/my_experiment')

# Visualize risk comparison
visualizer.visualize_risk_comparison(
    gt_risk=gt_risk_map,
    pred_risk=pred_risk_map,
    sample_token=sample_token
)

# Visualize risk with detections
visualizer.visualize_risk_with_detections(
    risk_map=pred_risk_map,
    bboxes_3d=predicted_boxes,
    scores=confidence_scores,
    labels=class_labels,
    sample_token=sample_token
)

# Visualize attention weights
visualizer.visualize_attention_weights(
    attention_weights=attention_weights,
    risk_map=risk_map,
    sample_token=sample_token
)
```

## Ablation Study

```python
from validation.evaluate import AblationAnalyzer

ablation = AblationAnalyzer()

# Add experiment results
ablation.add_experiment('baseline', {
    'mAP': 0.354,
    'NDS': 0.428,
    'risk_mae': None
})

ablation.add_experiment('with_risk_head', {
    'mAP': 0.361,
    'NDS': 0.435,
    'risk_mae': 0.082
})

ablation.add_experiment('with_risk_attention', {
    'mAP': 0.375,
    'NDS': 0.448,
    'risk_mae': 0.075
})

# Print comparison
ablation.print_comparison(baseline='baseline')
```

## Configuration Options

### Risk Head Configuration

```python
risk_head=dict(
    type='RiskPredictionHead',
    in_channels=256,        # BEV feature channels
    bev_h=50,               # BEV height
    bev_w=50,               # BEV width
    risk_h=200,             # Output risk map height
    risk_w=200,             # Output risk map width
    num_convs=3,            # Number of conv layers
    conv_channels=128,      # Intermediate channels
    norm_cfg=dict(type='BN'),
    act_cfg=dict(type='ReLU'),
    use_sigmoid=True,       # Sigmoid activation for [0, 1] output
)
```

### Dataset Configuration

```python
data=dict(
    train=dict(
        type='NuScenesRiskDataset',
        use_risk=True,
        risk_labels_path='data/emergence_risk_v5_full/risk_labels_train.pkl',
        risk_map_size=(200, 200),
        risk_threshold=0.0,   # Filter samples by max_risk > threshold
        ...
    ),
    ...
)
```

### Loss Weighting

```python
model = dict(
    risk_loss_weight=1.0,  # Weight for risk loss relative to detection loss
    ...
)
```

## Expected Performance

### Risk Prediction Metrics (v1.0-mini)

| Metric | Value |
|--------|-------|
| MSE | ~0.015 |
| MAE | ~0.080 |
| IoU@0.5 | ~0.65 |
| IoU@0.7 | ~0.52 |
| Pearson R | ~0.85 |

### Detection Metrics (Expected Improvements)

| Model | mAP | NDS | Risk MAE |
|-------|-----|-----|----------|
| Baseline BEVFormer | 0.354 | 0.428 | - |
| + Risk Head | 0.361 | 0.435 | 0.082 |
| + Risk Attention | 0.375 | 0.448 | 0.075 |

*Note: Actual performance may vary depending on training configuration and dataset*

## Project Structure

```
BEVFormer/
├── projects/
│   ├── configs/bevformer/
│   │   ├── bevformer_risk_tiny.py           # Risk-enabled config
│   │   └── bevformer_risk_attention_tiny.py # With attention
│   └── mmdet3d_plugin/
│       ├── bevformer/
│       │   ├── dense_heads/
│       │   │   └── risk_head.py             # Risk prediction heads
│       │   └── detectors/
│       │       └── bevformer_risk.py        # Risk-guided detector
│       └── datasets/
│           └── nuscenes_risk_dataset.py     # Dataset with risk labels
├── validation/
│   ├── test_model.py                        # Model tests
│   ├── test_data.py                         # Data pipeline tests
│   ├── integration_test.py                  # End-to-end tests
│   ├── visualize.py                         # Visualization tools
│   └── evaluate.py                          # Evaluation metrics
├── tools/
│   ├── create_risk_labels.py                # Risk label generation
│   └── risk_utils.py                        # Risk calculation (V5)
└── docs/
    ├── Risk_Label_Specification.md          # Risk label format spec
    └── RISK_GUIDED_BEVFORMER_README.md      # This file
```

## Troubleshooting

### Issue: Risk labels file not found

**Solution**: Generate risk labels first:
```bash
python tools/create_risk_labels.py \
    --dataroot data/nuscenes \
    --version v1.0-mini \
    --output_dir data/emergence_risk_v5
```

### Issue: Shape mismatch in risk head

**Solution**: Check that BEV dimensions match:
- Config: `bev_h`, `bev_w` in risk_head should match BEVFormer's BEV grid size
- Typical values: `bev_h=50, bev_w=50` for BEVFormer-tiny/small

### Issue: Memory error during training

**Solution**: Reduce batch size or use gradient checkpointing:
```python
data = dict(
    samples_per_gpu=1,  # Reduce from default
    ...
)
```

### Issue: NaN in risk loss

**Solution**: Check that:
1. Risk labels are in [0, 1] range
2. Learning rate is not too high (try 1e-4)
3. FP16 is stable (try FP32 first)

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{li2022bevformer,
  title={BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
  author={Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}

@misc{risk_guided_bevformer2025,
  title={Risk-Guided BEVFormer: Occlusion-Aware 3D Object Detection},
  author={Your Name},
  year={2025}
}
```

## License

This project is released under the MIT License. See LICENSE for details.

## Acknowledgements

- [BEVFormer](https://github.com/fundamentalvision/BEVFormer) - Original BEVFormer implementation
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) - 3D detection framework
- [nuScenes](https://www.nuscenes.org/) - Dataset

## Contact

For questions and issues, please open an issue on GitHub or contact [your-email@example.com].
