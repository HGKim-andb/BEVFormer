# Risk-Guided BEVFormer Quick Start Guide

Complete guide to get Risk-Guided BEVFormer running in < 30 minutes.

## Prerequisites Checklist

- [ ] Python 3.8+
- [ ] PyTorch 1.9+ with CUDA 11.1+
- [ ] BEVFormer environment set up
- [ ] nuScenes dataset downloaded (at least v1.0-mini)

## Step 1: Generate Risk Labels (5-10 min)

```bash
# For quick testing with mini dataset
python tools/create_risk_labels.py \
    --dataroot data/nuscenes \
    --version v1.0-mini \
    --output_dir data/emergence_risk_v5

# Expected output:
# ✅ Created: data/emergence_risk_v5_full/risk_labels_train.pkl (~100MB)
# ✅ Processing time: ~5-10 minutes on CPU
```

## Step 2: Run Validation Tests (2-3 min)

```bash
# Test model architecture
python validation/test_model.py

# Expected output:
# ✅ Risk Head Shape Test PASSED!
# ✅ Risk Head Loss Test PASSED!
# ✅ Gradient Flow Test PASSED!
# ...

# Test data pipeline
python validation/test_data.py

# Expected output:
# ✅ Risk Labels File Existence PASSED!
# ✅ Risk Label Format PASSED!
# ...
```

## Step 3: Quick Training Test (5 min)

Test that everything works with a short training run:

```bash
# Create a quick test config
cat > test_config.py << 'EOF'
_base_ = ['./projects/configs/bevformer/bevformer_risk_tiny.py']

# Override for quick testing
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
)

total_epochs = 1
evaluation = dict(interval=1)

log_config = dict(interval=10)
EOF

# Run 1 epoch to verify everything works
python tools/train.py test_config.py --work-dir work_dirs/quick_test
```

Expected output:
```
Epoch [1][10/404]  lr: 6.667e-05, loss_cls: 2.5xxx, loss_bbox: 0.8xxx, loss_risk: 0.15xxx
Epoch [1][20/404]  lr: 1.333e-04, loss_cls: 2.3xxx, loss_bbox: 0.7xxx, loss_risk: 0.12xxx
...
```

## Step 4: Visualize Results (2 min)

```python
# test_visualization.py
from validation.visualize import RiskVisualizer
import pickle
import numpy as np

# Load a sample from risk labels
with open('data/emergence_risk_v5_full/risk_labels_train.pkl', 'rb') as f:
    risk_labels = pickle.load(f)

# Get first sample
scene_token = list(risk_labels.keys())[0]
sample = risk_labels[scene_token][0]

# Create visualizer
visualizer = RiskVisualizer(output_dir='visualizations/quickstart')

# Visualize GT risk
gt_risk = sample['risk_map']
pred_risk = gt_risk + np.random.randn(200, 200) * 0.05  # Fake prediction
pred_risk = np.clip(pred_risk, 0, 1)

fig = visualizer.visualize_risk_comparison(
    gt_risk=gt_risk,
    pred_risk=pred_risk,
    sample_token=sample['sample_token']
)

print(f"✅ Visualization saved to visualizations/quickstart/")
```

Run it:
```bash
python test_visualization.py
```

Check output: `visualizations/quickstart/risk_comparison_*.png`

## Step 5: Run Full Training (Optional, 8-12 hours)

```bash
# Single GPU
python tools/train.py \
    projects/configs/bevformer/bevformer_risk_tiny.py \
    --work-dir work_dirs/bevformer_risk_tiny

# Multi-GPU (8 GPUs)
./tools/dist_train.sh \
    projects/configs/bevformer/bevformer_risk_tiny.py \
    8 \
    --work-dir work_dirs/bevformer_risk_tiny
```

## Step 6: Evaluate Model

```bash
# Evaluate trained model
./tools/dist_test.sh \
    projects/configs/bevformer/bevformer_risk_tiny.py \
    work_dirs/bevformer_risk_tiny/latest.pth \
    8 \
    --eval bbox risk
```

## Common Commands

### Monitor Training

```bash
# Watch log
tail -f work_dirs/bevformer_risk_tiny/20250118_*.log

# TensorBoard
tensorboard --logdir work_dirs/bevformer_risk_tiny
```

### Inference on Single Sample

```python
from mmdet3d.apis import init_model, inference_detector

# Load model
config_file = 'projects/configs/bevformer/bevformer_risk_tiny.py'
checkpoint_file = 'work_dirs/bevformer_risk_tiny/latest.pth'

model = init_model(config_file, checkpoint_file, device='cuda:0')

# Prepare data (from dataset)
# ... load sample data ...

# Inference
result = inference_detector(model, data)

# result contains:
# - 'pts_bbox': detection results
# - 'risk_map': predicted risk map [1, 200, 200]

print(f"Detected {len(result['pts_bbox']['boxes_3d'])} objects")
print(f"Max risk: {result['risk_map'].max():.4f}")
```

### Evaluate Metrics Programmatically

```python
from validation.evaluate import RiskEvaluator

evaluator = RiskEvaluator(thresholds=[0.3, 0.5, 0.7])

# Load predictions and GTs
# ... iterate over dataset ...

for pred, gt in zip(predictions, ground_truths):
    evaluator.add_batch(pred, gt)

# Compute and print metrics
metrics = evaluator.compute_metrics()
evaluator.print_metrics()
```

## Troubleshooting

### Error: "Risk labels file not found"

```bash
# Make sure you ran Step 1
ls data/emergence_risk_v5_full/risk_labels_train.pkl

# If missing, generate labels
python tools/create_risk_labels.py \
    --dataroot data/nuscenes \
    --version v1.0-mini \
    --output_dir data/emergence_risk_v5
```

### Error: "CUDA out of memory"

Reduce batch size in config:
```python
data = dict(
    samples_per_gpu=1,  # Reduce this
    ...
)
```

### Error: "Shape mismatch"

Check that BEV dimensions match:
```python
# In risk_head config
risk_head = dict(
    bev_h=50,  # Must match BEVFormer's BEV grid
    bev_w=50,
    ...
)
```

### Tests Fail with Import Errors

```bash
# Make sure BEVFormer is installed
cd BEVFormer
pip install -e .

# Check imports
python -c "import projects.mmdet3d_plugin"
```

## Next Steps

1. **Experiment with different configurations**:
   - Try different risk loss weights
   - Enable risk-guided attention
   - Adjust learning rates

2. **Full dataset training**:
   - Generate full risk labels for v1.0-trainval
   - Train for full 24 epochs
   - Compare with baseline BEVFormer

3. **Ablation studies**:
   - Baseline vs +Risk vs +Attention
   - Different attention types (spatial/channel/both)
   - Impact of risk threshold filtering

4. **Analyze results**:
   - Visualize failure cases
   - Risk vs detection performance correlation
   - High-risk region detection accuracy

## Performance Targets

Expected results after full training on v1.0-trainval:

### Risk Metrics
- MSE: < 0.02
- MAE: < 0.10
- IoU@0.5: > 0.60
- Pearson R: > 0.80

### Detection Metrics (v1.0-mini)
- mAP: 0.35-0.38 (baseline: ~0.35)
- NDS: 0.43-0.46 (baseline: ~0.43)

## Success Criteria

You're ready to proceed if:

- [✅] All validation tests pass
- [✅] Training runs without errors
- [✅] Risk loss decreases during training
- [✅] Visualizations show reasonable risk predictions
- [✅] No NaN or Inf in losses

## Need Help?

1. Check full documentation: [RISK_GUIDED_BEVFORMER_README.md](RISK_GUIDED_BEVFORMER_README.md)
2. Review risk label spec: [docs/Risk_Label_Specification.md](docs/Risk_Label_Specification.md)
3. Open an issue on GitHub
4. Check BEVFormer original docs for general questions

---

**Estimated Total Time**: 30-45 minutes (excluding full training)

**Ready to start?** → Begin with Step 1! 🚀
