# Validation & Testing Suite

Comprehensive validation and testing framework for Risk-Guided BEVFormer.

## Overview

This directory contains all validation, testing, evaluation, and visualization tools for the Risk-Guided BEVFormer project.

## Test Modules

### 1. test_model.py - Model Architecture Tests

Tests all model components in isolation:

```bash
python validation/test_model.py
```

**Tests included**:
- ✅ Risk head shape validation (3D/4D inputs)
- ✅ Risk head loss calculation
- ✅ Risk-guided attention mechanism
- ✅ Gradient flow through all layers
- ✅ Memory usage across batch sizes
- ✅ Deterministic output (reproducibility)

**Expected runtime**: ~2-3 minutes

### 2. test_data.py - Data Pipeline Tests

Tests data loading and preprocessing:

```bash
python validation/test_data.py
```

**Tests included**:
- ✅ Risk labels file existence
- ✅ Risk label format validation
- ✅ Dataset creation
- ✅ Single item loading
- ✅ Risk map alignment with BEV

**Expected runtime**: ~1-2 minutes

### 3. integration_test.py - End-to-End Tests

Tests complete pipeline from data to predictions:

```bash
python validation/integration_test.py
```

**Tests included**:
- ✅ End-to-end forward pass
- ✅ Single batch overfitting (learning capability)
- ✅ Multi-GPU compatibility
- ✅ Save/load checkpoint
- ✅ Inference speed benchmark

**Expected runtime**: ~3-5 minutes

### 4. visualize.py - Visualization Tools

Comprehensive visualization utilities:

```bash
# Run test
python validation/visualize.py

# Or use programmatically
from validation.visualize import RiskVisualizer

visualizer = RiskVisualizer(output_dir='visualizations/my_experiment')
fig = visualizer.visualize_risk_comparison(gt_risk, pred_risk, sample_token)
```

**Features**:
- Risk map comparisons (GT vs Pred)
- Risk maps with detection overlays
- Attention weight visualization
- Multi-sample grid comparisons
- Training metrics plots

### 5. evaluate.py - Evaluation Metrics

Evaluation framework with comprehensive metrics:

```bash
# Run test
python validation/evaluate.py

# Or use programmatically
from validation.evaluate import RiskEvaluator

evaluator = RiskEvaluator(thresholds=[0.3, 0.5, 0.7])
evaluator.add_batch(pred_risk, gt_risk)
metrics = evaluator.compute_metrics()
evaluator.print_metrics()
```

**Metrics included**:
- Regression: MSE, RMSE, MAE
- Correlation: Pearson R, Spearman R
- Classification: Precision, Recall, F1, IoU (per threshold)
- Calibration: Max risk MAE, correlation

## Run All Tests

```bash
# Quick validation of all components
cd validation
python test_model.py && \
python test_data.py && \
python integration_test.py

# If all pass, you're good to go! ✅
```

## Usage Examples

### Example 1: Quick Model Validation

```python
import torch
from projects.mmdet3d_plugin.bevformer.dense_heads.risk_head import RiskPredictionHead

# Create model
risk_head = RiskPredictionHead(
    in_channels=256,
    bev_h=50,
    bev_w=50,
    risk_h=200,
    risk_w=200
)

# Test forward pass
bev_features = torch.randn(2, 256, 50, 50)
risk_map = risk_head(bev_features)

print(f"Output shape: {risk_map.shape}")  # [2, 1, 200, 200]
print(f"Value range: [{risk_map.min():.3f}, {risk_map.max():.3f}]")
```

### Example 2: Evaluate Predictions

```python
from validation.evaluate import RiskEvaluator
import pickle
import torch

# Load predictions and GT
with open('predictions.pkl', 'rb') as f:
    predictions = pickle.load(f)

with open('data/emergence_risk_v5/risk_labels_train.pkl', 'rb') as f:
    risk_labels = pickle.load(f)

# Create evaluator
evaluator = RiskEvaluator(thresholds=[0.3, 0.5, 0.7])

# Add all samples
for pred, gt_label in zip(predictions, risk_labels):
    pred_risk = pred['risk_map']
    gt_risk = gt_label['risk_map']
    evaluator.add_batch(pred_risk[None], gt_risk[None])

# Compute metrics
metrics = evaluator.compute_metrics()
evaluator.print_metrics(metrics)
```

### Example 3: Visualize Results

```python
from validation.visualize import RiskVisualizer
import matplotlib.pyplot as plt

visualizer = RiskVisualizer(output_dir='visualizations/experiment_1')

# Load results
# ... load gt_risk, pred_risk, bboxes, etc. ...

# 1. Risk comparison
fig1 = visualizer.visualize_risk_comparison(
    gt_risk=gt_risk,
    pred_risk=pred_risk,
    sample_token=sample_token
)

# 2. Risk with detections
fig2 = visualizer.visualize_risk_with_detections(
    risk_map=pred_risk,
    bboxes_3d=predicted_boxes,
    scores=confidence_scores,
    labels=class_labels,
    sample_token=sample_token
)

# 3. Attention analysis
fig3 = visualizer.visualize_attention_weights(
    attention_weights=attention_map,
    risk_map=pred_risk,
    sample_token=sample_token
)

plt.show()
```

### Example 4: Ablation Study

```python
from validation.evaluate import AblationAnalyzer

# Create analyzer
ablation = AblationAnalyzer()

# Add results from different experiments
ablation.add_experiment('baseline_bevformer', {
    'mAP': 0.354,
    'NDS': 0.428,
    'inference_time_ms': 85.2
})

ablation.add_experiment('bevformer_with_risk', {
    'mAP': 0.361,
    'NDS': 0.435,
    'risk_mae': 0.082,
    'inference_time_ms': 92.5
})

ablation.add_experiment('bevformer_risk_attention', {
    'mAP': 0.375,
    'NDS': 0.448,
    'risk_mae': 0.075,
    'inference_time_ms': 98.3
})

# Print comparison
ablation.print_comparison(baseline='baseline_bevformer')
```

## Test Output Examples

### Successful Test Output

```
================================================================================
RUNNING ALL MODEL TESTS
================================================================================

================================================================================
TEST: Risk Head Shape Validation
================================================================================

📋 Test 1: 3D BEV features [B, H*W, C]
  Input shape: torch.Size([2, 2500, 256])
  Output shape: torch.Size([2, 1, 200, 200])
  Expected: (2, 1, 200, 200)
  ✅ Shape correct!
  ✅ Values in range [0, 1]: [0.023, 0.987]

📋 Test 2: 4D BEV features [B, C, H, W]
  Input shape: torch.Size([2, 256, 50, 50])
  Output shape: torch.Size([2, 1, 200, 200])
  ✅ Shape correct!

✅ Risk Head Shape Test PASSED!

================================================================================
TEST SUMMARY
================================================================================
✅ Risk Head Shape: PASSED
✅ Risk Head Loss: PASSED
✅ Risk-Guided Attention: PASSED
✅ Gradient Flow: PASSED
✅ Memory Usage: PASSED
✅ Deterministic Output: PASSED

Total: 6/6 tests passed

🎉 ALL TESTS PASSED! 🎉
```

### Evaluation Metrics Output

```
================================================================================
RISK EVALUATION METRICS
================================================================================

📊 Regression Metrics:
  MSE:  0.015234
  RMSE: 0.123452
  MAE:  0.082341

📊 Correlation:
  Pearson:  0.847 (p=1.23e-45)
  Spearman: 0.821 (p=3.45e-42)

📊 Threshold = 0.5:
  Precision: 0.7234
  Recall:    0.6845
  F1 Score:  0.7034
  IoU:       0.5423

📊 Calibration (Max Risk):
  MAE:         0.0523
  Correlation: 0.8912

================================================================================
```

## Continuous Integration

For automated testing in CI/CD:

```bash
# Run all tests and exit with code
#!/bin/bash
set -e

echo "Running model tests..."
python validation/test_model.py || exit 1

echo "Running data tests..."
python validation/test_data.py || exit 1

echo "Running integration tests..."
python validation/integration_test.py || exit 1

echo "All tests passed! ✅"
```

## Troubleshooting

### Issue: Tests fail with import errors

```bash
# Ensure project is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/BEVFormer"

# Or install in development mode
cd /path/to/BEVFormer
pip install -e .
```

### Issue: GPU memory errors in tests

```bash
# Run tests on CPU
export CUDA_VISIBLE_DEVICES=""
python validation/test_model.py
```

### Issue: Missing dependencies

```bash
pip install scikit-learn scipy matplotlib opencv-python
```

## Adding New Tests

To add a new test module:

1. Create `test_myfeature.py`
2. Follow the template:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_my_feature():
    """Test description"""
    print("\n" + "="*80)
    print("TEST: My Feature")
    print("="*80)

    # Test implementation
    # ...

    print("\n✅ My Feature Test PASSED!\n")
    return True

def run_all_tests():
    tests = [
        ("My Feature", test_my_feature),
        # Add more tests
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASSED" if result else "FAILED"
        except Exception as e:
            results[test_name] = f"ERROR: {str(e)}"

    # Print summary
    # ...

if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
```

3. Add to CI/CD pipeline

## Performance Benchmarks

Expected test runtimes on typical hardware:

| Test | CPU (i7-10700) | GPU (RTX 3090) |
|------|----------------|----------------|
| test_model.py | ~3 min | ~2 min |
| test_data.py | ~2 min | ~2 min |
| integration_test.py | ~5 min | ~3 min |
| Total | ~10 min | ~7 min |

## Documentation

- [Full README](../RISK_GUIDED_BEVFORMER_README.md)
- [Quick Start Guide](../RISK_QUICKSTART.md)
- [Risk Label Specification](../docs/Risk_Label_Specification.md)

## Contact

For questions about the validation framework:
- Open an issue on GitHub
- Check existing test examples
- Review the main README

---

**Remember**: Always run validation tests before committing changes! ✅
