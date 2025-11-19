# Dataset Split Quick Reference

## Current Status

### Mini Dataset (v1.0-mini) - Already Split ✓
```
data/emergence_risk_v5_full/
├── risk_labels_train.pkl  (8 scenes, 324 samples)
└── risk_labels_val.pkl    (2 scenes, 80 samples)
```

**Created with**: `tools/split_risk_labels.py` (simple 8:2 split)

**Config**: `projects/configs/bevformer/bevformer_risk_tiny.py`
- Train: `risk_labels_path='data/emergence_risk_v5_full/risk_labels_train.pkl'`
- Val: `risk_labels_path='data/emergence_risk_v5_full/risk_labels_val.pkl'`

**Status**: Training successfully started

---

## Full Dataset (v1.0-trainval) - To Be Split

### If pkl file is on another computer

See: [FULL_DATASET_SPLIT_GUIDE.md](FULL_DATASET_SPLIT_GUIDE.md)

**Recommended Method**:
```bash
# 1. Copy script to other computer
scp tools/split_risk_labels_official.py other_computer:/path/to/

# 2. SSH and run split on other computer
ssh other_computer
cd /path/to/BEVFormer
python tools/split_risk_labels_official.py \
    --input /path/to/risk_labels_all.pkl \
    --dataroot /path/to/nuscenes \
    --version v1.0-trainval \
    --output_dir /path/to/output

# 3. Copy results back
exit
scp other_computer:/path/to/output/risk_labels_train.pkl data/emergence_risk_v5_full/
scp other_computer:/path/to/output/risk_labels_val.pkl data/emergence_risk_v5_full/
```

**Expected Output**:
```
data/emergence_risk_v5_full/
├── risk_labels_train.pkl  (~5-6GB, 700 scenes, ~28k samples)
└── risk_labels_val.pkl    (~1-1.5GB, 150 scenes, ~6k samples)
```

### If pkl file is local

```bash
python tools/split_risk_labels_official.py \
    --input data/emergence_risk_v5_full/risk_labels_all.pkl \
    --dataroot /path/to/nuscenes \
    --version v1.0-trainval \
    --output_dir data/emergence_risk_v5_full
```

---

## Available Split Scripts

| Script | Method | Requires nuScenes | Use Case |
|--------|--------|-------------------|----------|
| `split_risk_labels.py` | Simple 8:2 split | Yes | Quick testing, mini dataset |
| `split_risk_labels_official.py` | Official train/val split | Yes | Production, full dataset |
| Embedded script in FULL_DATASET_SPLIT_GUIDE.md | Official split | No (only nuscenes-devkit) | Remote split without dataset |

---

## Split Comparison

### Simple Split (for mini dataset)
```python
# First N scenes → train, rest → val
train_scenes = scenes[:8]  # 80%
val_scenes = scenes[8:]    # 20%
```

**Pros**:
- Simple, fast
- Good for quick testing

**Cons**:
- Not aligned with official benchmark
- May have scene distribution bias

### Official Split (for full dataset)
```python
from nuscenes.utils.splits import create_splits_scenes

splits = create_splits_scenes()
train_scene_names = splits['train']  # 700 predefined scenes
val_scene_names = splits['val']      # 150 predefined scenes
```

**Pros**:
- Aligned with nuScenes benchmark
- Ensures fair comparison
- Balanced scene distribution

**Cons**:
- Requires nuscenes-devkit
- More complex

---

## Config Update for Full Dataset

After splitting full dataset, update config:

```python
# projects/configs/bevformer/bevformer_risk_base.py

data = dict(
    train=dict(
        type='NuScenesRiskDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_infos_temporal_train.pkl',
        risk_labels_path='data/emergence_risk_v5_full/risk_labels_train.pkl',
        # ... rest
    ),
    val=dict(
        type='NuScenesRiskDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_infos_temporal_val.pkl',
        risk_labels_path='data/emergence_risk_v5_full/risk_labels_val.pkl',
        # ... rest
    ),
)
```

---

## Troubleshooting

### "Risk labels file not found"
**Check**:
```bash
ls -lh data/emergence_risk_v5_full/risk_labels_*.pkl
ls -lh data/emergence_risk_v5_full/risk_labels_*.pkl
```

### "Scene name not in labels"
**Solution**: Risk labels must include `scene_name` field. Regenerate labels with:
```bash
python tools/create_risk_labels.py --save_scene_name
```

### "Memory error loading large pkl"
**Solution**: Use streaming or chunked loading (see FULL_DATASET_SPLIT_GUIDE.md)

---

## Summary

**Current Setup** (Mini Dataset):
- ✅ Split completed: 8 train / 2 val scenes
- ✅ Config updated
- ✅ Training started successfully

**Next Steps** (Full Dataset):
1. Access pkl file on other computer
2. Run split using recommended method
3. Transfer split files
4. Update config paths
5. Start full training

**Key Files**:
- [FULL_DATASET_SPLIT_GUIDE.md](FULL_DATASET_SPLIT_GUIDE.md) - Detailed splitting guide
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Training instructions
- [TRAINING_READY.md](TRAINING_READY.md) - Pre-training checklist
