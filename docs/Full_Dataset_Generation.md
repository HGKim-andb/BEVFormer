# nuScenes Full Dataset V5 Risk Label Generation Guide

## 📊 Dataset Overview

- **Version**: v1.0-trainval
- **Total scenes**: 850
- **Total samples**: 34,149
- **Estimated processing time**: 15-20 hours (with parallel processing)
- **Storage requirement**: ~15-20 GB for risk labels

---

## 🚀 Quick Start (3 Steps)

### Step 1: Generate Labels in Batches

```bash
# Interactive batch generation
bash tools/generate_full_dataset_batches.sh

# Options:
#   1) All batches (1-10) - Full dataset
#   2) Single batch - Test with one batch first (recommended)
#   3) Range of batches - Process specific range
```

**Recommendation**: Start with **Option 2 (Single batch)** to test and verify:
```
Choice [1/2/3]: 2
Enter batch number (1-10): 1
```

This will process ~85 scenes (~3,400 samples) in 2-3 hours.

### Step 2: Merge Batches

After all batches are complete:

```bash
python tools/merge_risk_batches.py \
    --input_dirs data/emergence_risk_v5_full_batch_* \
    --output_dir data/emergence_risk_v5_full
```

### Step 3: Verify Results

```bash
python tools/compare_versions.py  # Compare with V1-V4

# Visualize samples from full dataset
python tools/visualize_risk_samples.py \
    --labels data/emergence_risk_v5_full/risk_labels_train.pkl \
    --dataroot /home/hg-main/data2/datasets/nuscenes/data/nuscenes \
    --version v1.0-trainval \
    --num_samples 20 \
    --min_risk 0.7 \
    --output_dir visualizations/full_dataset_samples
```

---

## 📋 Detailed Methods

### Method 1: All at Once (Not Recommended)

**Pros**: Simple one-command execution
**Cons**: 15-20 hours continuous run, no checkpoints

```bash
python tools/create_risk_labels.py \
    --dataroot /home/hg-main/data2/datasets/nuscenes/data/nuscenes \
    --version v1.0-trainval \
    --output_dir data/emergence_risk_v5_full \
    --parallel
```

⚠️ **Warning**: If interrupted, you lose all progress!

### Method 2: Batch Processing (Recommended ⭐)

**Pros**:
- Resume from failures
- Parallel processing per batch
- Can run batches overnight
- Easy to monitor progress

**Cons**: Requires merge step

#### Batch Configuration

| Batch | Scenes | Samples (est.) | Time (est.) |
|-------|--------|----------------|-------------|
| 1     | 0-84   | ~3,400         | 2-3 hours   |
| 2     | 85-169 | ~3,400         | 2-3 hours   |
| 3     | 170-254| ~3,400         | 2-3 hours   |
| ...   | ...    | ...            | ...         |
| 10    | 765-849| ~3,400         | 2-3 hours   |

**Total**: 10 batches × 2.5 hours = **~25 hours** (conservative estimate)

#### Running Batches

**Option A: Interactive (Recommended for first-time)**
```bash
bash tools/generate_full_dataset_batches.sh
```

**Option B: Direct Batch Execution**
```bash
# Run specific batch number (1-10)
BATCH=1
python tools/create_risk_labels.py \
    --dataroot /home/hg-main/data2/datasets/nuscenes/data/nuscenes \
    --version v1.0-trainval \
    --output_dir data/emergence_risk_v5_full_batch_${BATCH} \
    --scenes <scene_list_for_batch> \
    --parallel
```

**Option C: Overnight Batch Queue**
```bash
# Create queue script
cat > run_batches_queue.sh << 'EOF'
#!/bin/bash
for batch in {1..10}; do
    echo "Starting batch $batch at $(date)"
    bash tools/generate_full_dataset_batches.sh <<< "2
$batch"
    echo "Batch $batch complete at $(date)"
done
EOF

chmod +x run_batches_queue.sh

# Run in background with logging
nohup ./run_batches_queue.sh > batch_queue.log 2>&1 &

# Monitor progress
tail -f batch_queue.log
```

---

## 🔧 Advanced Options

### Parallel Processing Optimization

**Default**: Uses `multiprocessing.cpu_count() - 1` cores

**Custom core count**:
```python
# Edit tools/create_risk_labels.py, line ~270
if args.parallel:
    num_workers = 8  # Set custom number of workers
    pool = multiprocessing.Pool(processes=num_workers)
```

### Selective Scene Processing

**By scene name pattern**:
```bash
# Only process scenes from specific locations
python -c "
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-trainval',
                dataroot='/path/to/nuscenes', verbose=False)

# Filter scenes (example: only Singapore scenes)
singapore_scenes = [s['name'] for s in nusc.scene
                   if 'singapore' in nusc.get('log', s['log_token'])['location']]
print(' '.join(singapore_scenes))
" > singapore_scenes.txt

python tools/create_risk_labels.py \
    --scenes $(cat singapore_scenes.txt) \
    --output_dir data/emergence_risk_v5_singapore
```

**By time of day**:
```python
# Get night scenes only
night_scenes = [s['name'] for s in nusc.scene
               if any('night' in desc.lower()
                     for desc in [s['description'], s['name']])]
```

### Memory Optimization

If running out of memory:

```python
# Edit tools/create_risk_labels.py
# Reduce grid step size (line ~90)
step = 4  # Process every 4th cell (default: 2)
```

This reduces memory usage by 4× but makes risk maps coarser.

---

## 📈 Expected Results

Based on mini dataset results (scaled up):

| Metric | Mini (10 scenes) | Full (850 scenes, estimated) |
|--------|------------------|------------------------------|
| Samples | 405 | ~34,149 |
| Max risk (avg) | 0.523 ± 0.326 | 0.52 ± 0.33 |
| Mean risk (avg) | 0.004 ± 0.006 | 0.004 ± 0.006 |
| High-risk cells | 36.9 ± 129.7 | ~37 ± 130 |
| Samples > 0.7 | 40.8% | ~41% |

**Storage breakdown**:
- risk_labels_train.pkl: ~12-15 GB
- risk_labels_val.pkl: ~3-4 GB
- risk_config.json: <1 KB

---

## 🛠️ Troubleshooting

### Issue 1: Out of Memory

**Symptoms**: Process killed, "MemoryError"

**Solutions**:
1. Reduce parallel workers:
   ```python
   num_workers = 4  # Instead of cpu_count()-1
   ```

2. Increase grid step:
   ```python
   step = 4  # Process every 4th cell
   ```

3. Process smaller batches:
   ```bash
   # Split 85 scenes into 2 sub-batches of 42-43 scenes
   ```

### Issue 2: Batch Interrupted

**Symptoms**: Process stopped mid-batch

**Solutions**:
1. Re-run the same batch - it will regenerate from scratch
2. Check batch output directory - if partial results exist, delete before re-running
3. Use `nohup` for long-running batches:
   ```bash
   nohup bash tools/generate_full_dataset_batches.sh > batch.log 2>&1 &
   ```

### Issue 3: Merge Fails with Duplicate Scenes

**Symptoms**: "Warning: Scene already exists"

**Solutions**:
1. Check for duplicate batch directories:
   ```bash
   ls -d data/emergence_risk_v5_full_batch_*
   ```

2. Remove duplicate batches before merging

3. Verify each batch has unique scenes:
   ```bash
   for dir in data/emergence_risk_v5_full_batch_*; do
       python -c "
   import pickle
   with open('$dir/risk_labels_train.pkl', 'rb') as f:
       data = pickle.load(f)
   print(f'$dir: {len(data)} scenes')
   "
   done
   ```

### Issue 4: Slow Processing

**Expected**: ~1.5-2 seconds per sample with parallel processing

**If slower**:
1. Check CPU usage: `htop` or `top`
2. Check disk I/O: `iostat -x 1`
3. Verify parallel processing is enabled:
   ```bash
   python tools/create_risk_labels.py --help | grep parallel
   ```

---

## 📊 Monitoring Progress

### Real-time Progress

```bash
# Watch log file
tail -f batch_queue.log

# Count completed batches
ls -d data/emergence_risk_v5_full_batch_* | wc -l

# Check latest batch progress
ls -lh data/emergence_risk_v5_full_batch_*/risk_labels_train.pkl
```

### Estimate Completion Time

```bash
# If batch 1 took 2.5 hours for 85 scenes
# Remaining: 9 batches × 2.5 hours = 22.5 hours
```

---

## 🎯 Use Cases

### 1. Train BEVFormer with Risk Labels

```python
# In projects/mmdet3d_plugin/datasets/nuscenes_dataset.py
def load_risk_labels(self):
    with open('data/emergence_risk_v5_full/risk_labels_train.pkl', 'rb') as f:
        self.risk_labels = pickle.load(f)
```

### 2. Risk-Based Sample Selection

```python
# Select high-risk samples for training
high_risk_samples = [
    label for scene_labels in labels_dict.values()
    for label in scene_labels
    if label['metadata']['max_risk'] > 0.7
]

print(f"High-risk samples: {len(high_risk_samples)}")
```

### 3. Scene Difficulty Ranking

```python
# Rank scenes by average risk
scene_risks = {
    scene_token: np.mean([s['metadata']['max_risk'] for s in labels])
    for scene_token, labels in labels_dict.items()
}

sorted_scenes = sorted(scene_risks.items(), key=lambda x: x[1], reverse=True)
print("Top 10 hardest scenes:")
for scene_token, avg_risk in sorted_scenes[:10]:
    print(f"  {scene_token}: {avg_risk:.3f}")
```

---

## 📝 Summary Commands

```bash
# 1. Test with single batch
bash tools/generate_full_dataset_batches.sh
# Choose option 2, batch 1

# 2. If successful, queue all batches
nohup bash -c '
for batch in {1..10}; do
    bash tools/generate_full_dataset_batches.sh <<< "2
$batch"
done
' > full_generation.log 2>&1 &

# 3. Monitor progress
tail -f full_generation.log

# 4. After all complete, merge
python tools/merge_risk_batches.py \
    --input_dirs data/emergence_risk_v5_full_batch_* \
    --output_dir data/emergence_risk_v5_full

# 5. Visualize results
python tools/visualize_risk_samples.py \
    --labels data/emergence_risk_v5_full/risk_labels_train.pkl \
    --dataroot /path/to/nuscenes \
    --version v1.0-trainval \
    --num_samples 50 --min_risk 0.7 \
    --output_dir visualizations/full_v5_samples
```

---

## 📚 Related Documentation

- [V5 Algorithm Details](Risk_Calculation_v5.md)
- [Mini Dataset Guide](../nuscenes_mini_가이드.md)
- [Version Comparison](../compare_versions.py)
- [Project Checklist](Project_checklist.md)

---

**Last Updated**: 2025-11-18
**Version**: V5 Continuous Function
