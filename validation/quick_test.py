"""
Quick test to verify dataset registration
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("="*80)
print("QUICK DATASET REGISTRATION TEST")
print("="*80)

# Step 1: Import mmdet3d_plugin
print("\n1. Importing projects.mmdet3d_plugin...")
try:
    import projects.mmdet3d_plugin
    print("   ✅ Successfully imported mmdet3d_plugin")
except Exception as e:
    print(f"   ❌ Failed to import: {e}")
    sys.exit(1)

# Step 2: Import dataset classes
print("\n2. Importing NuScenesRiskDataset...")
try:
    from projects.mmdet3d_plugin.datasets import NuScenesRiskDataset, NuScenesRiskDatasetVal
    print("   ✅ Successfully imported NuScenesRiskDataset")
    print("   ✅ Successfully imported NuScenesRiskDatasetVal")
except Exception as e:
    print(f"   ❌ Failed to import: {e}")
    sys.exit(1)

# Step 3: Check DATASETS registry
print("\n3. Checking DATASETS registry...")
try:
    from mmdet.datasets import DATASETS

    registered_datasets = list(DATASETS.module_dict.keys())
    print(f"   Total registered datasets: {len(registered_datasets)}")

    # Check if our datasets are registered
    if 'NuScenesRiskDataset' in registered_datasets:
        print("   ✅ NuScenesRiskDataset is registered")
    else:
        print("   ❌ NuScenesRiskDataset is NOT registered")
        print(f"   Available datasets: {registered_datasets[:10]}...")

    if 'NuScenesRiskDatasetVal' in registered_datasets:
        print("   ✅ NuScenesRiskDatasetVal is registered")
    else:
        print("   ❌ NuScenesRiskDatasetVal is NOT registered")

except Exception as e:
    print(f"   ❌ Failed to check registry: {e}")
    sys.exit(1)

# Step 4: Try to build dataset
print("\n4. Trying to build NuScenesRiskDataset...")
try:
    from mmdet.datasets import build_dataset

    # Create minimal config dict (don't use Config object)
    cfg_dict = dict(
        type='NuScenesRiskDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_infos_temporal_train.pkl',
        pipeline=[],
        classes=['car'],
        modality=dict(use_camera=True),
        test_mode=False,
        use_risk=True,
        risk_labels_path='data/emergence_risk_v5_full/risk_labels_train.pkl',
    )

    print(f"   Config type: {cfg_dict['type']}")

    # Try to build
    try:
        dataset = build_dataset(cfg_dict)
        print(f"   ✅ Successfully built dataset!")
        print(f"   Dataset type: {type(dataset)}")
    except FileNotFoundError as e:
        print(f"   ⚠️  Dataset built but files not found (expected): {e}")
        print(f"   This is OK - just means nuScenes data is not downloaded")
    except Exception as e:
        print(f"   ❌ Failed to build dataset: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"   ❌ Failed in build step: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
