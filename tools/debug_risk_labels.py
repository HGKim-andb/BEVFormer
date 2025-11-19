#!/usr/bin/env python3
"""
Debug risk labels matching issue
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
import numpy as np
from mmcv import Config
from mmdet.datasets import build_dataset
import projects.mmdet3d_plugin

# Load config
cfg = Config.fromfile('projects/configs/bevformer/bevformer_risk_tiny.py')

# Build validation dataset
dataset_cfg = cfg.data.val.copy()
dataset_cfg.pop('samples_per_gpu', None)
dataset_cfg.pop('workers_per_gpu', None)

dataset = build_dataset(dataset_cfg)

print(f"Dataset: {len(dataset)} samples")
print(f"Dataset type: {type(dataset)}")

# Check first few sample tokens
print("\nFirst 10 sample tokens from dataset:")
for i in range(min(10, len(dataset))):
    info = dataset.data_infos[i]
    token = info['token']
    print(f"  {i}: {token}")

# Check risk labels
print("\nRisk labels info:")
print(f"  Total scenes in risk_labels_dict: {len(dataset.risk_labels_dict)}")
print(f"  Total samples in risk_map_dict: {len(dataset.risk_map_dict)}")

# Check first few risk label sample tokens
print("\nFirst 10 sample tokens from risk_map_dict:")
for i, (token, label) in enumerate(list(dataset.risk_map_dict.items())[:10]):
    print(f"  {i}: {token} (max_risk: {label['metadata'].get('max_risk', 0):.3f})")

# Check if any match
print("\nChecking for matches...")
dataset_tokens = set(info['token'] for info in dataset.data_infos[:100])
risk_tokens = set(dataset.risk_map_dict.keys())

matches = dataset_tokens & risk_tokens
print(f"  Dataset tokens (first 100): {len(dataset_tokens)}")
print(f"  Risk label tokens: {len(risk_tokens)}")
print(f"  Matches: {len(matches)}")

if len(matches) > 0:
    print(f"\n✅ Found {len(matches)} matches")
    print("Example matches:")
    for token in list(matches)[:5]:
        risk_label = dataset.risk_map_dict[token]
        print(f"  {token[:8]}... max_risk={risk_label['metadata'].get('max_risk', 0):.3f}")
else:
    print("\n❌ NO MATCHES FOUND!")
    print("\nThis means the validation dataset is using different samples than risk labels.")
    print("Possible causes:")
    print("1. Risk labels were created from train set, not val set")
    print("2. Different nuScenes version (mini vs full)")
    print("3. Different temporal info pkl (train vs val)")
