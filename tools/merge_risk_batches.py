#!/usr/bin/env python3
"""
Merge multiple risk label batches into a single dataset

Usage:
    python tools/merge_risk_batches.py \
        --input_dirs data/emergence_risk_v5_full_batch_* \
        --output_dir data/emergence_risk_v5_full
"""

import argparse
import pickle
import json
from pathlib import Path
from tqdm import tqdm


def merge_batches(input_dirs, output_dir):
    """
    Merge multiple batch directories into a single dataset

    Args:
        input_dirs: List of input directory paths
        output_dir: Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MERGING RISK LABEL BATCHES")
    print("=" * 80)
    print(f"Input batches: {len(input_dirs)}")
    print(f"Output dir:    {output_dir}")
    print("=" * 80)
    print()

    # Initialize merged data
    merged_train = {}
    merged_val = {}
    config = None

    total_train_scenes = 0
    total_train_samples = 0
    total_val_scenes = 0
    total_val_samples = 0

    # Process each batch
    for i, input_dir in enumerate(tqdm(input_dirs, desc="Merging batches")):
        input_path = Path(input_dir)

        if not input_path.exists():
            print(f"⚠️  Warning: {input_path} does not exist, skipping...")
            continue

        # Load config (use first batch's config)
        config_path = input_path / 'risk_config.json'
        if config is None and config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)

        # Load train labels
        train_path = input_path / 'risk_labels_train.pkl'
        if train_path.exists():
            with open(train_path, 'rb') as f:
                batch_train = pickle.load(f)

            # Merge train data
            for scene_token, scene_labels in batch_train.items():
                if scene_token in merged_train:
                    print(f"⚠️  Warning: Scene {scene_token} already exists in train set")
                merged_train[scene_token] = scene_labels
                total_train_scenes += 1
                total_train_samples += len(scene_labels)

        # Load val labels
        val_path = input_path / 'risk_labels_val.pkl'
        if val_path.exists():
            with open(val_path, 'rb') as f:
                batch_val = pickle.load(f)

            # Merge val data
            for scene_token, scene_labels in batch_val.items():
                if scene_token in merged_val:
                    print(f"⚠️  Warning: Scene {scene_token} already exists in val set")
                merged_val[scene_token] = scene_labels
                total_val_scenes += 1
                total_val_samples += len(scene_labels)

    print()
    print("=" * 80)
    print("MERGE SUMMARY")
    print("=" * 80)
    print(f"Train: {total_train_scenes} scenes, {total_train_samples} samples")
    print(f"Val:   {total_val_scenes} scenes, {total_val_samples} samples")
    print("=" * 80)
    print()

    # Save merged data
    print("Saving merged train labels...")
    train_output = output_dir / 'risk_labels_train.pkl'
    with open(train_output, 'wb') as f:
        pickle.dump(merged_train, f)
    print(f"✓ Saved to {train_output}")

    if len(merged_val) > 0:
        print("Saving merged val labels...")
        val_output = output_dir / 'risk_labels_val.pkl'
        with open(val_output, 'wb') as f:
            pickle.dump(merged_val, f)
        print(f"✓ Saved to {val_output}")

    # Save config
    if config is not None:
        print("Saving config...")
        config_output = output_dir / 'risk_config.json'
        with open(config_output, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Saved to {config_output}")

    print()
    print("=" * 80)
    print("✅ MERGE COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print("Files created:")
    print(f"  - risk_labels_train.pkl ({total_train_scenes} scenes)")
    if len(merged_val) > 0:
        print(f"  - risk_labels_val.pkl ({total_val_scenes} scenes)")
    print(f"  - risk_config.json")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Merge risk label batches')
    parser.add_argument('--input_dirs', type=str, nargs='+', required=True,
                        help='Input batch directories (use glob: batch_*)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for merged labels')

    args = parser.parse_args()

    # Expand glob patterns
    from glob import glob
    input_dirs = []
    for pattern in args.input_dirs:
        expanded = glob(pattern)
        if len(expanded) == 0:
            # Not a glob, treat as literal path
            input_dirs.append(pattern)
        else:
            input_dirs.extend(expanded)

    # Sort for consistent ordering
    input_dirs = sorted(input_dirs)

    if len(input_dirs) == 0:
        print("❌ No input directories found!")
        return

    merge_batches(input_dirs, args.output_dir)


if __name__ == '__main__':
    main()
