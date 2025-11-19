#!/usr/bin/env python3
"""
Subsample risk labels to reduce dataset size

Usage:
    python tools/subsample_risk_labels.py \
        --input data/emergence_risk_v5_full/risk_labels_train.pkl \
        --output data/emergence_risk_v5_full/risk_labels_train_20pct.pkl \
        --ratio 0.2
"""

import argparse
import pickle
import numpy as np
from pathlib import Path


def subsample_risk_labels(input_path, output_path, ratio=0.2, seed=42):
    """
    Subsample risk labels by randomly selecting scenes

    Args:
        input_path: Input pkl file path
        output_path: Output pkl file path
        ratio: Sampling ratio (0.0 to 1.0)
        seed: Random seed for reproducibility
    """
    print("="*80)
    print(f"SUBSAMPLING RISK LABELS TO {ratio*100:.0f}%")
    print("="*80)

    # Load original labels
    print(f"\n1. Loading original labels from {input_path}...")
    with open(input_path, 'rb') as f:
        original_labels = pickle.load(f)

    original_scenes = len(original_labels)
    original_samples = sum(len(labels) for labels in original_labels.values())

    print(f"   Original: {original_scenes} scenes, {original_samples} samples")

    # Sample scenes
    np.random.seed(seed)
    scene_tokens = list(original_labels.keys())
    num_scenes_to_keep = max(1, int(len(scene_tokens) * ratio))

    sampled_scene_tokens = np.random.choice(
        scene_tokens,
        size=num_scenes_to_keep,
        replace=False
    )

    print(f"\n2. Sampling {num_scenes_to_keep} scenes ({ratio*100:.1f}%)...")

    # Create subsampled labels
    subsampled_labels = {
        token: original_labels[token]
        for token in sampled_scene_tokens
    }

    subsampled_samples = sum(len(labels) for labels in subsampled_labels.values())

    print(f"   Subsampled: {len(subsampled_labels)} scenes, {subsampled_samples} samples")
    print(f"   Reduction: {(1 - len(subsampled_labels)/original_scenes)*100:.1f}% scenes, "
          f"{(1 - subsampled_samples/original_samples)*100:.1f}% samples")

    # Statistics
    print(f"\n3. Statistics:")
    original_max_risks = []
    subsampled_max_risks = []

    for labels in original_labels.values():
        for label in labels:
            original_max_risks.append(label['risk_map'].max())

    for labels in subsampled_labels.values():
        for label in labels:
            subsampled_max_risks.append(label['risk_map'].max())

    print(f"   Original max risk: mean={np.mean(original_max_risks):.3f}, "
          f"std={np.std(original_max_risks):.3f}")
    print(f"   Subsampled max risk: mean={np.mean(subsampled_max_risks):.3f}, "
          f"std={np.std(subsampled_max_risks):.3f}")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n4. Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(subsampled_labels, f)

    # File size comparison
    import os
    original_size = os.path.getsize(input_path) / (1024**3)  # GB
    subsampled_size = os.path.getsize(output_path) / (1024**3)  # GB

    print(f"   Original size: {original_size:.2f} GB")
    print(f"   Subsampled size: {subsampled_size:.2f} GB")
    print(f"   Size reduction: {(1 - subsampled_size/original_size)*100:.1f}%")

    print("\n" + "="*80)
    print("✅ SUBSAMPLING COMPLETE")
    print("="*80)

    return subsampled_labels


def main():
    parser = argparse.ArgumentParser(description='Subsample risk labels')
    parser.add_argument('--input', type=str, required=True,
                        help='Input risk labels pkl file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output risk labels pkl file')
    parser.add_argument('--ratio', type=float, default=0.2,
                        help='Sampling ratio (0.0-1.0, default: 0.2 for 20%%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Validate ratio
    if not 0.0 < args.ratio <= 1.0:
        raise ValueError(f"Ratio must be between 0.0 and 1.0, got {args.ratio}")

    # Subsample
    subsample_risk_labels(args.input, args.output, args.ratio, args.seed)


if __name__ == '__main__':
    main()
