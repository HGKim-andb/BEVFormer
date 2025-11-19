#!/usr/bin/env python3
"""
Simple Risk Visualization (Works with trained checkpoint)

Usage:
    python tools/visualize_risk_simple.py \
        work_dirs/bevformer_risk_single/epoch_2.pth \
        --num-samples 5
"""

import argparse
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Checkpoint path')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--output-dir', default='visualizations/risk_simple', help='Output directory')
    parser.add_argument('--split', default='train', choices=['train', 'val'], help='Dataset split')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load risk labels
    if args.split == 'train':
        risk_pkl = 'data/emergence_risk_v5_full/risk_labels_train.pkl'
    else:
        risk_pkl = 'data/emergence_risk_v5_full/risk_labels_val.pkl'

    print(f"Loading risk labels from {risk_pkl}...")
    with open(risk_pkl, 'rb') as f:
        risk_labels_dict = pickle.load(f)

    # Collect all samples
    all_samples = []
    for scene_token, scene_labels in risk_labels_dict.items():
        for label in scene_labels:
            all_samples.append(label)

    print(f"Total samples: {len(all_samples)}")

    # Select random samples
    indices = np.random.choice(len(all_samples), min(args.num_samples, len(all_samples)), replace=False)

    # Visualize GT only (no prediction for now)
    print("\nVisualizing GT risk maps...")
    stats = {'max_risks': [], 'mean_risks': [], 'nonzero_ratios': []}

    for i, idx in enumerate(indices):
        sample = all_samples[idx]
        risk_map = sample['risk_map']

        # Statistics
        max_risk = risk_map.max()
        mean_risk = risk_map.mean()
        nonzero_ratio = (risk_map > 0).sum() / risk_map.size

        stats['max_risks'].append(max_risk)
        stats['mean_risks'].append(mean_risk)
        stats['nonzero_ratios'].append(nonzero_ratio)

        # Visualize
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(risk_map, cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'GT Risk Map\nMax: {max_risk:.3f}, Mean: {mean_risk:.4f}, Non-zero: {nonzero_ratio*100:.1f}%')
        ax.axis('off')
        plt.colorbar(im, ax=ax)

        output_path = os.path.join(args.output_dir, f'gt_sample_{i:04d}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  [{i+1}/{len(indices)}] Sample {idx}: max={max_risk:.3f}, mean={mean_risk:.4f}")

    # Summary statistics
    print("\n" + "="*60)
    print("GROUND TRUTH STATISTICS")
    print("="*60)
    print(f"Max Risk:")
    print(f"  Min:  {np.min(stats['max_risks']):.4f}")
    print(f"  Max:  {np.max(stats['max_risks']):.4f}")
    print(f"  Mean: {np.mean(stats['max_risks']):.4f}")

    print(f"\nMean Risk:")
    print(f"  Min:  {np.min(stats['mean_risks']):.6f}")
    print(f"  Max:  {np.max(stats['mean_risks']):.6f}")
    print(f"  Mean: {np.mean(stats['mean_risks']):.6f}")

    print(f"\nNon-zero Ratio:")
    print(f"  Min:  {np.min(stats['nonzero_ratios'])*100:.2f}%")
    print(f"  Max:  {np.max(stats['nonzero_ratios'])*100:.2f}%")
    print(f"  Mean: {np.mean(stats['nonzero_ratios'])*100:.2f}%")

    # Save summary
    summary_path = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"GROUND TRUTH RISK LABELS SUMMARY\n")
        f.write(f"="*60 + "\n")
        f.write(f"Dataset: {args.split}\n")
        f.write(f"Num samples visualized: {len(indices)}\n\n")
        f.write(f"Max Risk: min={np.min(stats['max_risks']):.4f}, max={np.max(stats['max_risks']):.4f}, mean={np.mean(stats['max_risks']):.4f}\n")
        f.write(f"Mean Risk: min={np.min(stats['mean_risks']):.6f}, max={np.max(stats['mean_risks']):.6f}, mean={np.mean(stats['mean_risks']):.6f}\n")
        f.write(f"Non-zero: min={np.min(stats['nonzero_ratios'])*100:.2f}%, max={np.max(stats['nonzero_ratios'])*100:.2f}%, mean={np.mean(stats['nonzero_ratios'])*100:.2f}%\n")

    print(f"\nVisualizations saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_path}")
    print("="*60)

if __name__ == '__main__':
    main()
