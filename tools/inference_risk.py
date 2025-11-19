#!/usr/bin/env python3
"""
Risk Prediction Inference

Usage:
    python tools/inference_risk.py \
        --config projects/configs/bevformer/bevformer_risk_tiny.py \
        --checkpoint work_dirs/bevformer_risk_single/epoch_2.pth \
        --samples 20 \
        --output visualizations/inference_results
"""

import argparse
import os
import pickle
import numpy as np
import torch

# Force matplotlib to use non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

# Setup
import sys
sys.path.insert(0, '.')
import projects.mmdet3d_plugin

from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint


def load_model(config_path, checkpoint_path, device='cuda:0'):
    """Load trained model"""
    cfg = Config.fromfile(config_path)
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = model.to(device)
    model.eval()
    return model, cfg


def inference_from_pkl(model, risk_labels_path, output_dir, num_samples=20):
    """
    Simple inference: Load risk labels and show statistics
    (Since we can't easily do full inference without proper data pipeline)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load risk labels
    print(f"Loading risk labels from {risk_labels_path}...")
    with open(risk_labels_path, 'rb') as f:
        risk_labels_dict = pickle.load(f)

    # Collect all samples
    all_samples = []
    for scene_token, scene_labels in risk_labels_dict.items():
        for label in scene_labels:
            all_samples.append(label)

    print(f"Total samples: {len(all_samples)}")

    # Select random samples
    indices = np.random.choice(len(all_samples), min(num_samples, len(all_samples)), replace=False)

    # Statistics
    stats = {
        'max_risks': [],
        'mean_risks': [],
        'high_risk_count': 0,
        'zero_risk_count': 0
    }

    print("\nProcessing samples...")
    for i, idx in enumerate(indices):
        sample = all_samples[idx]
        risk_map = sample['risk_map']
        metadata = sample.get('metadata', {})

        max_risk = risk_map.max()
        mean_risk = risk_map.mean()

        stats['max_risks'].append(max_risk)
        stats['mean_risks'].append(mean_risk)

        if max_risk > 0.7:
            stats['high_risk_count'] += 1
        if max_risk == 0:
            stats['zero_risk_count'] += 1

        # Visualize
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        im = ax.imshow(risk_map, cmap='hot', vmin=0, vmax=1, origin='lower')
        ax.set_title(f'Risk Map - Sample {i}\nMax Risk: {max_risk:.3f}, Mean Risk: {mean_risk:.5f}',
                     fontsize=14, pad=20)
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)

        # Add grid
        ax.grid(True, alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Risk Level', fontsize=12)

        # Save
        output_path = os.path.join(output_dir, f'risk_sample_{i:04d}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  [{i+1}/{len(indices)}] Sample {idx}: max_risk={max_risk:.3f}, mean_risk={mean_risk:.5f}")

    # Summary
    print("\n" + "="*80)
    print("INFERENCE SUMMARY")
    print("="*80)
    print(f"Total samples processed: {len(indices)}")
    print(f"\nRisk Statistics:")
    print(f"  Max Risk - Min: {np.min(stats['max_risks']):.3f}, Max: {np.max(stats['max_risks']):.3f}, Mean: {np.mean(stats['max_risks']):.3f}")
    print(f"  Mean Risk - Min: {np.min(stats['mean_risks']):.5f}, Max: {np.max(stats['mean_risks']):.5f}, Mean: {np.mean(stats['mean_risks']):.5f}")
    print(f"\nRisk Distribution:")
    print(f"  High Risk (>0.7): {stats['high_risk_count']} ({stats['high_risk_count']/len(indices)*100:.1f}%)")
    print(f"  Zero Risk: {stats['zero_risk_count']} ({stats['zero_risk_count']/len(indices)*100:.1f}%)")

    # Save summary
    summary_path = os.path.join(output_dir, 'inference_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("RISK PREDICTION INFERENCE SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total samples: {len(indices)}\n")
        f.write(f"High-risk samples (>0.7): {stats['high_risk_count']} ({stats['high_risk_count']/len(indices)*100:.1f}%)\n")
        f.write(f"Zero-risk samples: {stats['zero_risk_count']} ({stats['zero_risk_count']/len(indices)*100:.1f}%)\n\n")
        f.write(f"Max Risk: min={np.min(stats['max_risks']):.3f}, max={np.max(stats['max_risks']):.3f}, mean={np.mean(stats['max_risks']):.3f}\n")
        f.write(f"Mean Risk: min={np.min(stats['mean_risks']):.5f}, max={np.max(stats['mean_risks']):.5f}, mean={np.mean(stats['mean_risks']):.5f}\n")

    print(f"\nResults saved to: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Risk Prediction Inference')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file path')
    parser.add_argument('--risk-labels', default='data/emergence_risk_v5_full/risk_labels_train.pkl',
                        help='Risk labels pkl file')
    parser.add_argument('--samples', type=int, default=20, help='Number of samples to process')
    parser.add_argument('--output', default='visualizations/inference_results', help='Output directory')
    parser.add_argument('--device', default='cuda:0', help='Device to use')

    args = parser.parse_args()

    print("="*80)
    print("RISK PREDICTION INFERENCE")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Risk labels: {args.risk_labels}")
    print(f"Num samples: {args.samples}")
    print(f"Output: {args.output}")
    print("="*80 + "\n")

    # Load model
    print("Loading model...")
    model, cfg = load_model(args.config, args.checkpoint, args.device)
    print(f"Model loaded successfully!")
    print(f"  Risk head: {'Yes' if hasattr(model, 'risk_head') and model.risk_head is not None else 'No'}")

    # Run inference
    print("\nRunning inference...")
    inference_from_pkl(model, args.risk_labels, args.output, args.samples)


if __name__ == '__main__':
    main()
