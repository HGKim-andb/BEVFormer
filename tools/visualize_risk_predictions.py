#!/usr/bin/env python3
"""
Visualize Risk Predictions

Usage:
    python tools/visualize_risk_predictions.py \
        --config projects/configs/bevformer/bevformer_risk_tiny.py \
        --checkpoint work_dirs/bevformer_risk_single2/epoch_1.pth \
        --num-samples 10 \
        --output-dir visualizations/risk_predictions
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import mmdet3d_plugin to register custom datasets
import projects.mmdet3d_plugin

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel


def visualize_risk_comparison(gt_risk, pred_risk, sample_info, output_path):
    """Visualize GT vs Predicted risk maps"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # GT Risk Map
    im1 = axes[0].imshow(gt_risk, cmap='hot', vmin=0, vmax=1)
    axes[0].set_title(f'Ground Truth Risk\nMax: {gt_risk.max():.3f}, Mean: {gt_risk.mean():.4f}')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])

    # Predicted Risk Map
    im2 = axes[1].imshow(pred_risk, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title(f'Predicted Risk\nMax: {pred_risk.max():.3f}, Mean: {pred_risk.mean():.4f}')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])

    # Difference Map
    diff = np.abs(gt_risk - pred_risk)
    im3 = axes[2].imshow(diff, cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title(f'Absolute Difference\nMAE: {diff.mean():.4f}, Max: {diff.max():.3f}')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])

    plt.suptitle(f"Sample: {sample_info.get('sample_token', 'unknown')[:8]}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_metrics(gt_risks, pred_risks, threshold=0.7):
    """Calculate risk prediction metrics"""
    metrics = {
        'mse': [],
        'mae': [],
        'max_risk_gt': [],
        'max_risk_pred': [],
        'mean_risk_gt': [],
        'mean_risk_pred': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'zero_prediction_ratio': [],
    }

    for gt, pred in zip(gt_risks, pred_risks):
        # Basic metrics
        mse = ((gt - pred) ** 2).mean()
        mae = np.abs(gt - pred).mean()

        metrics['mse'].append(mse)
        metrics['mae'].append(mae)
        metrics['max_risk_gt'].append(gt.max())
        metrics['max_risk_pred'].append(pred.max())
        metrics['mean_risk_gt'].append(gt.mean())
        metrics['mean_risk_pred'].append(pred.mean())

        # High-risk detection metrics
        gt_high = gt > threshold
        pred_high = pred > threshold

        if gt_high.sum() > 0:
            tp = (gt_high & pred_high).sum()
            fp = (~gt_high & pred_high).sum()
            fn = (gt_high & ~pred_high).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1 = 0

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)

        # Zero prediction ratio
        zero_ratio = (pred == 0).sum() / pred.size
        metrics['zero_prediction_ratio'].append(zero_ratio)

    # Aggregate
    result = {}
    for key, values in metrics.items():
        result[f'{key}_mean'] = np.mean(values)
        result[f'{key}_std'] = np.std(values)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config file')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--output-dir', default='visualizations/risk_predictions', help='Output directory')
    parser.add_argument('--split', default='val', choices=['train', 'val'], help='Dataset split')
    parser.add_argument('--high-risk-only', action='store_true', help='Only visualize high-risk samples')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    cfg = Config.fromfile(args.config)

    # Build dataset
    if args.split == 'val':
        dataset_cfg = cfg.data.val.copy()
    else:
        dataset_cfg = cfg.data.train.copy()

    # Remove training-specific keys
    dataset_cfg.pop('samples_per_gpu', None)
    dataset_cfg.pop('workers_per_gpu', None)

    dataset = build_dataset(dataset_cfg)

    print(f"Dataset: {len(dataset)} samples")

    # Build model
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    print(f"Model loaded from {args.checkpoint}")

    # First, find indices that have risk labels
    print("Finding samples with risk labels...")
    valid_indices = []
    high_risk_indices = []

    for i in range(len(dataset)):
        info = dataset.data_infos[i]
        risk_label = dataset.get_risk_label(info['token'])

        if risk_label is not None:
            valid_indices.append(i)
            if risk_label['metadata'].get('max_risk', 0) > 0.7:
                high_risk_indices.append(i)

    print(f"Found {len(valid_indices)} samples with risk labels out of {len(dataset)} total samples")
    print(f"  High-risk samples (>0.7): {len(high_risk_indices)}")

    if len(valid_indices) == 0:
        print("\n❌ ERROR: No samples with risk labels found!")
        print("This means:")
        print("  1. Risk labels were created for a different dataset split")
        print("  2. The dataset is using different samples than risk labels")
        print("\nTry using --split train to use training samples instead")
        return

    # Select samples
    if args.high_risk_only:
        if len(high_risk_indices) == 0:
            print(f"\n⚠️  No high-risk samples found, using all samples instead")
            sample_indices = np.random.choice(valid_indices, min(args.num_samples, len(valid_indices)), replace=False)
        else:
            sample_indices = np.random.choice(high_risk_indices, min(args.num_samples, len(high_risk_indices)), replace=False)
    else:
        sample_indices = np.random.choice(valid_indices, min(args.num_samples, len(valid_indices)), replace=False)

    # Collect predictions
    gt_risks = []
    pred_risks = []
    sample_infos = []

    print("\nProcessing samples...")
    for idx in sample_indices:
        # Get data
        data = dataset[idx]

        # Get GT risk
        info = dataset.data_infos[idx]
        risk_label = dataset.get_risk_label(info['token'])

        if risk_label is None:
            print(f"  Skipping sample {idx}: no risk label")
            continue

        gt_risk = risk_label['risk_map']

        # Forward pass
        with torch.no_grad():
            # Prepare input - handle DataContainer
            if hasattr(data['img'], 'data'):
                # Test mode: img is already in correct format
                img = data['img'].data[0].unsqueeze(0).cuda() if isinstance(data['img'].data, list) else data['img'].data.unsqueeze(0).cuda()
            else:
                img = data['img'][0].data.unsqueeze(0).cuda()

            if hasattr(data['img_metas'], 'data'):
                img_metas = [data['img_metas'].data[0]] if isinstance(data['img_metas'].data, list) else [data['img_metas'].data]
            else:
                img_metas = [data['img_metas'][0].data]

            # Get BEV features and predict risk
            # simple_test returns (new_prev_bev, bbox_list)
            prev_bev, bbox_list = model.module.simple_test(img_metas, img, rescale=True)

            # Extract risk prediction from bbox_list
            if len(bbox_list) > 0 and 'risk_map' in bbox_list[0]:
                pred_risk = bbox_list[0]['risk_map']
                if isinstance(pred_risk, torch.Tensor):
                    pred_risk = pred_risk.cpu().numpy()
                if pred_risk.ndim == 3:
                    pred_risk = pred_risk[0]  # Remove channel dim [1, 200, 200] -> [200, 200]
            else:
                print(f"  Skipping sample {idx}: no risk prediction in result")
                continue

        gt_risks.append(gt_risk)
        pred_risks.append(pred_risk)
        sample_infos.append({
            'sample_token': info['token'],
            'scene_token': info.get('scene_token', 'unknown'),
            'max_risk_gt': gt_risk.max(),
            'mean_risk_gt': gt_risk.mean(),
        })

        # Visualize
        output_path = os.path.join(args.output_dir, f'sample_{idx:04d}.png')
        visualize_risk_comparison(gt_risk, pred_risk, sample_infos[-1], output_path)

        print(f"  [{len(gt_risks)}/{len(sample_indices)}] Sample {idx}: GT max={gt_risk.max():.3f}, Pred max={pred_risk.max():.3f}")

    # Calculate metrics
    print("\n" + "="*80)
    print("METRICS")
    print("="*80)

    metrics = calculate_metrics(gt_risks, pred_risks, threshold=0.7)

    print(f"\nBasic Metrics:")
    print(f"  MSE:  {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
    print(f"  MAE:  {metrics['mae_mean']:.6f} ± {metrics['mae_std']:.6f}")

    print(f"\nMax Risk:")
    print(f"  GT:   {metrics['max_risk_gt_mean']:.4f} ± {metrics['max_risk_gt_std']:.4f}")
    print(f"  Pred: {metrics['max_risk_pred_mean']:.4f} ± {metrics['max_risk_pred_std']:.4f}")

    print(f"\nMean Risk:")
    print(f"  GT:   {metrics['mean_risk_gt_mean']:.6f} ± {metrics['mean_risk_gt_std']:.6f}")
    print(f"  Pred: {metrics['mean_risk_pred_mean']:.6f} ± {metrics['mean_risk_pred_std']:.6f}")

    print(f"\nHigh-Risk Detection (threshold=0.7):")
    print(f"  Precision: {metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f}")
    print(f"  Recall:    {metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}")
    print(f"  F1:        {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")

    print(f"\nZero Prediction Ratio: {metrics['zero_prediction_ratio_mean']:.4f} ± {metrics['zero_prediction_ratio_std']:.4f}")

    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("RISK PREDICTION METRICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.split}\n")
        f.write(f"Num samples: {len(gt_risks)}\n\n")

        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Visualizations saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
