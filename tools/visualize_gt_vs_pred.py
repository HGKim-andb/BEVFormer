#!/usr/bin/env python3
"""
GT vs Prediction Risk Map Comparison Visualization

Usage:
    python tools/visualize_gt_vs_pred.py \
        --checkpoint work_dirs/risk_w500_12ep/epoch_12.pth \
        --output visualizations/gt_vs_pred_comparison.png
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mmcv import Config
from mmcv.runner import load_checkpoint
import os
import sys

# Add projects to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'projects'))

import mmdet3d_plugin  # noqa: F401
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='GT vs Pred Risk Map Comparison')
    parser.add_argument('--config', type=str,
                        default='projects/configs/bevformer/bevformer_risk_tiny_attention.py',
                        help='Config file path')
    parser.add_argument('--checkpoint', type=str,
                        default='work_dirs/risk_w500_12ep/epoch_12.pth',
                        help='Checkpoint file path')
    parser.add_argument('--output', type=str,
                        default='visualizations/gt_vs_pred_comparison.png',
                        help='Output image path')
    parser.add_argument('--num-samples', type=int, default=4,
                        help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figure')
    return parser.parse_args()


def get_risk_prediction(model, dataset, idx, device):
    """Get GT and predicted risk maps for a sample."""

    # Get data
    data = dataset[idx]

    # Get GT risk map
    gt_risk_map = None
    if 'gt_risk_map' in data:
        gt_data = data['gt_risk_map']
        if hasattr(gt_data, 'data'):
            gt_risk_map = gt_data.data
        else:
            gt_risk_map = gt_data
        if isinstance(gt_risk_map, torch.Tensor):
            gt_risk_map = gt_risk_map.numpy()

    # Prepare input for model
    if isinstance(data['img'], list):
        img_item = data['img'][0]
        img_tensor = img_item.data if hasattr(img_item, 'data') else img_item
        img = img_tensor.unsqueeze(0).to(device)

        meta_item = data['img_metas'][0]
        img_metas = [meta_item.data if hasattr(meta_item, 'data') else meta_item]
    else:
        img_data = data['img'].data if hasattr(data['img'], 'data') else data['img']
        img = img_data.unsqueeze(0).to(device)
        img_metas = [data['img_metas'].data if hasattr(data['img_metas'], 'data') else data['img_metas']]

    if not isinstance(img_metas, list):
        img_metas = [img_metas]

    # Run inference
    with torch.no_grad():
        _, results = model.simple_test(
            img_metas=img_metas,
            img=img,
            prev_bev=None,
            rescale=True
        )

    # Get predicted risk map
    pred_risk_map = results[0].get('risk_map', None)
    if pred_risk_map is not None and isinstance(pred_risk_map, torch.Tensor):
        pred_risk_map = pred_risk_map.cpu().numpy()

    # Get sample info
    sample_info = {
        'idx': idx,
        'token': img_metas[0].get('sample_idx', f'sample_{idx}'),
    }

    return gt_risk_map, pred_risk_map, sample_info


def plot_comparison(gt_maps, pred_maps, sample_infos, output_path, dpi=300):
    """Create side-by-side GT vs Prediction comparison."""

    num_samples = len(gt_maps)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, (gt, pred, info) in enumerate(zip(gt_maps, pred_maps, sample_infos)):
        # GT Risk Map
        ax = axes[i, 0]
        if gt is not None:
            gt_squeezed = np.squeeze(gt)
            im = ax.imshow(gt_squeezed, cmap='hot', vmin=0, vmax=1, origin='lower')
            ax.set_title(f'GT Risk Map\nMax: {gt_squeezed.max():.4f}, Mean: {gt_squeezed.mean():.4f}', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, 'No GT Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('GT Risk Map', fontsize=10)
        ax.set_ylabel(f'Sample {info["idx"]}', fontsize=10, fontweight='bold')
        ax.set_xlabel('X (BEV)')

        # Predicted Risk Map
        ax = axes[i, 1]
        if pred is not None:
            pred_squeezed = np.squeeze(pred)
            im = ax.imshow(pred_squeezed, cmap='hot', vmin=0, vmax=1, origin='lower')
            ax.set_title(f'Predicted Risk Map\nMax: {pred_squeezed.max():.4f}, Mean: {pred_squeezed.mean():.4f}', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, 'No Prediction', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Predicted Risk Map', fontsize=10)
        ax.set_xlabel('X (BEV)')

        # Difference Map (|GT - Pred|)
        ax = axes[i, 2]
        if gt is not None and pred is not None:
            gt_squeezed = np.squeeze(gt)
            pred_squeezed = np.squeeze(pred)
            diff = np.abs(gt_squeezed - pred_squeezed)
            im = ax.imshow(diff, cmap='coolwarm', vmin=0, vmax=1, origin='lower')
            mse = np.mean((gt_squeezed - pred_squeezed) ** 2)
            mae = np.mean(np.abs(gt_squeezed - pred_squeezed))
            ax.set_title(f'|GT - Pred|\nMSE: {mse:.6f}, MAE: {mae:.4f}', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Difference', fontsize=10)
        ax.set_xlabel('X (BEV)')

    # Main title
    fig.suptitle('Risk Map: Ground Truth vs Prediction Comparison\n(weight=500, 12 epochs)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved comparison to: {output_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("GT vs Prediction Risk Map Comparison")
    print("=" * 60)

    # Load config
    print(f"\nLoading config: {args.config}")
    cfg = Config.fromfile(args.config)

    # Build dataset (validation set)
    print("Building dataset...")
    val_cfg = cfg.data.val.copy()
    dataset = build_dataset(val_cfg)
    print(f"Dataset size: {len(dataset)}")

    # Build model
    print("Building model...")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Available checkpoints in work_dirs/:")
        for root, dirs, files in os.walk('work_dirs'):
            for f in files:
                if f.endswith('.pth'):
                    print(f"  {os.path.join(root, f)}")
        return

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # Move to device and set eval mode
    model = model.to(args.device)
    model.eval()

    # Debug: Check risk_head weights
    print("\n=== Risk Head Debug ===")
    if hasattr(model, 'risk_head') and model.risk_head is not None:
        print("Risk head exists")
        for name, param in model.risk_head.named_parameters():
            print(f"  {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")

        # Check BatchNorm buffers
        print("\nBatchNorm buffers:")
        for name, buf in model.risk_head.named_buffers():
            if 'running' in name:
                print(f"  {name}: mean={buf.mean().item():.6f}, std={buf.std().item():.6f}")
    else:
        print("WARNING: No risk_head found!")

    # Select samples (spread across dataset)
    num_samples = min(args.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    print(f"\nSelected sample indices: {indices.tolist()}")

    # Collect predictions
    gt_maps = []
    pred_maps = []
    sample_infos = []

    for idx in indices:
        print(f"\nProcessing sample {idx}...")
        try:
            gt, pred, info = get_risk_prediction(model, dataset, idx, args.device)
            gt_maps.append(gt)
            pred_maps.append(pred)
            sample_infos.append(info)

            if gt is not None:
                print(f"  GT shape: {np.squeeze(gt).shape}, range: [{np.squeeze(gt).min():.4f}, {np.squeeze(gt).max():.4f}]")
            else:
                print(f"  GT: None")

            if pred is not None:
                print(f"  Pred shape: {np.squeeze(pred).shape}, range: [{np.squeeze(pred).min():.4f}, {np.squeeze(pred).max():.4f}]")
            else:
                print(f"  Pred: None")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Create comparison plot
    print("\nCreating comparison plot...")
    plot_comparison(gt_maps, pred_maps, sample_infos, args.output, args.dpi)

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    all_mse = []
    all_mae = []
    for gt, pred in zip(gt_maps, pred_maps):
        if gt is not None and pred is not None:
            gt_sq = np.squeeze(gt)
            pred_sq = np.squeeze(pred)
            mse = np.mean((gt_sq - pred_sq) ** 2)
            mae = np.mean(np.abs(gt_sq - pred_sq))
            all_mse.append(mse)
            all_mae.append(mae)

    if all_mse:
        print(f"Average MSE: {np.mean(all_mse):.6f}")
        print(f"Average MAE: {np.mean(all_mae):.6f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
