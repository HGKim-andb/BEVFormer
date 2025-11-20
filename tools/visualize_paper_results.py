#!/usr/bin/env python3
"""
Generate visualizations for paper

This script creates publication-quality visualizations including:
1. Risk map predictions
2. Attention weight visualizations
3. Detection results with/without attention
4. Comparison between baseline and risk-guided attention

Usage:
    python tools/visualize_paper_results.py \
        --config projects/configs/bevformer/bevformer_risk_tiny_attention.py \
        --checkpoint work_dirs/bevformer_risk_attention_fixed/epoch_6.pth \
        --output-dir paper_visualizations \
        --num-samples 10
"""

import argparse
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint
from tqdm import tqdm
import cv2

# Import custom modules
try:
    import projects.mmdet3d_plugin
except ImportError:
    print("Warning: Could not import projects.mmdet3d_plugin")
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='Generate paper visualizations')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file path')
    parser.add_argument('--output-dir', default='paper_visualizations', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for saved figures')
    parser.add_argument('--high-risk-only', action='store_true', help='Only visualize high-risk samples')
    return parser.parse_args()


def setup_matplotlib():
    """Setup matplotlib for publication-quality figures"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 13


def plot_risk_attention_detection(img, risk_map, attention_weights, boxes, scores, labels,
                                   class_names, sample_idx, output_path, gt_risk_map=None):
    """
    Create a comprehensive visualization showing:
    - Input image (multi-view)
    - Ground truth risk map (if available)
    - Predicted risk map
    - Attention weights
    - Detection results overlaid on BEV

    Args:
        img: Input images [N_cams, 3, H, W]
        risk_map: Predicted risk map [1, 200, 200]
        attention_weights: Attention weights [1, 50, 50]
        boxes: Detection boxes
        scores: Detection scores
        labels: Detection labels
        class_names: List of class names
        sample_idx: Sample index
        output_path: Output file path
        gt_risk_map: Ground truth risk map (optional)
    """
    n_plots = 4 if gt_risk_map is not None else 3
    fig = plt.figure(figsize=(20, 5))
    gs = GridSpec(1, n_plots, figure=fig, wspace=0.3)

    # 1. Input image (show front camera)
    ax1 = fig.add_subplot(gs[0, 0])
    if img is not None and len(img) > 0:
        front_img = img[0].cpu().numpy().transpose(1, 2, 0)
        # Denormalize
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        front_img = (front_img * std + mean) / 255.0
        front_img = np.clip(front_img, 0, 1)
        ax1.imshow(front_img)
    ax1.set_title('Front Camera View')
    ax1.axis('off')

    # 2. Ground truth risk map (if available)
    plot_idx = 1
    if gt_risk_map is not None:
        ax_gt = fig.add_subplot(gs[0, plot_idx])
        im_gt = ax_gt.imshow(gt_risk_map.squeeze().cpu().numpy(), cmap='hot', vmin=0, vmax=1)
        ax_gt.set_title('Ground Truth Risk Map')
        ax_gt.set_xlabel('X (BEV)')
        ax_gt.set_ylabel('Y (BEV)')
        plt.colorbar(im_gt, ax=ax_gt, label='Risk Score', fraction=0.046)
        plot_idx += 1

    # 3. Predicted risk map
    ax2 = fig.add_subplot(gs[0, plot_idx])
    im2 = ax2.imshow(risk_map.squeeze().cpu().numpy(), cmap='hot', vmin=0, vmax=1)
    ax2.set_title('Predicted Risk Map')
    ax2.set_xlabel('X (BEV)')
    ax2.set_ylabel('Y (BEV)')
    plt.colorbar(im2, ax=ax2, label='Risk Score', fraction=0.046)

    # Add statistics
    risk_text = f"Mean: {risk_map.mean():.3f}\nMax: {risk_map.max():.3f}\nMin: {risk_map.min():.3f}"
    ax2.text(0.02, 0.98, risk_text, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=8)

    # 4. Attention weights
    ax3 = fig.add_subplot(gs[0, plot_idx + 1])
    im3 = ax3.imshow(attention_weights.squeeze().cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
    ax3.set_title('Attention Weights')
    ax3.set_xlabel('X (BEV)')
    ax3.set_ylabel('Y (BEV)')
    plt.colorbar(im3, ax=ax3, label='Attention', fraction=0.046)

    # Add statistics
    att_text = f"Mean: {attention_weights.mean():.3f}\nMax: {attention_weights.max():.3f}\nMin: {attention_weights.min():.3f}"
    ax3.text(0.02, 0.98, att_text, transform=ax3.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=8)

    plt.suptitle(f'Sample {sample_idx} - Risk-Guided Attention Visualization', fontsize=14, y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_bev_with_detections(boxes, scores, labels, class_names, output_path,
                              risk_map=None, title='BEV Detection Results'):
    """
    Plot BEV with detection boxes

    Args:
        boxes: Detection boxes [N, 9] (x, y, z, w, l, h, yaw, vx, vy)
        scores: Detection scores [N]
        labels: Detection labels [N]
        class_names: List of class names
        output_path: Output file path
        risk_map: Optional risk map to overlay
        title: Figure title
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot risk map as background if available
    if risk_map is not None:
        im = ax.imshow(risk_map.squeeze().cpu().numpy(),
                       cmap='hot', alpha=0.3, extent=[-51.2, 51.2, -51.2, 51.2],
                       vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Risk Score', fraction=0.046)

    # Set BEV range
    ax.set_xlim(-51.2, 51.2)
    ax.set_ylim(-51.2, 51.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)

    # Draw ego vehicle
    ego_size = 4.0
    ego_rect = patches.Rectangle((-ego_size/2, -ego_size/2), ego_size, ego_size,
                                  linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3)
    ax.add_patch(ego_rect)
    ax.text(0, 0, 'Ego', ha='center', va='center', fontsize=10, color='white', weight='bold')

    # Color map for different classes
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Plot detections
    boxes_np = boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes
    scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else scores
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

    for box, score, label in zip(boxes_np, scores_np, labels_np):
        if score < 0.3:  # Score threshold
            continue

        x, y, z, w, l, h, yaw = box[:7]

        # Draw bounding box
        color = colors[int(label) % len(colors)]

        # Rotated rectangle
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # Box corners (front-left, front-right, rear-right, rear-left)
        corners = np.array([
            [l/2, w/2],
            [l/2, -w/2],
            [-l/2, -w/2],
            [-l/2, w/2],
            [l/2, w/2]  # Close the box
        ])

        # Rotate and translate
        rot_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corners_rot = corners @ rot_matrix.T
        corners_rot[:, 0] += x
        corners_rot[:, 1] += y

        ax.plot(corners_rot[:, 0], corners_rot[:, 1], color=color, linewidth=2)
        ax.fill(corners_rot[:, 0], corners_rot[:, 1], color=color, alpha=0.2)

        # Add label and score
        ax.text(x, y, f'{class_names[int(label)]}\n{score:.2f}',
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compare_with_without_attention(model, data, device, output_path):
    """
    Compare detection results with and without risk-guided attention

    This is for visualization purposes only - shows the difference
    """
    # This would require running inference twice with different configs
    # For now, we'll just visualize the attention-guided results
    pass


def main():
    args = parse_args()

    # Setup
    setup_matplotlib()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("PAPER VISUALIZATION GENERATION")
    print("="*80)

    # Load config and build model
    print(f"\nLoading config from {args.config}")
    cfg = Config.fromfile(args.config)

    print("Building dataset...")
    # Remove samples_per_gpu from val config as it's not needed for dataset building
    val_cfg = cfg.data.val.copy()
    if hasattr(val_cfg, 'samples_per_gpu'):
        delattr(val_cfg, 'samples_per_gpu')
    dataset = build_dataset(val_cfg)

    print("Building model...")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = model.to(args.device)
    model.eval()

    print(f"\nGenerating visualizations for {args.num_samples} samples...")
    print(f"Output directory: {args.output_dir}")

    # Generate visualizations
    for idx in tqdm(range(min(args.num_samples, len(dataset)))):
        data = dataset[idx]

        # Get ground truth risk map if available
        gt_risk_map = None
        if 'gt_risk_map' in data:
            gt_risk_map = data['gt_risk_map'].data if hasattr(data['gt_risk_map'], 'data') else data['gt_risk_map']

        # Prepare input - handle both DataContainer and direct tensor
        if isinstance(data['img'], list):
            # MultiScaleFlipAug format
            img_item = data['img'][0]
            img_tensor = img_item.data if hasattr(img_item, 'data') else img_item
            img = img_tensor.unsqueeze(0).to(args.device)

            meta_item = data['img_metas'][0]
            img_metas = [meta_item.data if hasattr(meta_item, 'data') else meta_item]
        else:
            img_data = data['img'].data if hasattr(data['img'], 'data') else data['img']
            img = img_data.unsqueeze(0).to(args.device)
            img_metas = [data['img_metas'].data if hasattr(data['img_metas'], 'data') else data['img_metas']]

        # Ensure img_metas is list
        if not isinstance(img_metas, list):
            img_metas = [img_metas]

        # Inference
        with torch.no_grad():
            prev_bev = None
            new_prev_bev, results = model.simple_test(
                img_metas=img_metas,
                img=img,
                prev_bev=prev_bev,
                rescale=True
            )

        result = results[0]
        pts_bbox = result['pts_bbox']

        # Check if risk map and attention are available
        if 'risk_map' not in result:
            print(f"\nWarning: Sample {idx} does not have risk map in results")
            continue

        risk_map = result['risk_map']

        # Get attention weights if available
        attention_weights = result.get('attention_weights', risk_map)  # Fallback to risk map

        # Filter high-risk samples if requested
        if args.high_risk_only and risk_map.max() < 0.5:
            continue

        # 1. Comprehensive visualization
        output_path = os.path.join(args.output_dir, f'sample_{idx:03d}_comprehensive.png')
        # Get original image data
        if isinstance(data['img'], list):
            img_item = data['img'][0]
            img_for_viz = img_item.data if hasattr(img_item, 'data') else img_item
        else:
            img_for_viz = data['img'].data if hasattr(data['img'], 'data') else data['img']

        plot_risk_attention_detection(
            img_for_viz,
            risk_map,
            attention_weights,
            pts_bbox['boxes_3d'].tensor,
            pts_bbox['scores_3d'],
            pts_bbox['labels_3d'],
            dataset.CLASSES,
            idx,
            output_path,
            gt_risk_map
        )

        # 2. BEV with detections
        output_path = os.path.join(args.output_dir, f'sample_{idx:03d}_bev_detections.png')
        plot_bev_with_detections(
            pts_bbox['boxes_3d'].tensor,
            pts_bbox['scores_3d'],
            pts_bbox['labels_3d'],
            dataset.CLASSES,
            output_path,
            risk_map=risk_map,
            title=f'Sample {idx} - BEV Detection with Risk Map'
        )

    print(f"\n✅ Visualizations saved to {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
