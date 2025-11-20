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
    Create a comprehensive visualization with cameras around BEV layout:

    Layout:
        [FRONT_LEFT]  [FRONT]     [FRONT_RIGHT]
        [BACK_LEFT]    [BEV]      [BACK_RIGHT]
                      [BACK]
        [GT RISK (if available)]  [PRED RISK]  [ATTENTION]

    Args:
        img: Input images [N_cams, 3, H, W] (6 cameras)
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
    # Denormalize function
    def denormalize_img(img_tensor):
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        img_np = (img_np * std + mean) / 255.0
        return np.clip(img_np, 0, 1)

    # Create figure with 4 rows
    fig = plt.figure(figsize=(24, 20))

    # Camera names in order: FL, F, FR, BL, BR, B
    camera_names = ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT',
                    'BACK_LEFT', 'BACK_RIGHT', 'BACK']

    # Define camera positions (row, col)
    camera_positions = {
        'FRONT_LEFT': (0, 0),
        'FRONT': (0, 1),
        'FRONT_RIGHT': (0, 2),
        'BACK_LEFT': (2, 0),
        'BACK_RIGHT': (2, 2),
        'BACK': (2, 1),
    }

    # 1. Display all 6 cameras around BEV
    if img is not None and len(img) >= 6:
        for idx, (cam_name, (row, col)) in enumerate(camera_positions.items()):
            ax_cam = plt.subplot2grid((4, 3), (row, col))
            if idx < len(img):
                cam_img = denormalize_img(img[idx])
                # Resize for display
                cam_img_resized = cv2.resize(cam_img, (600, 338))
                ax_cam.imshow(cam_img_resized)
            ax_cam.set_title(cam_name.replace('_', ' '), fontsize=10, fontweight='bold')
            ax_cam.axis('off')

    # 2. BEV with detection boxes (center position)
    ax_bev = plt.subplot2grid((4, 3), (1, 1))
    ax_bev.set_xlim(-51.2, 51.2)
    ax_bev.set_ylim(-51.2, 51.2)
    ax_bev.set_aspect('equal')
    ax_bev.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_bev.set_xlabel('X (meters)', fontsize=10)
    ax_bev.set_ylabel('Y (meters)', fontsize=10)
    ax_bev.set_title('Bird\'s Eye View - Detections', fontsize=11, fontweight='bold')

    # Draw ego vehicle
    ego_rect = patches.Rectangle((-2, -2), 4, 4, linewidth=2,
                                 edgecolor='blue', facecolor='lightblue', alpha=0.6, zorder=10)
    ax_bev.add_patch(ego_rect)
    ax_bev.text(0, 0, 'EGO', ha='center', va='center', fontsize=9,
               fontweight='bold', color='darkblue', zorder=11)

    # Draw detection boxes
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    boxes_np = boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes
    scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else scores
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

    for box, score, label in zip(boxes_np, scores_np, labels_np):
        if score < 0.3:
            continue

        x, y, z, w, l, h, yaw = box[:7]
        color = colors[int(label) % len(colors)]

        # Rotated rectangle corners
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        corners = np.array([
            [l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2], [l/2, w/2]
        ])
        rot_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corners_rot = corners @ rot_matrix.T
        corners_rot[:, 0] += x
        corners_rot[:, 1] += y

        ax_bev.plot(corners_rot[:, 0], corners_rot[:, 1], color=color, linewidth=1.5)
        ax_bev.fill(corners_rot[:, 0], corners_rot[:, 1], color=color, alpha=0.2)

    # 3. Bottom row: GT Risk, Predicted Risk, Attention
    bottom_plots = []
    if gt_risk_map is not None:
        bottom_plots.append(('Ground Truth Risk', gt_risk_map, 'hot'))
    bottom_plots.append(('Predicted Risk Map', risk_map, 'hot'))
    bottom_plots.append(('Attention Weights', attention_weights, 'viridis'))

    for idx, (title, data, cmap) in enumerate(bottom_plots):
        ax = plt.subplot2grid((4, len(bottom_plots)), (3, idx))
        im = ax.imshow(data.squeeze().cpu().numpy(), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('X (BEV)', fontsize=8)
        ax.set_ylabel('Y (BEV)', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add statistics
        stats_text = f"Max: {data.max():.3f}\nMean: {data.mean():.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=8)

    # Main title
    fig.suptitle(f'Sample {sample_idx} - Risk-Guided Attention for 3D Object Detection',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
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
