#!/usr/bin/env python3
"""
Risk-Guided Attention Model Inference Script

Usage:
    python inference_risk_attention.py \
        --config projects/configs/bevformer/bevformer_risk_tiny_attention.py \
        --checkpoint work_dirs/bevformer_risk_attention_v2/epoch_6.pth \
        --sample-idx 0
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mmcv import Config
from mmcv.runner import load_checkpoint
import os
import sys
import cv2

# Add projects to path before importing mmdet3d
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'projects'))

# Now import - this will register custom modules
import mmdet3d_plugin  # noqa: F401
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with Risk-Guided Attention')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file path')
    parser.add_argument('--sample-idx', type=int, default=0, help='Sample index to visualize')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    parser.add_argument('--output-dir', default='inference_outputs', help='Output directory')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for saved figures')
    return parser.parse_args()


def denormalize_img(img_tensor):
    """Denormalize image tensor to [0, 1] range for display"""
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img_np = (img_np * std + mean) / 255.0
    return np.clip(img_np, 0, 1)


def plot_comprehensive_visualization(img, risk_map, attention_weights, boxes, scores, labels,
                                     class_names, sample_idx, output_path, gt_risk_map=None):
    """
    Create comprehensive visualization with cameras around BEV layout.

    Layout:
        [FRONT_LEFT]  [FRONT]     [FRONT_RIGHT]
        [BACK_LEFT]    [BEV]      [BACK_RIGHT]
                      [BACK]
        [GT RISK (if available)]  [PRED RISK]  [ATTENTION]
    """
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

    # 3. Bottom row: GT Risk (if available), Predicted Risk, Attention
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
    fig.suptitle(f'Sample {sample_idx} - Risk-Guided Attention Inference Results',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Comprehensive visualization saved to: {output_path}")


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("RISK-GUIDED ATTENTION INFERENCE")
    print("="*80)

    print(f"\nLoading config from {args.config}")
    cfg = Config.fromfile(args.config)

    # Build dataset
    print("Building dataset...")
    val_cfg = cfg.data.val.copy()
    if hasattr(val_cfg, 'samples_per_gpu'):
        delattr(val_cfg, 'samples_per_gpu')
    dataset = build_dataset(val_cfg)

    # Build model
    print("Building model...")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # Move to device
    model = model.to(args.device)
    model.eval()

    # Get sample
    print(f"\nLoading sample {args.sample_idx}")
    data = dataset[args.sample_idx]

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

    print("\nInput shapes:")
    print(f"  Images: {img.shape}")
    print(f"  Scene: {img_metas[0].get('scene_token', 'N/A')}")
    print(f"  Sample: {img_metas[0].get('sample_idx', args.sample_idx)}")

    # Inference
    print("\nRunning inference...")
    with torch.no_grad():
        prev_bev = None
        new_prev_bev, results = model.simple_test(
            img_metas=img_metas,
            img=img,
            prev_bev=prev_bev,
            rescale=True
        )

    # Extract results
    result = results[0]
    pts_bbox = result['pts_bbox']

    print("\nDetection Results:")
    print(f"  Number of detections: {len(pts_bbox['boxes_3d'])}")
    print(f"  Score range: [{pts_bbox['scores_3d'].min():.3f}, {pts_bbox['scores_3d'].max():.3f}]")

    # Risk map and attention
    risk_map = result.get('risk_map', None)
    attention_weights = result.get('attention_weights', risk_map)

    if risk_map is not None:
        print(f"\nRisk Map:")
        print(f"  Shape: {risk_map.shape}")
        print(f"  Range: [{risk_map.min():.4f}, {risk_map.max():.4f}]")
        print(f"  Mean: {risk_map.mean():.4f}")

    if attention_weights is not None:
        print(f"\nAttention Weights:")
        print(f"  Shape: {attention_weights.shape}")
        print(f"  Range: [{attention_weights.min():.4f}, {attention_weights.max():.4f}]")
        print(f"  Mean: {attention_weights.mean():.4f}")

    # Generate visualization
    print("\nGenerating visualization...")
    output_path = os.path.join(args.output_dir, f'sample_{args.sample_idx}_comprehensive.png')

    # Get original image data
    if isinstance(data['img'], list):
        img_item = data['img'][0]
        img_for_viz = img_item.data if hasattr(img_item, 'data') else img_item
    else:
        img_for_viz = data['img'].data if hasattr(data['img'], 'data') else data['img']

    plot_comprehensive_visualization(
        img_for_viz,
        risk_map if risk_map is not None else torch.zeros(1, 200, 200),
        attention_weights if attention_weights is not None else torch.zeros(1, 50, 50),
        pts_bbox['boxes_3d'].tensor,
        pts_bbox['scores_3d'],
        pts_bbox['labels_3d'],
        dataset.CLASSES,
        args.sample_idx,
        output_path,
        gt_risk_map=gt_risk_map
    )

    # Save detection results to text file
    output_file = os.path.join(args.output_dir, f'sample_{args.sample_idx}_detections.txt')
    with open(output_file, 'w') as f:
        f.write(f"Sample Index: {args.sample_idx}\n")
        f.write(f"Scene Token: {img_metas[0].get('scene_token', 'N/A')}\n\n")

        f.write("Detections:\n")
        boxes = pts_bbox['boxes_3d'].tensor.cpu().numpy()
        scores = pts_bbox['scores_3d'].cpu().numpy()
        labels = pts_bbox['labels_3d'].cpu().numpy()

        class_names = dataset.CLASSES
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if score < 0.3:
                continue
            f.write(f"  {i+1}. {class_names[label]}: score={score:.3f}, ")
            f.write(f"center=({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}), ")
            f.write(f"size=({box[3]:.2f}, {box[4]:.2f}, {box[5]:.2f})\n")

    print(f"  Detection results saved to: {output_file}")
    print("\n" + "="*80)
    print("Inference completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
