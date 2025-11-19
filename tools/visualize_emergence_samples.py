#!/usr/bin/env python3
"""
Emergence Sample Visualization Script

This script creates visualizations of emergence labels overlaid on camera images
and BEV representations. It helps verify that the labels are correctly generated.
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
from tqdm import tqdm
import sys

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion
except ImportError:
    print("Error: nuscenes-devkit not installed. Install with: pip install nuscenes-devkit")
    sys.exit(1)


def load_labels(pkl_path):
    """Load labels from pickle file"""
    print(f"Loading labels from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        labels = pickle.load(f)
    return labels


def get_positive_samples(labels_dict, min_emergences=1):
    """
    Extract samples that have emergences

    Args:
        labels_dict: Dict of scene_token -> list of labels
        min_emergences: Minimum number of emergences required

    Returns:
        List of label dicts with emergences
    """
    positive_samples = []

    for scene_token, scene_labels in labels_dict.items():
        for label in scene_labels:
            if label['num_emergences'] >= min_emergences:
                positive_samples.append(label)

    return positive_samples


def visualize_sample(nusc, label_data, save_path, dataroot):
    """
    Create visualization for one sample with 6 cameras in BEV layout

    Args:
        nusc: NuScenes instance
        label_data: Label dict for this sample
        save_path: Path to save the visualization
        dataroot: Path to nuScenes data
    """
    sample_token = label_data['sample_token']
    sample = nusc.get('sample', sample_token)

    # Create figure with more rows for 6 cameras
    fig = plt.figure(figsize=(24, 18))

    # Camera layout mapping (BEV perspective - as if looking from above)
    # Layout (6 cameras + BEV):
    #   Row 0:  FRONT_LEFT    FRONT        FRONT_RIGHT
    #   Row 1:  BACK_LEFT     BACK         BACK_RIGHT
    #   Row 2:  ----------    BEV     ----------
    camera_layout = {
        'CAM_FRONT_LEFT': (0, 0),
        'CAM_FRONT': (0, 1),
        'CAM_FRONT_RIGHT': (0, 2),
        'CAM_BACK_LEFT': (1, 0),
        'CAM_BACK': (1, 1),
        'CAM_BACK_RIGHT': (1, 2),
    }

    # Grid: 6 rows x 3 columns
    # Row 0-1: 6 cameras in BEV-style layout
    # Row 2: BEV with detections (full width)
    # Row 3-5: Emergence heatmaps t+1, t+2, t+3

    # 1. Display all 6 cameras
    for cam_name, (row, col) in camera_layout.items():
        ax_cam = plt.subplot2grid((6, 3), (row, col))

        cam_token = sample['data'][cam_name]
        cam_data = nusc.get('sample_data', cam_token)
        img_path = Path(dataroot) / cam_data['filename']

        if img_path.exists():
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax_cam.imshow(img)
        else:
            ax_cam.text(0.5, 0.5, f'{cam_name}\nNot found',
                       ha='center', va='center', transform=ax_cam.transAxes,
                       fontsize=10)

        # Camera name as title
        cam_display_name = cam_name.replace('CAM_', '').replace('_', ' ')
        ax_cam.set_title(cam_display_name, fontsize=10, fontweight='bold')
        ax_cam.axis('off')

    # Add main title with sample info
    fig.text(0.5, 0.98, f'Sample: {sample_token[:16]}...  |  Scene: {label_data["scene_name"]}  |  Emergences: {label_data["num_emergences"]}',
             ha='center', fontsize=14, fontweight='bold', va='top')

    # 2. Current BEV with detections (middle center position - where BACK camera would be)
    ax_bev = plt.subplot2grid((6, 3), (2, 0), colspan=3)
    ax_bev.set_xlim(-50, 50)
    ax_bev.set_ylim(-50, 50)
    ax_bev.set_aspect('equal')
    ax_bev.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_bev.set_xlabel('X (meters)', fontsize=11)
    ax_bev.set_ylabel('Y (meters)', fontsize=11)
    ax_bev.set_title('Bird\'s Eye View - Current Frame with Future Emergences',
                    fontsize=13, fontweight='bold')

    # Draw ego vehicle (blue rectangle at origin)
    ego_rect = Rectangle((-2, -1), 4, 2, linewidth=2,
                         edgecolor='blue', facecolor='lightblue', alpha=0.6, zorder=10)
    ax_bev.add_patch(ego_rect)
    ax_bev.text(0, 0, 'EGO', ha='center', va='center', fontsize=9,
               fontweight='bold', color='darkblue', zorder=11)

    # Draw current detections (green circles)
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        x, y, z = ann['translation']

        # Only show objects within BEV range
        if -50 <= x <= 50 and -50 <= y <= 50:
            circle = Circle((x, y), 1.0, color='green', alpha=0.4, zorder=5)
            ax_bev.add_patch(circle)
            ax_bev.plot(x, y, 'go', markersize=6, zorder=6)

    # Draw future emergences with different colors per frame
    frame_colors = {1: '#FF1744', 2: '#FF9800', 3: '#FFC107'}  # Red, Orange, Yellow
    frame_markers = {1: '*', 2: 'D', 3: 's'}  # Star, Diamond, Square
    frame_labels = {1: [], 2: [], 3: []}

    for info in label_data['emergence_info']:
        x, y = info['position']
        frame = info['frame']
        color = frame_colors[frame]
        marker = frame_markers[frame]

        # Draw marker
        ax_bev.plot(x, y, marker, color=color, markersize=15,
                   markeredgecolor='white', markeredgewidth=1.5,
                   label=f't+{frame}' if frame not in frame_labels[frame] else '',
                   zorder=7)

        # Add text label with category
        cat = info['category'].split('.')[-1]  # Get last part of category
        ax_bev.text(x, y - 2.5, f"{cat}\nt+{frame}",
                   fontsize=7, ha='center', color=color,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color),
                   zorder=8)

        frame_labels[frame].append(True)

    # Add legend
    ax_bev.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # Add range circles
    for r in [10, 20, 30, 40]:
        circle = Circle((0, 0), r, fill=False, edgecolor='gray',
                       linestyle=':', linewidth=1, alpha=0.4, zorder=1)
        ax_bev.add_patch(circle)
        ax_bev.text(r, 0, f'{r}m', fontsize=8, color='gray', alpha=0.6)

    # 3-5. Emergence heatmaps (rows 3-5, 3 columns)
    for t in range(3):
        # Each heatmap takes 1 column
        ax = plt.subplot2grid((6, 3), (3, t), rowspan=3)

        mask = label_data['emergence_mask'][t]

        # Create heatmap
        im = ax.imshow(mask, cmap='hot', vmin=0, vmax=1,
                      extent=[-50, 50, -50, 50], origin='lower', aspect='equal')

        ax.set_title(f'Emergence Heatmap t+{t+1}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (meters)', fontsize=10)
        ax.set_ylabel('Y (meters)', fontsize=10)
        ax.grid(True, alpha=0.2, color='white', linewidth=0.5)

        # Mark ego position
        ax.plot(0, 0, 'b*', markersize=15, markeredgecolor='white', markeredgewidth=1.5)

        # Mark actual emergence locations for this frame
        for info in label_data['emergence_info']:
            if info['frame'] == t + 1:
                x, y = info['position']
                # Draw green star at actual location
                ax.plot(x, y, 'g*', markersize=12, markeredgecolor='white',
                       markeredgewidth=1, zorder=10)

                # Draw white circle around it
                circle = Circle((x, y), 2, fill=False, edgecolor='lime',
                              linewidth=2, zorder=9)
                ax.add_patch(circle)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Probability', fontsize=9)

        # Add count of emergences for this frame
        count = sum(1 for info in label_data['emergence_info'] if info['frame'] == t + 1)
        ax.text(0.02, 0.98, f'{count} emergence(s)',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Adjust layout and save (more space for top title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize emergence label samples')
    parser.add_argument('--labels', type=str,
                        default='data/emergence_labels/emergence_labels_train.pkl',
                        help='Path to labels pickle file')
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='nuScenes version')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='visualizations/emergence_samples',
                        help='Output directory for visualizations')
    parser.add_argument('--min_emergences', type=int, default=1,
                        help='Minimum number of emergences per sample')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sample selection')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("EMERGENCE SAMPLE VISUALIZATION")
    print("="*80)
    print(f"Labels:      {args.labels}")
    print(f"Dataroot:    {args.dataroot}")
    print(f"Output dir:  {output_dir}")
    print(f"Num samples: {args.num_samples}")
    print("="*80 + "\n")

    # Load nuScenes
    print("Loading nuScenes dataset...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # Load labels
    labels_dict = load_labels(args.labels)

    # Get positive samples
    print(f"Finding samples with >= {args.min_emergences} emergence(s)...")
    positive_samples = get_positive_samples(labels_dict, args.min_emergences)
    print(f"Found {len(positive_samples)} positive samples")

    if len(positive_samples) == 0:
        print("❌ No positive samples found!")
        return

    # Select random samples
    num_to_viz = min(args.num_samples, len(positive_samples))
    selected_samples = random.sample(positive_samples, num_to_viz)

    print(f"\nGenerating visualizations for {num_to_viz} samples...")

    # Generate visualizations
    for i, label_data in enumerate(tqdm(selected_samples, desc="Visualizing")):
        sample_token = label_data['sample_token']
        save_path = output_dir / f"sample_{i:03d}_{sample_token[:8]}.png"

        try:
            visualize_sample(nusc, label_data, save_path, args.dataroot)
        except Exception as e:
            print(f"\n⚠️  Error visualizing sample {sample_token}: {e}")
            continue

    print(f"\n✅ Generated {num_to_viz} visualizations in {output_dir}")
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Files created: sample_000.png to sample_{num_to_viz-1:03d}.png")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
