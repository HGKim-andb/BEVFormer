#!/usr/bin/env python3
"""
Generate Figure 2 for Paper (Revised)

Reviewer requested changes:
1. Add legend - explain green/yellow circles
2. Make risk map larger (reduce raw image size)
3. Add colorbar with 0-1 scale

Output: visualizations/figure2_revised.png (300dpi)
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import sys

try:
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion
except ImportError:
    print("Error: nuscenes-devkit not installed")
    sys.exit(1)


def create_risk_colormap():
    """Black to red colormap for risk visualization"""
    colors = [
        (0.0, 0.0, 0.0),      # Black (0.0)
        (0.3, 0.0, 0.0),      # Very dark red
        (0.6, 0.0, 0.0),      # Dark red
        (0.9, 0.0, 0.0),      # Red
        (1.0, 0.2, 0.0),      # Bright red-orange (1.0)
    ]
    return LinearSegmentedColormap.from_list('black_red', colors, N=256)


RISK_CMAP = create_risk_colormap()


def setup_matplotlib():
    """Setup matplotlib for publication-quality figures"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14


def generate_figure2(nusc, label_data, save_path, dataroot):
    """
    Generate Figure 2 with reviewer requested changes:
    1. Legend for green/yellow circles
    2. Larger risk map (reduced camera images)
    3. Colorbar with 0-1 scale

    Layout (revised):
        Row 0: [FRONT_LEFT] [FRONT] [FRONT_RIGHT] (smaller)
        Row 1: [BACK_LEFT]  [RISK MAP - LARGE]  [BACK_RIGHT]
        Row 2:              [BACK]

    Risk map takes center 2 columns for larger display
    """
    sample_token = label_data['sample_token']
    sample = nusc.get('sample', sample_token)
    risk_map = label_data['risk_map']

    # Create figure - adjusted for larger risk map
    fig = plt.figure(figsize=(18, 14))

    # Use GridSpec for flexible layout
    # 6 columns: [BACK_LEFT] [gap] [RISK MAP] [colorbar] [gap] [BACK_RIGHT]
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1, 2.5, 0.8],
                  width_ratios=[1, 0.1, 2.5, 0.06, 0.25, 1], wspace=0.02, hspace=0.25)

    # Camera positions - smaller images
    camera_layout = {
        'CAM_FRONT_LEFT': gs[0, 0],
        'CAM_FRONT': gs[0, 2],  # Center column (risk map column)
        'CAM_FRONT_RIGHT': gs[0, 5],
        'CAM_BACK_LEFT': gs[1, 0],
        'CAM_BACK_RIGHT': gs[1, 5],  # Rightmost column
        'CAM_BACK': gs[2, 2],  # Center column
    }

    # 1. Display camera images (smaller)
    for cam_name, grid_spec in camera_layout.items():
        ax_cam = fig.add_subplot(grid_spec)

        if cam_name in sample['data']:
            cam_token = sample['data'][cam_name]
            cam_data = nusc.get('sample_data', cam_token)
            img_path = Path(dataroot) / cam_data['filename']

            if img_path.exists():
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Smaller size for side cameras
                if cam_name in ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                               'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                    img_resized = cv2.resize(img, (320, 180))
                else:
                    img_resized = cv2.resize(img, (480, 270))
                ax_cam.imshow(img_resized)

        cam_label = cam_name.replace('CAM_', '').replace('_', ' ')
        ax_cam.set_title(cam_label, fontsize=10, fontweight='bold')
        ax_cam.axis('off')

    # 2. RISK MAP - LARGER (center column)
    ax_risk = fig.add_subplot(gs[1, 2])

    # Rotate risk map for proper orientation
    risk_map_display = np.rot90(risk_map, k=-1)

    im = ax_risk.imshow(risk_map_display, cmap=RISK_CMAP, vmin=0, vmax=1,
                        extent=[-50, 50, -50, 50], origin='lower', aspect='equal')

    ax_risk.set_title('Emergence Risk Map', fontsize=13, fontweight='bold', pad=10)
    ax_risk.set_xlabel('Lateral (m)', fontsize=11)
    ax_risk.set_ylabel('Longitudinal (m)', fontsize=11)
    ax_risk.grid(True, alpha=0.2, color='white', linewidth=0.5)

    # Draw ego vehicle
    ego_width = 2.0
    ego_length = 4.5
    ego_vehicle = FancyBboxPatch(
        (-ego_width/2, -ego_length/2), ego_width, ego_length,
        boxstyle="round,pad=0.1", linewidth=2,
        edgecolor='cyan', facecolor='blue', alpha=0.7, zorder=10
    )
    ax_risk.add_patch(ego_vehicle)

    # Ego heading arrow
    ax_risk.arrow(0, ego_length/2, 0, 3.0, head_width=1.5, head_length=1.0,
                  fc='cyan', ec='white', alpha=0.9, zorder=11, linewidth=2)

    # Draw detected objects on risk map
    ego_pose = nusc.get('ego_pose', sample['data']['LIDAR_TOP'])
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])

    vehicle_patches = []
    pedestrian_patches = []

    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)

        # Transform to ego frame
        obj_translation_world = np.array(ann['translation'])
        obj_translation_ego = obj_translation_world - ego_translation
        obj_translation_ego = ego_rotation.inverse.rotate(obj_translation_ego)

        x, y = obj_translation_ego[:2]
        # Counter-clockwise 90-degree rotation
        x_rot, y_rot = -y, x

        if -50 <= x_rot <= 50 and -50 <= y_rot <= 50:
            category = ann['category_name']

            if 'vehicle' in category:
                # Green circle for vehicles
                circle = Circle((x_rot, y_rot), 2.0,
                               facecolor='limegreen', edgecolor='white',
                               linewidth=1.5, alpha=0.8, zorder=8)
                ax_risk.add_patch(circle)
                vehicle_patches.append(circle)
            elif 'pedestrian' in category:
                # Yellow/orange circle for pedestrians
                circle = Circle((x_rot, y_rot), 1.2,
                               facecolor='gold', edgecolor='white',
                               linewidth=1.5, alpha=0.8, zorder=8)
                ax_risk.add_patch(circle)
                pedestrian_patches.append(circle)
            else:
                # Gray for other objects
                circle = Circle((x_rot, y_rot), 1.5,
                               facecolor='lightgray', edgecolor='white',
                               linewidth=1.0, alpha=0.6, zorder=7)
                ax_risk.add_patch(circle)

    # 3. COLORBAR (RIGHT side of risk map - separate axes)
    cbar_ax = fig.add_subplot(gs[1, 3])
    cbar = plt.colorbar(im, cax=cbar_ax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.set_yticklabels(['0.0\n(Safe)', '0.25', '0.5', '0.75', '1.0\n(High)'])
    cbar.ax.set_title('Risk\nScore', fontsize=10, fontweight='bold', pad=3)

    # 4. LEGEND (below risk map or in corner)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen',
               markersize=12, markeredgecolor='white', markeredgewidth=1.5,
               label='Vehicle'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
               markersize=10, markeredgecolor='white', markeredgewidth=1.5,
               label='Pedestrian'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
               markersize=10, markeredgecolor='white', markeredgewidth=1.0,
               label='Other'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',
               markersize=10, markeredgecolor='cyan', markeredgewidth=2,
               label='Ego Vehicle'),
    ]

    ax_risk.legend(handles=legend_elements, loc='upper left',
                   framealpha=0.95, edgecolor='black', fontsize=10,
                   title='Detected Objects', title_fontsize=11)

    # 5. Statistics box (bottom right of risk map)
    stats_text = (
        f"Max Risk: {label_data['metadata']['max_risk']:.2f}\n"
        f"Mean Risk: {label_data['metadata']['mean_risk']:.3f}\n"
        f"High-risk cells: {label_data['metadata']['high_risk_cells']}"
    )
    ax_risk.text(0.98, 0.02, stats_text, transform=ax_risk.transAxes,
                 fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                          edgecolor='gray', linewidth=1))

    # Add range circles
    for r in [10, 20, 30, 40]:
        circle = Circle((0, 0), r, fill=False, edgecolor='white',
                        linestyle=':', linewidth=0.8, alpha=0.4, zorder=1)
        ax_risk.add_patch(circle)

    # Save with high DPI
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {save_path}")


def load_labels(pkl_path):
    """Load labels from pickle file"""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def get_high_risk_sample(labels_dict, min_risk=0.6):
    """Get a representative high-risk sample for Figure 2"""
    best_sample = None
    best_score = 0

    for scene_token, scene_labels in labels_dict.items():
        for label in scene_labels:
            max_risk = label['metadata']['max_risk']
            high_cells = label['metadata']['high_risk_cells']

            # Score: prefer samples with good risk distribution
            score = max_risk * 0.5 + min(high_cells / 100, 1.0) * 0.5

            if max_risk >= min_risk and score > best_score:
                best_score = score
                best_sample = label

    return best_sample


def main():
    parser = argparse.ArgumentParser(description='Generate Figure 2 (Revised)')
    parser.add_argument('--labels', type=str,
                        default='data/emergence_risk_v5/risk_labels_train.pkl',
                        help='Path to labels pickle file')
    parser.add_argument('--dataroot', type=str,
                        default='/home/hg-main/data/nuscenes',
                        help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='nuScenes version')
    parser.add_argument('--output', type=str,
                        default='visualizations/figure2_revised.png',
                        help='Output path')
    parser.add_argument('--sample_token', type=str, default=None,
                        help='Specific sample token to visualize')
    parser.add_argument('--min_risk', type=float, default=0.6,
                        help='Minimum risk for auto-selection')

    args = parser.parse_args()

    # Setup
    setup_matplotlib()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("GENERATING FIGURE 2 (REVISED)")
    print("="*60)
    print("Reviewer changes:")
    print("  1. Legend added for green/yellow circles")
    print("  2. Risk map enlarged")
    print("  3. Colorbar with 0-1 scale added")
    print("="*60)

    # Load nuScenes
    print(f"\nLoading nuScenes from {args.dataroot}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # Load labels
    print(f"Loading labels from {args.labels}...")
    labels_dict = load_labels(args.labels)

    # Get sample
    if args.sample_token:
        # Find specific sample
        sample_label = None
        for scene_labels in labels_dict.values():
            for label in scene_labels:
                if label['sample_token'] == args.sample_token:
                    sample_label = label
                    break
        if sample_label is None:
            print(f"Error: Sample {args.sample_token} not found")
            return
    else:
        # Auto-select high-risk sample
        print(f"Auto-selecting high-risk sample (min_risk={args.min_risk})...")
        sample_label = get_high_risk_sample(labels_dict, args.min_risk)
        if sample_label is None:
            print("Error: No suitable sample found")
            return

    print(f"\nSelected sample: {sample_label['sample_token'][:16]}...")
    print(f"  Scene: {sample_label.get('scene_name', 'N/A')}")
    print(f"  Max risk: {sample_label['metadata']['max_risk']:.3f}")
    print(f"  High-risk cells: {sample_label['metadata']['high_risk_cells']}")

    # Generate figure
    print(f"\nGenerating figure...")
    generate_figure2(nusc, sample_label, args.output, args.dataroot)

    print("\n" + "="*60)
    print(f"OUTPUT: {args.output}")
    print(f"DPI: 300")
    print("="*60)


if __name__ == '__main__':
    main()
