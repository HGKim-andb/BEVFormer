#!/usr/bin/env python3
"""
Risk Sample Visualization Script

This script creates visualizations of risk maps overlaid on camera images
and BEV representations. It helps verify that the risk maps are correctly generated.
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
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import sys

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion
except ImportError:
    print("Error: nuscenes-devkit not installed. Install with: pip install nuscenes-devkit")
    sys.exit(1)


# Create custom colormap: Black -> Dark Red -> Red -> Bright Red
def create_risk_colormap():
    """
    Create a black to red colormap for risk visualization

    0.0: Black (no risk)
    0.3: Dark red (low risk)
    0.5: Red (medium risk)
    0.7: Bright red (high risk)
    1.0: Very bright red (extreme risk)
    """
    colors = [
        (0.0, 0.0, 0.0),      # Black (0.0)
        (0.3, 0.0, 0.0),      # Very dark red (0.25)
        (0.6, 0.0, 0.0),      # Dark red (0.5)
        (0.9, 0.0, 0.0),      # Red (0.75)
        (1.0, 0.2, 0.0),      # Bright red-orange (1.0)
    ]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('black_red', colors, N=n_bins)
    return cmap


# Global colormap
RISK_CMAP = create_risk_colormap()


def load_labels(pkl_path):
    """Load labels from pickle file"""
    print(f"Loading labels from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        labels = pickle.load(f)
    return labels


def get_high_risk_samples(labels_dict, min_max_risk=0.5):
    """
    Extract samples with high risk

    Args:
        labels_dict: Dict of scene_token -> list of labels
        min_max_risk: Minimum max risk required

    Returns:
        List of label dicts with high risk
    """
    high_risk_samples = []

    for scene_token, scene_labels in labels_dict.items():
        for label in scene_labels:
            if label['metadata']['max_risk'] >= min_max_risk:
                high_risk_samples.append(label)

    return high_risk_samples


def visualize_sample(nusc, label_data, save_path, dataroot):
    """
    Create visualization for one sample with 6 cameras around BEV space

    Args:
        nusc: NuScenes instance
        label_data: Label dict for this sample
        save_path: Path to save the visualization
        dataroot: Path to nuScenes data
    """
    sample_token = label_data['sample_token']
    sample = nusc.get('sample', sample_token)
    risk_map = label_data['risk_map']

    # Create figure with modern BEV layout: cameras around the BEV space
    fig = plt.figure(figsize=(24, 18))

    # Layout:
    #   [FRONT_LEFT]  [FRONT]  [FRONT_RIGHT]
    #   [BACK_LEFT]    [BEV]   [BACK_RIGHT]
    #                 [BACK]
    # Then full-width risk map at bottom

    # Add main title with sample info
    fig.text(0.5, 0.98,
             f'Sample: {sample_token[:16]}...  |  Scene: {label_data["scene_name"]}  |  '
             f'Max Risk: {label_data["metadata"]["max_risk"]:.3f}  |  '
             f'High-Risk Cells: {label_data["metadata"]["high_risk_cells"]}',
             ha='center', fontsize=16, fontweight='bold', va='top')

    # Define camera positions around BEV (using grid layout)
    # Grid: 4 rows x 3 columns for top section
    # Layout: FRONT cameras on row 0, BEV on row 1, side cameras on row 2, BACK camera also on row 2
    camera_positions = {
        'CAM_FRONT_LEFT': (0, 0),   # Top-left
        'CAM_FRONT': (0, 1),         # Top-center
        'CAM_FRONT_RIGHT': (0, 2),   # Top-right
        'CAM_BACK_LEFT': (2, 0),     # Bottom-left (moved down)
        'CAM_BACK_RIGHT': (2, 2),    # Bottom-right (moved down)
        'CAM_BACK': (2, 1),          # Bottom-center
    }

    # 1. Load and display all 6 camera images around the BEV
    for cam_name, (row, col) in camera_positions.items():
        ax_cam = plt.subplot2grid((4, 3), (row, col))

        if cam_name in sample['data']:
            cam_token = sample['data'][cam_name]
            cam_data = nusc.get('sample_data', cam_token)
            img_path = Path(dataroot) / cam_data['filename']

            if img_path.exists():
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize for consistent display
                img_resized = cv2.resize(img, (600, 338))  # 16:9 aspect ratio
                ax_cam.imshow(img_resized)
            else:
                ax_cam.text(0.5, 0.5, f'{cam_name}\nNot found',
                           ha='center', va='center', transform=ax_cam.transAxes,
                           fontsize=10, color='red')
        else:
            ax_cam.text(0.5, 0.5, f'{cam_name}\nN/A',
                       ha='center', va='center', transform=ax_cam.transAxes,
                       fontsize=10, color='gray')

        # Camera label (simplified)
        cam_label = cam_name.replace('CAM_', '').replace('_', ' ')
        ax_cam.set_title(cam_label, fontsize=10, fontweight='bold')
        ax_cam.axis('off')

    # 2. BEV with detected objects (center position)
    # Apply counter-clockwise 90-degree rotation to entire BEV space
    ax_bev = plt.subplot2grid((4, 3), (1, 1))
    ax_bev.set_xlim(-50, 50)
    ax_bev.set_ylim(-50, 50)
    ax_bev.set_aspect('equal')
    ax_bev.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_bev.set_xlabel('X (meters)', fontsize=10)
    ax_bev.set_ylabel('Y (meters)', fontsize=10)
    ax_bev.set_title('Bird\'s Eye View - Detected Objects', fontsize=11, fontweight='bold')

    # Draw ego vehicle (rotated 90 degrees counter-clockwise)
    # Original: width=4 (x-direction), height=2 (y-direction)
    # After rotation: width=2, height=4
    ego_rect = Rectangle((-1, -2), 2, 4, linewidth=2,
                         edgecolor='blue', facecolor='lightblue', alpha=0.6, zorder=10)
    ax_bev.add_patch(ego_rect)
    ax_bev.text(0, 0, 'EGO', ha='center', va='center', fontsize=9,
               fontweight='bold', color='darkblue', zorder=11)

    # Draw ego heading arrow (rotated 90 degrees counter-clockwise)
    ego_state = label_data['ego_state']
    heading = ego_state['heading']
    arrow_length = 5.0
    dx = arrow_length * np.cos(heading)
    dy = arrow_length * np.sin(heading)
    # Counter-clockwise 90deg rotation: (x, y) -> (-y, x)
    dx_rot = -dy
    dy_rot = dx
    ax_bev.arrow(0, 0, dx_rot, dy_rot, head_width=1.5, head_length=1.0,
                fc='blue', ec='blue', alpha=0.7, zorder=9)

    # Get ego pose for transforming objects
    ego_pose = nusc.get('ego_pose', sample['data']['LIDAR_TOP'])
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])

    # Draw detected objects
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)

        # Transform to ego frame
        obj_translation_world = np.array(ann['translation'])
        obj_translation_ego = obj_translation_world - ego_translation
        obj_translation_ego = ego_rotation.inverse.rotate(obj_translation_ego)

        x, y = obj_translation_ego[:2]

        # Apply counter-clockwise 90-degree rotation: (x, y) -> (-y, x)
        x_rot = -y
        y_rot = x

        # Only show objects within BEV range
        if -50 <= x_rot <= 50 and -50 <= y_rot <= 50:
            # Color by category
            category = ann['category_name']
            if 'vehicle' in category:
                color = 'green'
                size = 2.0
            elif 'pedestrian' in category:
                color = 'orange'
                size = 1.0
            else:
                color = 'gray'
                size = 1.5

            circle = Circle((x_rot, y_rot), size, color=color, alpha=0.5, zorder=5)
            ax_bev.add_patch(circle)
            ax_bev.plot(x_rot, y_rot, 'o', color=color, markersize=4, zorder=6)

    # Add range circles
    for r in [10, 20, 30, 40]:
        circle = Circle((0, 0), r, fill=False, edgecolor='gray',
                       linestyle=':', linewidth=1, alpha=0.4, zorder=1)
        ax_bev.add_patch(circle)
        ax_bev.text(r, 0, f'{r}m', fontsize=7, color='gray', alpha=0.6)

    # 3. Risk Map (full width at bottom)
    ax_risk = plt.subplot2grid((4, 3), (3, 0), colspan=3)

    # Apply 270-degree counter-clockwise rotation (= 90-degree clockwise)
    # This is equivalent to k=3 or k=-1
    risk_map_display = np.rot90(risk_map, k=-1)

    im = ax_risk.imshow(risk_map_display, cmap=RISK_CMAP, vmin=0, vmax=1,
                       extent=[-50, 50, -50, 50], origin='lower', aspect='equal')

    ax_risk.set_title('Risk Map with High-Risk Regions Highlighted (Risk > 0.5)',
                     fontsize=12, fontweight='bold')
    ax_risk.set_xlabel('X (meters)', fontsize=10)
    ax_risk.set_ylabel('Y (meters)', fontsize=10)
    ax_risk.grid(True, alpha=0.2, color='white', linewidth=0.5)

    # Draw ego vehicle as rectangle (pointing upward = forward in BEV)
    ego_width = 2.0   # meters (vehicle width)
    ego_length = 4.5  # meters (vehicle length)

    # Vehicle centered at (0, 0), pointing upward (Y+ direction = forward)
    ego_vehicle = FancyBboxPatch(
        (-ego_width/2, -ego_length/2),  # Bottom-left corner
        ego_width, ego_length,           # Width, height
        boxstyle="round,pad=0.1",
        linewidth=2,
        edgecolor='cyan',
        facecolor='blue',
        alpha=0.7,
        zorder=10
    )
    ax_risk.add_patch(ego_vehicle)

    # Add heading arrow pointing upward (forward direction)
    arrow_start_y = ego_length/2  # Start from front of vehicle
    arrow_length = 3.0
    ax_risk.arrow(0, arrow_start_y, 0, arrow_length,
                 head_width=1.5, head_length=1.0,
                 fc='cyan', ec='white', alpha=0.9, zorder=11, linewidth=2)

    # Draw detected objects on risk map with white outlines
    ego_pose = nusc.get('ego_pose', sample['data']['LIDAR_TOP'])
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])

    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)

        # Transform to ego frame
        obj_translation_world = np.array(ann['translation'])
        obj_translation_ego = obj_translation_world - ego_translation
        obj_translation_ego = ego_rotation.inverse.rotate(obj_translation_ego)

        x, y = obj_translation_ego[:2]

        # Apply counter-clockwise 90-degree rotation: (x, y) -> (-y, x)
        x_rot = -y
        y_rot = x

        if -50 <= x_rot <= 50 and -50 <= y_rot <= 50:
            # Draw white outline for visibility on risk map
            ax_risk.plot(x_rot, y_rot, 'o', color='white', markersize=6,
                        markeredgecolor='black', markeredgewidth=1.2, zorder=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_risk, fraction=0.03, pad=0.02)
    cbar.set_label('Risk Level', fontsize=10)

    # Add statistics text box outside the risk map (in figure coordinates)
    ego_state = label_data['ego_state']
    stats_text = (
        f'Statistics:\n'
        f'Max Risk: {label_data["metadata"]["max_risk"]:.3f}\n'
        f'Mean Risk: {label_data["metadata"]["mean_risk"]:.3f}\n'
        f'High (>0.7): {label_data["metadata"]["high_risk_cells"]} cells\n'
        f'Medium (0.3-0.7): {label_data["metadata"]["medium_risk_cells"]} cells\n'
        f'Low (0.0-0.3): {label_data["metadata"]["low_risk_cells"]} cells\n'
        f'Ego Velocity: {ego_state["velocity"]:.1f} m/s'
    )
    # Place text box to the left of risk map in figure coordinates
    fig.text(0.02, 0.18, stats_text,
            fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='black', linewidth=1.5))

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_by_scene(nusc, labels_dict, args, output_dir):
    """
    Visualize samples organized by scene in chronological order

    Args:
        nusc: NuScenes instance
        labels_dict: Labels dictionary {scene_token: [labels]}
        args: Command line arguments
        output_dir: Output directory
    """
    # Filter scenes if specified
    if args.scenes:
        # Get scene tokens for specified scene names
        scene_name_to_token = {nusc.get('scene', s['token'])['name']: s['token']
                               for s in nusc.scene}
        selected_scene_tokens = [scene_name_to_token.get(name)
                                for name in args.scenes if name in scene_name_to_token]

        # Filter labels
        filtered_labels = {token: labels_dict[token]
                          for token in selected_scene_tokens
                          if token in labels_dict}
    else:
        filtered_labels = labels_dict

    print(f"\nProcessing {len(filtered_labels)} scene(s)...\n")

    total_visualized = 0

    for scene_token, scene_labels in filtered_labels.items():
        scene = nusc.get('scene', scene_token)
        scene_name = scene['name']

        # Filter by min_risk
        high_risk_samples = [label for label in scene_labels
                            if label['metadata']['max_risk'] >= args.min_risk]

        if len(high_risk_samples) == 0:
            print(f"⚠️  Scene {scene_name}: No samples with risk >= {args.min_risk}")
            continue

        # Scene labels are already in chronological order
        # Select samples (either all or limited by num_samples)
        num_to_viz = min(args.num_samples, len(high_risk_samples))
        selected_samples = high_risk_samples[:num_to_viz]

        print(f"📁 Scene {scene_name}: Visualizing {num_to_viz}/{len(high_risk_samples)} samples")

        # Create scene-specific subdirectory
        scene_output_dir = output_dir / scene_name
        scene_output_dir.mkdir(exist_ok=True)

        # Visualize each sample
        for i, label_data in enumerate(tqdm(selected_samples, desc=f"  {scene_name}")):
            sample_token = label_data['sample_token']
            max_risk = label_data['metadata']['max_risk']

            # Filename: scene_name/frame_XXX_risk_Y.YYY_token.png
            save_path = scene_output_dir / f"frame_{i:03d}_risk{max_risk:.3f}_{sample_token[:8]}.png"

            try:
                visualize_sample(nusc, label_data, save_path, args.dataroot)
                total_visualized += 1
            except Exception as e:
                print(f"\n⚠️  Error visualizing {sample_token}: {e}")
                continue

        print(f"  ✓ Saved to {scene_output_dir}/\n")

    print(f"\n✅ Generated {total_visualized} visualizations across {len(filtered_labels)} scene(s)")
    print(f"Output directory: {output_dir}")


def visualize_random_samples(nusc, labels_dict, args, output_dir):
    """
    Original visualization mode: random selection across all scenes

    Args:
        nusc: NuScenes instance
        labels_dict: Labels dictionary
        args: Command line arguments
        output_dir: Output directory
    """
    # Get high-risk samples
    print(f"Finding samples with max_risk >= {args.min_risk}...")
    high_risk_samples = get_high_risk_samples(labels_dict, args.min_risk)
    print(f"Found {len(high_risk_samples)} high-risk samples")

    if len(high_risk_samples) == 0:
        print("❌ No high-risk samples found! Try lowering --min_risk")
        return

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Select random samples
    num_to_viz = min(args.num_samples, len(high_risk_samples))
    selected_samples = random.sample(high_risk_samples, num_to_viz)

    # Sort by max_risk (descending) for better visualization
    selected_samples.sort(key=lambda x: x['metadata']['max_risk'], reverse=True)

    print(f"\nGenerating visualizations for {num_to_viz} samples...")

    # Generate visualizations
    for i, label_data in enumerate(tqdm(selected_samples, desc="Visualizing")):
        sample_token = label_data['sample_token']
        max_risk = label_data['metadata']['max_risk']
        save_path = output_dir / f"sample_{i:03d}_risk{max_risk:.3f}_{sample_token[:8]}.png"

        try:
            visualize_sample(nusc, label_data, save_path, args.dataroot)
        except Exception as e:
            print(f"\n⚠️  Error visualizing sample {sample_token}: {e}")
            continue

    print(f"\n✅ Generated {num_to_viz} visualizations in {output_dir}")
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Files created: sample_000.png to sample_{num_to_viz-1:03d}.png")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize risk map samples')
    parser.add_argument('--labels', type=str,
                        default='data/emergence_risk/risk_labels_train.pkl',
                        help='Path to labels pickle file')
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='nuScenes version')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to visualize per scene')
    parser.add_argument('--output_dir', type=str, default='visualizations/risk_samples',
                        help='Output directory for visualizations')
    parser.add_argument('--min_risk', type=float, default=0.5,
                        help='Minimum max risk for sample selection')
    parser.add_argument('--scenes', type=str, nargs='+', default=None,
                        help='Specific scenes to visualize (e.g., scene-0061)')
    parser.add_argument('--by_scene', action='store_true',
                        help='Organize output by scene in chronological order')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("RISK SAMPLE VISUALIZATION")
    print("=" * 80)
    print(f"Labels:      {args.labels}")
    print(f"Dataroot:    {args.dataroot}")
    print(f"Output dir:  {output_dir}")
    print(f"Mode:        {'Scene-based (chronological)' if args.by_scene else 'Random selection'}")
    if args.scenes:
        print(f"Scenes:      {', '.join(args.scenes)}")
    print(f"Samples:     {args.num_samples} {'per scene' if args.by_scene else 'total'}")
    print(f"Min risk:    {args.min_risk}")
    print("=" * 80 + "\n")

    # Load nuScenes
    print("Loading nuScenes dataset...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # Load labels
    labels_dict = load_labels(args.labels)

    if args.by_scene:
        # Scene-based visualization (chronological order)
        visualize_by_scene(nusc, labels_dict, args, output_dir)
    else:
        # Original random selection mode
        visualize_random_samples(nusc, labels_dict, args, output_dir)


if __name__ == '__main__':
    main()
