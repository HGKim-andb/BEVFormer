#!/usr/bin/env python3
"""
Risk Factor Visualization Script

This script visualizes individual risk factors (I_traj, O, U, P) and the final risk R
for selected nuScenes samples.

Risk Formula: R = I_traj × O × U × P

Where:
- I_traj: Trajectory indicator (binary: 0 or 1)
- O: Occlusion factor (0~1)
- U: Urgency factor (0~1) - TTC-based
- P: Proximity weight (0~1) - distance to trajectory

Usage:
    python tools/visualize_risk_factors.py \
        --dataroot /path/to/nuscenes \
        --labels data/emergence_risk/risk_labels_val.pkl \
        --output factor_visualization.png
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import sys

try:
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion
except ImportError:
    print("Error: nuscenes-devkit not installed. Install with: pip install nuscenes-devkit")
    sys.exit(1)

# Import risk utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.risk_utils import (
    CONFIG, get_ego_state, get_detected_objects,
    compute_cell_features, grid_to_world
)


# Colormaps for different factors
def create_colormaps():
    """Create colormaps for each factor visualization"""
    # I_traj: Binary (black/white)
    i_traj_colors = [(0.1, 0.1, 0.1), (0.9, 0.9, 0.9)]
    cmap_i_traj = LinearSegmentedColormap.from_list('i_traj', i_traj_colors, N=2)

    # O: Occlusion (black to purple)
    o_colors = [(0.0, 0.0, 0.0), (0.3, 0.0, 0.3), (0.6, 0.0, 0.6), (0.9, 0.3, 0.9)]
    cmap_o = LinearSegmentedColormap.from_list('occlusion', o_colors, N=256)

    # U: Urgency (black to orange)
    u_colors = [(0.0, 0.0, 0.0), (0.4, 0.2, 0.0), (0.8, 0.4, 0.0), (1.0, 0.6, 0.0)]
    cmap_u = LinearSegmentedColormap.from_list('urgency', u_colors, N=256)

    # P: Proximity (black to blue)
    p_colors = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.4), (0.0, 0.3, 0.7), (0.3, 0.6, 1.0)]
    cmap_p = LinearSegmentedColormap.from_list('proximity', p_colors, N=256)

    # R: Final risk (black to red)
    r_colors = [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.6, 0.0, 0.0), (0.9, 0.0, 0.0), (1.0, 0.2, 0.0)]
    cmap_r = LinearSegmentedColormap.from_list('risk', r_colors, N=256)

    return {
        'I_traj': cmap_i_traj,
        'O': cmap_o,
        'U': cmap_u,
        'P': cmap_p,
        'R': cmap_r
    }


def compute_individual_factors(nusc, sample, config=CONFIG):
    """
    Compute individual risk factors for each grid cell

    Args:
        nusc: NuScenes instance
        sample: Sample dict
        config: Configuration

    Returns:
        Dict with factor maps: I_traj, O, U, P, R
    """
    bev_h = config['bev_h']
    bev_w = config['bev_w']
    params = config['risk_params']

    # Initialize factor maps
    I_traj_map = np.zeros((bev_h, bev_w), dtype=np.float32)
    O_map = np.zeros((bev_h, bev_w), dtype=np.float32)
    U_map = np.zeros((bev_h, bev_w), dtype=np.float32)
    P_map = np.zeros((bev_h, bev_w), dtype=np.float32)
    R_map = np.zeros((bev_h, bev_w), dtype=np.float32)

    # Get ego state
    ego_state = get_ego_state(nusc, sample)

    # Get ego pose
    ego_pose = nusc.get('ego_pose', sample['data']['LIDAR_TOP'])

    # Get detected objects
    objects = get_detected_objects(nusc, sample, ego_pose)

    # Process each grid cell
    for gy in range(bev_h):
        for gx in range(bev_w):
            # Convert grid to world coordinates
            world_x, world_y = grid_to_world(gx, gy, config)
            cell_pos = np.array([world_x, world_y])

            # Compute features
            features = compute_cell_features(cell_pos, ego_state, objects, config)

            # Extract individual factors

            # 1. I_traj: Trajectory indicator (binary)
            is_on_traj = features.get('is_on_trajectory', False)
            is_future = features.get('is_future', False)
            I_traj = 1.0 if (is_on_traj and is_future) else 0.0
            I_traj_map[gy, gx] = I_traj

            # If not on trajectory or not in future, other factors are 0
            if I_traj == 0.0:
                continue

            # 2. O: Occlusion factor
            if features.get('is_occluded', False):
                A_ref = params['A_ref']
                occluder_area = features.get('occluder_area', 0.0)
                O = min(occluder_area / A_ref, 1.0)
            else:
                O = 0.0
            O_map[gy, gx] = O

            # If not occluded, R is 0
            if O == 0.0:
                continue

            # 3. U: Urgency factor (TTC-based)
            ttc = features.get('time_to_collision', float('inf'))
            T_safe = params['T_safe']
            T_critical = params['T_critical']

            if ttc >= T_safe:
                U = 0.0
            elif ttc <= T_critical:
                U = 1.0
            else:
                U = (T_safe - ttc) / (T_safe - T_critical)
            U_map[gy, gx] = U

            # 4. P: Proximity weight
            d_traj = features.get('distance_to_trajectory', float('inf'))
            d_close = params['d_close']
            d_far = params['d_far']

            if d_traj <= d_close:
                P = 1.0
            elif d_traj >= d_far:
                P = 0.0
            else:
                P = (d_far - d_traj) / (d_far - d_close)
            P_map[gy, gx] = P

            # 5. R: Final risk
            R = O * U * P
            R_map[gy, gx] = R

    return {
        'I_traj': I_traj_map,
        'O': O_map,
        'U': U_map,
        'P': P_map,
        'R': R_map,
        'ego_state': ego_state
    }


def find_representative_samples(labels_dict, nusc, scenarios=['urban', 'parking', 'highway', 'open']):
    """
    Find 4 representative samples matching different scenarios with high risk values

    Args:
        labels_dict: Labels dictionary
        nusc: NuScenes instance
        scenarios: List of scenario types to find

    Returns:
        List of (scenario_name, sample_token, label_data) tuples
    """
    selected = []
    used_sample_tokens = set()

    # Flatten all samples with their scene info
    all_samples = []
    for scene_token, scene_labels in labels_dict.items():
        scene = nusc.get('scene', scene_token)
        scene_name = scene['name'].lower()
        scene_desc = scene['description'].lower()

        for label in scene_labels:
            all_samples.append({
                'scene_token': scene_token,
                'scene_name': scene_name,
                'scene_desc': scene_desc,
                'label': label,
                'max_risk': label['metadata']['max_risk'],
                'high_risk_cells': label['metadata'].get('high_risk_cells', 0)
            })

    # Sort by max_risk (descending) to prefer high-risk samples
    all_samples.sort(key=lambda x: (x['max_risk'], x['high_risk_cells']), reverse=True)

    # Select top 4 samples with highest risk from different scenes
    used_scenes = set()
    scenario_names = ['urban', 'parking', 'highway', 'open']

    # First pass: strictly enforce different scenes
    for sample_info in all_samples:
        if len(selected) >= 4:
            break

        scene_token = sample_info['scene_token']
        sample_token = sample_info['label']['sample_token']

        # Skip if scene already used (to get variety)
        if scene_token in used_scenes:
            continue

        scenario = scenario_names[len(selected)] if len(selected) < len(scenario_names) else f"sample_{len(selected)+1}"
        selected.append((scenario, sample_token, sample_info['label']))
        used_scenes.add(scene_token)
        used_sample_tokens.add(sample_token)
        print(f"  Selected {scenario}: scene={sample_info['scene_name']}, "
              f"risk={sample_info['max_risk']:.3f}, high_risk_cells={sample_info['high_risk_cells']}")

    # If we don't have 4 samples, relax scene constraint
    if len(selected) < 4:
        for sample_info in all_samples:
            if len(selected) >= 4:
                break

            sample_token = sample_info['label']['sample_token']
            if sample_token in used_sample_tokens:
                continue

            scenario = scenario_names[len(selected)] if len(selected) < len(scenario_names) else f"sample_{len(selected)+1}"
            selected.append((scenario, sample_token, sample_info['label']))
            used_sample_tokens.add(sample_token)
            print(f"  Added {scenario}: scene={sample_info['scene_name']}, "
                  f"risk={sample_info['max_risk']:.3f}")

    return selected


def draw_front_camera(ax, nusc, sample, dataroot):
    """
    Draw front camera image

    Args:
        ax: Matplotlib axis
        nusc: NuScenes instance
        sample: Sample dict
        dataroot: Path to nuScenes data
    """
    import cv2

    cam_token = sample['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)
    img_path = Path(dataroot) / cam_data['filename']

    if img_path.exists():
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, 'Image not found', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='red')
        ax.set_facecolor('#1a1a2e')

    ax.axis('off')


def draw_bev_with_objects(ax, nusc, sample, label_data):
    """
    Draw BEV image with detected objects

    Args:
        ax: Matplotlib axis
        nusc: NuScenes instance
        sample: Sample dict
        label_data: Label data with risk map
    """
    from matplotlib.patches import Circle, Rectangle
    from pyquaternion import Quaternion

    # Set BEV range
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')  # Dark background

    # Draw range circles
    for r in [10, 20, 30, 40]:
        circle = Circle((0, 0), r, fill=False, edgecolor='#404060',
                        linestyle=':', linewidth=0.8, alpha=0.5)
        ax.add_patch(circle)

    # Get ego pose
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

        # Apply rotation for display: (x, y) -> (-y, x)
        x_rot = -y
        y_rot = x

        if -50 <= x_rot <= 50 and -50 <= y_rot <= 50:
            category = ann['category_name']

            # Color and size by category
            if 'vehicle' in category:
                color = '#00ff88'
                size = 2.5
            elif 'pedestrian' in category:
                color = '#ffaa00'
                size = 1.2
            elif 'bicycle' in category or 'motorcycle' in category:
                color = '#00aaff'
                size = 1.5
            else:
                color = '#888888'
                size = 1.5

            circle = Circle((x_rot, y_rot), size, color=color, alpha=0.6, zorder=5)
            ax.add_patch(circle)

    # Draw ego vehicle
    ego_width = 2.0
    ego_length = 4.5
    ego_rect = Rectangle((-ego_width/2, -ego_length/2), ego_width, ego_length,
                         linewidth=2, edgecolor='cyan', facecolor='#0066cc', alpha=0.8, zorder=10)
    ax.add_patch(ego_rect)

    # Heading arrow
    ax.arrow(0, ego_length/2, 0, 4,
            head_width=1.5, head_length=1.0,
            fc='cyan', ec='white', alpha=0.9, zorder=11, linewidth=2)

    # Grid
    ax.grid(True, alpha=0.15, color='white', linewidth=0.3)


def visualize_factors(factor_maps_list, sample_infos, nusc, output_path, dataroot, dpi=300):
    """
    Create visualization with 4 rows (samples) x 7 columns (Front + BEV + 5 factors)

    Args:
        factor_maps_list: List of factor map dicts for each sample
        sample_infos: List of (scenario_name, sample_token, label_data)
        nusc: NuScenes instance
        output_path: Output file path
        dataroot: Path to nuScenes data
        dpi: DPI for output
    """
    colormaps = create_colormaps()
    factor_names = ['I_traj', 'O', 'U', 'P', 'R']
    factor_titles = [
        r'$I_{traj}$ (Trajectory)',
        r'$O$ (Occlusion)',
        r'$U$ (Urgency)',
        r'$P$ (Proximity)',
        r'$R$ (Final Risk)'
    ]

    # Create figure: 4 rows x 7 columns (Front + BEV + 5 factors)
    fig, axes = plt.subplots(4, 7, figsize=(28, 16))

    # Scenario labels for rows
    row_labels = [
        '(a) Urban Intersection',
        '(b) Parking Area',
        '(c) Highway',
        '(d) Open Road'
    ]

    for row_idx, (factor_maps, sample_info) in enumerate(zip(factor_maps_list, sample_infos)):
        scenario_name, sample_token, label_data = sample_info

        # Use predefined row labels
        row_label = row_labels[row_idx] if row_idx < len(row_labels) else f'({chr(97+row_idx)}) {scenario_name.capitalize()}'

        # Get sample for visualization
        sample = nusc.get('sample', sample_token)

        # Column 0: Front camera
        ax_front = axes[row_idx, 0]
        draw_front_camera(ax_front, nusc, sample, dataroot)
        if row_idx == 0:
            ax_front.set_title('Front Camera', fontsize=12, fontweight='bold', pad=10)
        # Add row label as text on the left side
        ax_front.text(-0.1, 0.5, row_label, transform=ax_front.transAxes,
                     fontsize=11, fontweight='bold', va='center', ha='right', rotation=90)

        # Column 1: BEV with objects
        ax_bev = axes[row_idx, 1]
        draw_bev_with_objects(ax_bev, nusc, sample, label_data)

        if row_idx == 0:
            ax_bev.set_title('BEV Scene', fontsize=12, fontweight='bold', pad=10)
        if row_idx == 3:
            ax_bev.set_xlabel('X (m)', fontsize=9)
        ax_bev.set_xticks([-40, -20, 0, 20, 40])
        ax_bev.set_yticks([-40, -20, 0, 20, 40])
        ax_bev.tick_params(labelsize=7)

        # Columns 2-6: Factor maps
        for col_idx, factor_name in enumerate(factor_names):
            ax = axes[row_idx, col_idx + 2]

            # Get factor map and rotate for display
            factor_map = factor_maps[factor_name]
            factor_display = np.rot90(factor_map, k=-1)  # Rotate for BEV view

            # Plot heatmap
            cmap = colormaps[factor_name]
            im = ax.imshow(factor_display, cmap=cmap, vmin=0, vmax=1,
                          extent=[-50, 50, -50, 50], origin='lower', aspect='equal')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

            # Draw ego vehicle
            ego_width = 2.0
            ego_length = 4.5
            ego_vehicle = FancyBboxPatch(
                (-ego_width/2, -ego_length/2), ego_width, ego_length,
                boxstyle="round,pad=0.1", linewidth=1.5,
                edgecolor='cyan', facecolor='blue', alpha=0.7, zorder=10
            )
            ax.add_patch(ego_vehicle)

            # Heading arrow
            ax.arrow(0, ego_length/2, 0, 3,
                    head_width=1.2, head_length=0.8,
                    fc='cyan', ec='white', alpha=0.9, zorder=11, linewidth=1.5)

            # Column title (only for first row)
            if row_idx == 0:
                ax.set_title(factor_titles[col_idx], fontsize=12, fontweight='bold', pad=10)

            # Set axis labels only for bottom row
            if row_idx == 3:
                ax.set_xlabel('X (m)', fontsize=9)

            # Minimal ticks
            ax.set_xticks([-40, -20, 0, 20, 40])
            ax.set_yticks([-40, -20, 0, 20, 40])
            ax.tick_params(labelsize=7)

            # Light grid
            ax.grid(True, alpha=0.2, color='white', linewidth=0.3)
            ax.set_xlim(-50, 50)
            ax.set_ylim(-50, 50)

    # Main title
    fig.suptitle('Individual Risk Factor Visualization\n' +
                 r'$R = I_{traj} \times O \times U \times P$',
                 fontsize=16, fontweight='bold', y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize individual risk factors')
    parser.add_argument('--dataroot', type=str, required=True,
                       help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                       help='nuScenes version')
    parser.add_argument('--labels', type=str,
                       default='data/emergence_risk/risk_labels_val.pkl',
                       help='Path to risk labels pickle file')
    parser.add_argument('--output', type=str, default='factor_visualization.png',
                       help='Output file path')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Output DPI')
    parser.add_argument('--sample_tokens', type=str, nargs='+', default=None,
                       help='Specific sample tokens to visualize (4 required)')

    args = parser.parse_args()

    print("=" * 80)
    print("RISK FACTOR VISUALIZATION")
    print("=" * 80)
    print(f"Dataroot:  {args.dataroot}")
    print(f"Labels:    {args.labels}")
    print(f"Output:    {args.output}")
    print(f"DPI:       {args.dpi}")
    print("=" * 80)

    # Load nuScenes
    print("\nLoading nuScenes dataset...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # Load labels
    print("Loading labels...")
    with open(args.labels, 'rb') as f:
        labels_dict = pickle.load(f)

    print(f"Loaded {sum(len(v) for v in labels_dict.values())} samples from {len(labels_dict)} scenes")

    # Find or use specified samples
    if args.sample_tokens:
        # Use specified sample tokens
        sample_infos = []
        for i, token in enumerate(args.sample_tokens[:4]):
            # Find label for this token
            found = False
            for scene_labels in labels_dict.values():
                for label in scene_labels:
                    if label['sample_token'] == token:
                        sample_infos.append((f"sample_{i+1}", token, label))
                        found = True
                        break
                if found:
                    break
            if not found:
                print(f"Warning: Sample token {token} not found in labels")
    else:
        # Find representative samples
        print("\nFinding representative samples for different scenarios...")
        sample_infos = find_representative_samples(labels_dict, nusc)

    if len(sample_infos) < 4:
        print(f"\nError: Need 4 samples but only found {len(sample_infos)}")
        return

    # Compute factors for each sample
    print("\nComputing risk factors for each sample...")
    factor_maps_list = []

    for i, (scenario, sample_token, label_data) in enumerate(sample_infos):
        print(f"  Processing sample {i+1}/4: {scenario} (token: {sample_token[:8]}...)")

        sample = nusc.get('sample', sample_token)
        factor_maps = compute_individual_factors(nusc, sample)
        factor_maps_list.append(factor_maps)

        # Print statistics
        R = factor_maps['R']
        print(f"    R: max={R.max():.3f}, mean={R.mean():.4f}, nonzero={np.count_nonzero(R)}")

    # Create visualization
    print("\nCreating visualization...")
    visualize_factors(factor_maps_list, sample_infos, nusc, args.output, args.dataroot, args.dpi)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
