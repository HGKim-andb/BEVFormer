#!/usr/bin/env python3
"""
Emergence Label Generation Script

This script creates emergence labels for the nuScenes dataset.
Emergence is defined as: objects that appear in future frames (t+1, t+2, t+3)
but were NOT present in past frames (t-5 to t-1).

Output format:
- emergence_labels_train.pkl: Training set labels
- emergence_labels_val.pkl: Validation set labels
- label_statistics.json: Statistics
- label_config.json: Configuration used
"""

import numpy as np
import pickle
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import sys

try:
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion
except ImportError:
    print("Error: nuscenes-devkit not installed. Install with: pip install nuscenes-devkit pyquaternion")
    sys.exit(1)


# Configuration
CONFIG = {
    'lookback_frames': 5,
    'lookahead_frames': 3,
    'bev_range': [-50, 50, -50, 50],  # [x_min, x_max, y_min, y_max]
    'bev_resolution': 0.5,  # meters per pixel
    'grid_size': 200,  # 100m / 0.5m = 200 pixels
    'min_distance': 5.0,  # meters (increased from 2.0 to reduce very close objects)
    'max_distance': 40.0,  # meters
    'gaussian_sigma': 2.0,  # pixels
    'emergence_mode': 'strict',  # 'strict' = only vis 1->2+, 'relaxed' = vis 0-1 -> 2+
    'valid_categories': [
        'vehicle.car',
        'vehicle.truck',
        'vehicle.bus',
        'vehicle.bicycle',
        'vehicle.motorcycle',
        'human.pedestrian.adult',
        'human.pedestrian.child',
        'human.pedestrian.construction_worker',
        'human.pedestrian.police_officer',
    ]
}

# Category mapping for simplified classes
CATEGORY_MAP = {
    'vehicle': 1,
    'pedestrian': 2,
    'bicycle': 3,
    'motorcycle': 4,
}


def simplify_category(full_category_name):
    """Convert nuScenes category to simplified category"""
    if 'bicycle' in full_category_name:
        return 'bicycle'
    elif 'motorcycle' in full_category_name:
        return 'motorcycle'
    elif 'vehicle' in full_category_name:
        return 'vehicle'
    elif 'pedestrian' in full_category_name:
        return 'pedestrian'
    return 'other'


def world_to_grid(x, y, bev_range, resolution):
    """
    Convert world coordinates to BEV grid indices

    Args:
        x, y: World coordinates in meters (relative to ego vehicle)
        bev_range: [x_min, x_max, y_min, y_max]
        resolution: meters per pixel

    Returns:
        grid_x, grid_y: Grid indices (None if out of bounds)
    """
    x_min, x_max, y_min, y_max = bev_range

    # Convert to grid coordinates
    grid_x = int((x - x_min) / resolution)
    grid_y = int((y - y_min) / resolution)

    # Check bounds
    grid_w = int((x_max - x_min) / resolution)
    grid_h = int((y_max - y_min) / resolution)

    if 0 <= grid_x < grid_w and 0 <= grid_y < grid_h:
        return grid_x, grid_y
    return None, None


def add_gaussian_to_grid(grid, center, sigma):
    """
    Add a Gaussian blob to the grid at the specified center

    Args:
        grid: numpy array [H, W]
        center: (y, x) in grid coordinates
        sigma: Gaussian standard deviation in pixels
    """
    y, x = center
    h, w = grid.shape

    # Kernel size (3 sigma covers 99.7% of the distribution)
    kernel_size = int(sigma * 3)

    # Define region of interest
    y_min = max(0, y - kernel_size)
    y_max = min(h, y + kernel_size + 1)
    x_min = max(0, x - kernel_size)
    x_max = min(w, x + kernel_size + 1)

    if y_min >= y_max or x_min >= x_max:
        return

    # Create coordinate grids
    y_range = np.arange(y_min, y_max)
    x_range = np.arange(x_min, x_max)
    yy, xx = np.meshgrid(y_range, x_range, indexing='ij')

    # Compute Gaussian
    gaussian = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))

    # Add to grid (use maximum to handle overlaps)
    grid[y_min:y_max, x_min:x_max] = np.maximum(
        grid[y_min:y_max, x_min:x_max],
        gaussian
    )


def get_ego_pose(nusc, sample_token):
    """
    Get ego vehicle pose for a sample

    Args:
        nusc: NuScenes instance
        sample_token: Sample token

    Returns:
        (translation, rotation_matrix) tuple
    """
    sample = nusc.get('sample', sample_token)

    # Use LIDAR_TOP sensor for ego pose
    sample_data_token = sample['data']['LIDAR_TOP']
    sample_data = nusc.get('sample_data', sample_data_token)
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])

    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])

    return ego_translation, ego_rotation.rotation_matrix


def global_to_ego(global_position, ego_translation, ego_rotation_matrix):
    """
    Transform global coordinates to ego-relative coordinates

    Args:
        global_position: (x, y, z) in global frame
        ego_translation: Ego translation in global frame
        ego_rotation_matrix: Ego rotation matrix (3x3)

    Returns:
        (x, y, z) in ego-relative frame
    """
    global_pos = np.array(global_position)

    # Translate to ego origin, then rotate to ego frame
    ego_relative = ego_rotation_matrix.T @ (global_pos - ego_translation)

    return ego_relative


def get_scene_samples(nusc, scene_token):
    """
    Get all sample tokens in a scene in chronological order

    Args:
        nusc: NuScenes instance
        scene_token: Scene token

    Returns:
        List of sample tokens
    """
    scene = nusc.get('scene', scene_token)
    samples = []

    sample_token = scene['first_sample_token']
    while sample_token:
        samples.append(sample_token)
        sample = nusc.get('sample', sample_token)
        sample_token = sample['next']

    return samples


def get_sample_objects(nusc, sample_token, valid_categories):
    """
    Get all object instance tokens in a sample

    Args:
        nusc: NuScenes instance
        sample_token: Sample token
        valid_categories: List of valid category names

    Returns:
        Set of instance tokens
    """
    if sample_token is None or sample_token == '':
        return set()

    sample = nusc.get('sample', sample_token)
    instance_tokens = set()

    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if ann['category_name'] in valid_categories:
            instance_tokens.add(ann['instance_token'])

    return instance_tokens


def get_object_info(nusc, sample_token, instance_token):
    """
    Get position and category of an object in a sample

    Args:
        nusc: NuScenes instance
        sample_token: Sample token
        instance_token: Instance token

    Returns:
        dict with 'position', 'category', 'distance' or None
    """
    if sample_token is None:
        return None

    sample = nusc.get('sample', sample_token)

    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if ann['instance_token'] == instance_token:
            x, y, z = ann['translation']
            distance = np.sqrt(x**2 + y**2)

            return {
                'position': (x, y),
                'category': ann['category_name'],
                'distance': distance
            }

    return None


def process_sample(nusc, sample_tokens, sample_idx, config):
    """
    Process a single sample to generate emergence labels

    Args:
        nusc: NuScenes instance
        sample_tokens: List of all sample tokens in the scene
        sample_idx: Index of current sample
        config: Configuration dict

    Returns:
        Label dict or None if not enough context
    """
    # Check if we have enough past and future frames
    lookback = config['lookback_frames']
    lookahead = config['lookahead_frames']

    if sample_idx < lookback or sample_idx + lookahead >= len(sample_tokens):
        return None

    current_token = sample_tokens[sample_idx]
    sample = nusc.get('sample', current_token)

    # Get ego pose for future frames (we'll need these for coordinate transformation)
    future_ego_poses = {}
    for future_frame in range(1, lookahead + 1):
        future_idx = sample_idx + future_frame
        future_token = sample_tokens[future_idx]
        ego_trans, ego_rot = get_ego_pose(nusc, future_token)
        future_ego_poses[future_frame] = (ego_trans, ego_rot)

    # Step 1: Collect objects from past frames (t-5 to t-1)
    # Track max visibility for each instance
    past_instance_visibility = {}  # instance_token -> max visibility level

    for i in range(sample_idx - lookback, sample_idx):
        past_token = sample_tokens[i]
        past_sample = nusc.get('sample', past_token)

        for ann_token in past_sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            if ann['category_name'] in config['valid_categories']:
                instance_token = ann['instance_token']

                # Get visibility level
                # visibility_token: '1'=0-40%, '2'=40-60%, '3'=60-80%, '4'=80-100%
                visibility_token = ann.get('visibility_token', '0')
                vis_level = int(visibility_token) if visibility_token else 0

                # Track maximum visibility across all past frames
                if instance_token not in past_instance_visibility:
                    past_instance_visibility[instance_token] = vis_level
                else:
                    past_instance_visibility[instance_token] = max(
                        past_instance_visibility[instance_token], vis_level
                    )

    # Step 2: Find emergences in future frames (t+1, t+2, t+3)
    # Emergence = objects that:
    #   1. Were NOT visible enough in past (max visibility < 2, i.e., < 40%)
    #   2. OR were completely new (not in past at all)
    #   3. AND are now visible enough in future (visibility >= 2)
    emergence_info = []

    for future_frame in range(1, lookahead + 1):
        future_idx = sample_idx + future_frame
        future_token = sample_tokens[future_idx]
        future_sample = nusc.get('sample', future_token)

        for ann_token in future_sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)

            if ann['category_name'] not in config['valid_categories']:
                continue

            instance_token = ann['instance_token']

            # Get current visibility
            visibility_token = ann.get('visibility_token', '0')
            current_vis = int(visibility_token) if visibility_token else 0

            # Check if this is an emergence:
            # 1. Currently visible enough (>= 40%)
            # 2. AND was low visibility in past (1-40%, not completely absent)
            # This captures objects that were occluded/far and then appeared
            past_vis = past_instance_visibility.get(instance_token, 0)

            # Require that object existed in past but with low visibility
            # past_vis == 0: completely new object (not emergence)
            # past_vis == 1: existed but occluded (TRUE emergence)
            # past_vis >= 2: already visible (not emergence)
            is_emergence = (current_vis >= 2) and (past_vis == 1)

            if is_emergence:
                # Get position in global coordinates
                global_position = ann['translation']

                # Transform to ego-relative coordinates (for this future frame)
                ego_trans, ego_rot = future_ego_poses[future_frame]
                ego_relative_pos = global_to_ego(global_position, ego_trans, ego_rot)

                x, y, z = ego_relative_pos
                distance = np.sqrt(x**2 + y**2)

                # Check distance constraints
                if config['min_distance'] <= distance <= config['max_distance']:
                    grid_x, grid_y = world_to_grid(x, y, config['bev_range'], config['bev_resolution'])

                    if grid_x is not None and grid_y is not None:
                        emergence_info.append({
                            'frame': future_frame,
                            'position': (x, y),
                            'grid_pos': (grid_x, grid_y),
                            'category': ann['category_name'],
                            'distance': float(distance)
                        })

    # Step 3: Create BEV grids
    grid_size = config['grid_size']
    emergence_mask = np.zeros((lookahead, grid_size, grid_size), dtype=np.float32)
    emergence_class = np.zeros((lookahead, grid_size, grid_size), dtype=np.int32)

    for info in emergence_info:
        frame_idx = info['frame'] - 1  # 0-indexed
        grid_x, grid_y = info['grid_pos']

        # Add Gaussian to mask
        add_gaussian_to_grid(
            emergence_mask[frame_idx],
            (grid_y, grid_x),
            config['gaussian_sigma']
        )

        # Set class
        simple_cat = simplify_category(info['category'])
        class_id = CATEGORY_MAP.get(simple_cat, 0)
        emergence_class[frame_idx, grid_y, grid_x] = class_id

    # Get scene info
    scene_token = sample['scene_token']
    scene = nusc.get('scene', scene_token)

    return {
        'sample_token': current_token,
        'scene_token': scene_token,
        'scene_name': scene['name'],
        'emergence_mask': emergence_mask,
        'emergence_class': emergence_class,
        'num_emergences': len(emergence_info),
        'emergence_info': emergence_info
    }


def process_split(nusc, split_name, config):
    """
    Process all scenes in a split (train or val)

    Args:
        nusc: NuScenes instance
        split_name: 'train' or 'val'
        config: Configuration dict

    Returns:
        dict mapping scene_token to list of labels
    """
    # Get scenes for this split
    split_scenes = []
    for scene in nusc.scene:
        scene_name = scene['name']
        # nuScenes naming convention: scene-XXXX (scene-0001 to scene-0850)
        # Typically train/val split is provided in the metadata
        # For simplicity, we'll use a heuristic or load from splits
        # The proper way is to use nusc.list_scenes() but let's use scene metadata
        if 'description' in scene:
            split_scenes.append(scene['token'])

    # Actually, let's use the proper train/val split
    from nuscenes.utils.splits import create_splits_scenes
    splits = create_splits_scenes()

    if split_name == 'train':
        scene_names = splits['train']
    elif split_name == 'val':
        scene_names = splits['val']
    else:
        raise ValueError(f"Unknown split: {split_name}")

    # Get scene tokens
    scene_tokens = []
    for scene in nusc.scene:
        if scene['name'] in scene_names:
            scene_tokens.append(scene['token'])

    print(f"\nProcessing {split_name} split: {len(scene_tokens)} scenes")

    all_labels = {}
    total_samples = 0
    total_emergences = 0
    positive_samples = 0

    # Process each scene
    for scene_token in tqdm(scene_tokens, desc=f"Processing {split_name} scenes"):
        scene = nusc.get('scene', scene_token)
        sample_tokens = get_scene_samples(nusc, scene_token)

        scene_labels = []
        scene_emergences = 0

        # Process each sample in the scene
        for sample_idx in range(len(sample_tokens)):
            try:
                label = process_sample(nusc, sample_tokens, sample_idx, config)

                if label is not None:
                    scene_labels.append(label)
                    total_samples += 1

                    if label['num_emergences'] > 0:
                        positive_samples += 1
                        total_emergences += label['num_emergences']
                        scene_emergences += label['num_emergences']

            except Exception as e:
                print(f"\nError processing sample {sample_tokens[sample_idx]}: {e}")
                continue

        if scene_labels:
            all_labels[scene_token] = scene_labels

            if scene_emergences > 0:
                print(f"\nScene {scene['name']}: {len(scene_labels)} samples, {scene_emergences} emergences")

    # Print statistics
    print(f"\n{'='*60}")
    print(f"{split_name.upper()} Statistics:")
    print(f"{'='*60}")
    print(f"Total scenes: {len(scene_tokens)}")
    print(f"Total samples: {total_samples}")
    print(f"Samples with emergence: {positive_samples}")
    print(f"Positive ratio: {positive_samples/total_samples*100:.2f}%")
    print(f"Total emergence events: {total_emergences}")
    if positive_samples > 0:
        print(f"Avg emergences per positive sample: {total_emergences/positive_samples:.2f}")
    print(f"{'='*60}\n")

    return all_labels


def compute_statistics(train_labels, val_labels):
    """
    Compute statistics across all labels

    Args:
        train_labels: Training labels dict
        val_labels: Validation labels dict

    Returns:
        Statistics dict
    """
    stats = {
        'train': defaultdict(int),
        'val': defaultdict(int),
        'overall': defaultdict(int)
    }

    for split_name, labels in [('train', train_labels), ('val', val_labels)]:
        total_samples = 0
        positive_samples = 0
        total_emergences = 0

        frame_dist = defaultdict(int)
        category_dist = defaultdict(int)
        distances = []

        for scene_labels in labels.values():
            for label in scene_labels:
                total_samples += 1

                if label['num_emergences'] > 0:
                    positive_samples += 1
                    total_emergences += label['num_emergences']

                    for info in label['emergence_info']:
                        frame_dist[f"t+{info['frame']}"] += 1
                        simple_cat = simplify_category(info['category'])
                        category_dist[simple_cat] += 1
                        distances.append(info['distance'])

        stats[split_name] = {
            'total_samples': total_samples,
            'positive_samples': positive_samples,
            'positive_ratio': positive_samples / total_samples if total_samples > 0 else 0,
            'total_emergences': total_emergences,
            'avg_per_positive': total_emergences / positive_samples if positive_samples > 0 else 0,
            'frame_distribution': dict(frame_dist),
            'category_distribution': dict(category_dist),
            'distance_stats': {
                'mean': float(np.mean(distances)) if distances else 0,
                'median': float(np.median(distances)) if distances else 0,
                'min': float(np.min(distances)) if distances else 0,
                'max': float(np.max(distances)) if distances else 0,
                'std': float(np.std(distances)) if distances else 0,
            } if distances else {}
        }

    return stats


def main():
    parser = argparse.ArgumentParser(description='Generate emergence labels for nuScenes')
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='nuScenes version (default: v1.0-trainval)')
    parser.add_argument('--output_dir', type=str, default='data/emergence_labels',
                        help='Output directory for labels')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Emergence Label Generation")
    print(f"{'='*60}")
    print(f"Dataroot: {args.dataroot}")
    print(f"Version: {args.version}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Load nuScenes
    print("Loading nuScenes dataset...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=args.verbose)
    print(f"Loaded {len(nusc.scene)} scenes\n")

    # Save config
    config_path = output_dir / 'label_config.json'
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f"Saved config to {config_path}\n")

    # Process train split
    train_labels = process_split(nusc, 'train', CONFIG)
    train_path = output_dir / 'emergence_labels_train.pkl'
    with open(train_path, 'wb') as f:
        pickle.dump(train_labels, f)
    print(f"Saved training labels to {train_path}")

    # Process val split
    val_labels = process_split(nusc, 'val', CONFIG)
    val_path = output_dir / 'emergence_labels_val.pkl'
    with open(val_path, 'wb') as f:
        pickle.dump(val_labels, f)
    print(f"Saved validation labels to {val_path}")

    # Compute and save statistics
    stats = compute_statistics(train_labels, val_labels)
    stats_path = output_dir / 'label_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - emergence_labels_train.pkl")
    print(f"  - emergence_labels_val.pkl")
    print(f"  - label_statistics.json")
    print(f"  - label_config.json")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
