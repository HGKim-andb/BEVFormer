#!/usr/bin/env python3
"""
BEV Risk Map Label Generation

This script generates risk maps for occluded regions in the nuScenes dataset.
Each risk map is a 200x200 grid where each cell represents the risk of an
object suddenly appearing from an occluded region.
"""

import numpy as np
import pickle
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import sys
from typing import Dict, List
import multiprocessing as mp
from functools import partial

try:
    from nuscenes.nuscenes import NuScenes
except ImportError:
    print("Error: nuscenes-devkit not installed. Install with: pip install nuscenes-devkit")
    sys.exit(1)

from risk_utils import (
    CONFIG,
    get_ego_state,
    get_detected_objects,
    compute_cell_features,
    compute_risk_score,
    grid_to_world,
)


def generate_risk_map(nusc, sample: Dict, config: Dict = CONFIG) -> Dict:
    """
    Generate risk map for a single sample

    Args:
        nusc: NuScenes instance
        sample: Sample dict
        config: Configuration dict

    Returns:
        {
            'sample_token': str,
            'scene_token': str,
            'risk_map': np.array([200, 200]),
            'ego_state': dict,
            'metadata': dict,
        }
    """
    # Extract ego state
    ego_state = get_ego_state(nusc, sample)

    # Extract detected objects
    ego_pose = nusc.get('ego_pose', sample['data']['LIDAR_TOP'])
    objects = get_detected_objects(nusc, sample, ego_pose)

    # Initialize risk map
    risk_map = np.zeros((config['bev_h'], config['bev_w']), dtype=np.float32)

    # Optimization: Skip computation if no objects
    if len(objects) == 0:
        metadata = {
            'max_risk': 0.0,
            'mean_risk': 0.0,
            'high_risk_cells': 0,
            'medium_risk_cells': 0,
            'low_risk_cells': 0,
        }

        return {
            'sample_token': sample['token'],
            'scene_token': sample['scene_token'],
            'scene_name': nusc.get('scene', sample['scene_token'])['name'],
            'risk_map': risk_map,
            'ego_state': {
                'position': ego_state['position'].tolist(),
                'velocity': float(ego_state['velocity']),
                'heading': float(ego_state['heading']),
            },
            'metadata': metadata,
        }

    # Compute risk for each cell
    # Use step size for faster computation (can be adjusted)
    step = 2  # Process every 2nd cell for faster testing
    for grid_y in range(0, config['bev_h'], step):
        for grid_x in range(0, config['bev_w'], step):
            # Convert grid to world coordinates
            cell_world_x, cell_world_y = grid_to_world(grid_x, grid_y, config)
            cell_pos = np.array([cell_world_x, cell_world_y])

            # Skip cells too far from ego (optimization)
            cell_dist = np.linalg.norm(cell_pos - ego_state['position'])
            if cell_dist > 50.0:  # Beyond BEV range
                continue

            # Compute features
            features = compute_cell_features(cell_pos, ego_state, objects, config)

            # Compute risk
            risk = compute_risk_score(features)

            # Fill in the stepped cells (simple interpolation)
            if step > 1:
                for dy in range(step):
                    for dx in range(step):
                        y_idx = min(grid_y + dy, config['bev_h'] - 1)
                        x_idx = min(grid_x + dx, config['bev_w'] - 1)
                        risk_map[y_idx, x_idx] = risk
            else:
                risk_map[grid_y, grid_x] = risk

    # Compute metadata
    metadata = {
        'max_risk': float(risk_map.max()),
        'mean_risk': float(risk_map.mean()),
        'high_risk_cells': int((risk_map > 0.7).sum()),
        'medium_risk_cells': int(((risk_map > 0.3) & (risk_map <= 0.7)).sum()),
        'low_risk_cells': int(((risk_map > 0.0) & (risk_map <= 0.3)).sum()),
    }

    return {
        'sample_token': sample['token'],
        'scene_token': sample['scene_token'],
        'scene_name': nusc.get('scene', sample['scene_token'])['name'],
        'risk_map': risk_map,
        'ego_state': {
            'position': ego_state['position'].tolist(),
            'velocity': float(ego_state['velocity']),
            'heading': float(ego_state['heading']),
        },
        'metadata': metadata,
    }


def get_scene_samples(nusc, scene_token: str) -> List[str]:
    """
    Get all sample tokens in a scene

    Args:
        nusc: NuScenes instance
        scene_token: Scene token

    Returns:
        List of sample tokens
    """
    scene = nusc.get('scene', scene_token)
    sample_tokens = []

    # Start from first sample
    current_sample_token = scene['first_sample_token']

    while current_sample_token != '':
        sample_tokens.append(current_sample_token)
        sample = nusc.get('sample', current_sample_token)
        current_sample_token = sample['next']

    return sample_tokens


def process_sample_wrapper(sample_token: str, nusc, config: Dict) -> Dict:
    """
    Wrapper function for multiprocessing

    Args:
        sample_token: Sample token
        nusc: NuScenes instance
        config: Configuration dict

    Returns:
        Risk label dict or None if error
    """
    try:
        sample = nusc.get('sample', sample_token)
        label = generate_risk_map(nusc, sample, config)
        return label
    except Exception as e:
        print(f"\n⚠️  Error processing sample {sample_token}: {e}")
        return None


def process_scenes(nusc, scenes: List[Dict], config: Dict, use_parallel: bool = False) -> Dict:
    """
    Process multiple scenes to generate risk labels

    Args:
        nusc: NuScenes instance
        scenes: List of scene dicts
        config: Configuration dict
        use_parallel: Whether to use parallel processing

    Returns:
        Dict of {scene_token: [labels]}
    """
    all_labels = {}

    for scene in tqdm(scenes, desc="Processing scenes"):
        scene_labels = []

        # Get all samples in scene
        sample_tokens = get_scene_samples(nusc, scene['token'])

        if use_parallel:
            # Parallel processing
            with mp.Pool(processes=mp.cpu_count()) as pool:
                process_func = partial(process_sample_wrapper, nusc=nusc, config=config)
                results = pool.map(process_func, sample_tokens)

            # Filter out None results
            scene_labels = [r for r in results if r is not None]
        else:
            # Sequential processing
            for sample_token in tqdm(sample_tokens, desc=f"  {scene['name']}", leave=False):
                try:
                    sample = nusc.get('sample', sample_token)
                    label = generate_risk_map(nusc, sample, config)
                    scene_labels.append(label)
                except Exception as e:
                    print(f"\n⚠️  Error on sample {sample_token}: {e}")
                    continue

        all_labels[scene['token']] = scene_labels

    return all_labels


def print_statistics(train_labels: Dict, val_labels: Dict):
    """
    Print statistics about generated labels

    Args:
        train_labels: Training labels dict
        val_labels: Validation labels dict
    """
    print("\n" + "=" * 80)
    print("LABEL GENERATION STATISTICS")
    print("=" * 80)

    def compute_stats(labels_dict, split_name):
        total_samples = sum(len(scene_labels) for scene_labels in labels_dict.values())
        total_scenes = len(labels_dict)

        all_max_risks = []
        all_mean_risks = []
        all_high_risk_cells = []

        for scene_labels in labels_dict.values():
            for label in scene_labels:
                all_max_risks.append(label['metadata']['max_risk'])
                all_mean_risks.append(label['metadata']['mean_risk'])
                all_high_risk_cells.append(label['metadata']['high_risk_cells'])

        print(f"\n{split_name.upper()} SET:")
        print(f"  Scenes:  {total_scenes}")
        print(f"  Samples: {total_samples}")

        if len(all_max_risks) == 0:
            print(f"  No samples in {split_name} set")
            return

        print(f"\n  Risk Statistics:")
        print(f"    Max risk (avg):  {np.mean(all_max_risks):.3f} ± {np.std(all_max_risks):.3f}")
        print(f"    Mean risk (avg): {np.mean(all_mean_risks):.3f} ± {np.std(all_mean_risks):.3f}")
        print(f"    High-risk cells (avg): {np.mean(all_high_risk_cells):.1f} ± {np.std(all_high_risk_cells):.1f}")
        print(f"\n  Risk Distribution:")
        print(f"    Samples with max_risk > 0.7: {sum(1 for r in all_max_risks if r > 0.7)} ({100*sum(1 for r in all_max_risks if r > 0.7)/len(all_max_risks):.1f}%)")
        print(f"    Samples with max_risk > 0.5: {sum(1 for r in all_max_risks if r > 0.5)} ({100*sum(1 for r in all_max_risks if r > 0.5)/len(all_max_risks):.1f}%)")
        print(f"    Samples with max_risk > 0.3: {sum(1 for r in all_max_risks if r > 0.3)} ({100*sum(1 for r in all_max_risks if r > 0.3)/len(all_max_risks):.1f}%)")

    compute_stats(train_labels, "train")
    compute_stats(val_labels, "val")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Generate BEV Risk Map labels for nuScenes')
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='nuScenes version (default: v1.0-trainval)')
    parser.add_argument('--output_dir', type=str, default='data/emergence_risk',
                        help='Output directory for labels')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel processing (faster but more memory)')
    parser.add_argument('--scenes', type=str, nargs='+', default=None,
                        help='Process specific scenes only (for testing)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("BEV RISK MAP LABEL GENERATION")
    print("=" * 80)
    print(f"Dataroot:    {args.dataroot}")
    print(f"Version:     {args.version}")
    print(f"Output dir:  {output_dir}")
    print(f"Parallel:    {args.parallel}")
    print("=" * 80 + "\n")

    # Load nuScenes
    print("Loading nuScenes dataset...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print(f"✓ Loaded {len(nusc.scene)} scenes\n")

    # Get train/val splits
    if args.scenes:
        # Process specific scenes (for testing)
        selected_scenes = [s for s in nusc.scene if s['name'] in args.scenes]
        train_scenes = selected_scenes
        val_scenes = []
        print(f"Processing {len(selected_scenes)} specific scenes: {args.scenes}\n")
    else:
        # Use standard train/val split
        # nuScenes doesn't have a built-in split, so we'll use a common split
        # Typically, first 700 scenes are train, rest are val
        train_scenes = nusc.scene[:700]
        val_scenes = nusc.scene[700:]
        print(f"Train scenes: {len(train_scenes)}")
        print(f"Val scenes:   {len(val_scenes)}\n")

    # Save configuration
    config_path = output_dir / 'risk_config.json'
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f"✓ Saved configuration to {config_path}\n")

    # Process train set
    if len(train_scenes) > 0:
        print("Processing train set...")
        train_labels = process_scenes(nusc, train_scenes, CONFIG, use_parallel=args.parallel)

        # Save train labels
        train_path = output_dir / 'risk_labels_train.pkl'
        with open(train_path, 'wb') as f:
            pickle.dump(train_labels, f)
        print(f"✓ Saved train labels to {train_path}\n")
    else:
        train_labels = {}

    # Process val set
    if len(val_scenes) > 0:
        print("Processing val set...")
        val_labels = process_scenes(nusc, val_scenes, CONFIG, use_parallel=args.parallel)

        # Save val labels
        val_path = output_dir / 'risk_labels_val.pkl'
        with open(val_path, 'wb') as f:
            pickle.dump(val_labels, f)
        print(f"✓ Saved val labels to {val_path}\n")
    else:
        val_labels = {}

    # Print statistics
    if len(train_labels) > 0 or len(val_labels) > 0:
        print_statistics(train_labels, val_labels)

    print("\n✅ LABEL GENERATION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    if len(train_labels) > 0:
        print(f"  - risk_labels_train.pkl")
    if len(val_labels) > 0:
        print(f"  - risk_labels_val.pkl")
    print(f"  - risk_config.json")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
