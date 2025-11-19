#!/usr/bin/env python3
"""Debug risk calculation for a single sample"""

import sys
sys.path.insert(0, 'tools')
import numpy as np
from nuscenes.nuscenes import NuScenes
from risk_utils import (
    get_ego_state,
    get_detected_objects,
    compute_cell_features,
    compute_risk_score,
    CONFIG
)

# Load nuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='/home/hg-main/data2/datasets/nuscenes/data/nuscenes', verbose=False)

# Get a sample with moving vehicle
scene = [s for s in nusc.scene if s['name'] == 'scene-0061'][0]
sample_tokens = []
current = scene['first_sample_token']
while current != '':
    sample_tokens.append(current)
    sample = nusc.get('sample', current)
    current = sample['next']

# Use 10th sample (should have some velocity)
sample_token = sample_tokens[10]
sample = nusc.get('sample', sample_token)

# Get ego state and objects
ego_state = get_ego_state(nusc, sample)
ego_pose = nusc.get('ego_pose', sample['data']['LIDAR_TOP'])
objects = get_detected_objects(nusc, sample, ego_pose)

print("=" * 80)
print(f"Sample: {sample_token[:16]}")
print(f"Ego velocity: {ego_state['velocity']:.2f} m/s")
print(f"Ego heading: {ego_state['heading']:.2f} rad")
print(f"Number of objects: {len(objects)}")
print("=" * 80)

# Find large vehicles and test cells behind them
# Object 8: vehicle.truck pos=( 46.82, 26.44) - too far
# Object 3: vehicle.car pos=( 23.97,-40.09) - to the right
# Let's test cells that would be:
# 1. Behind a vehicle
# 2. In the direction ego is heading (forward, x+)
# 3. Close enough for urgency
test_cells = [
    (30, 3),     # Behind object 7 (pedestrian at 31.60, 2.58)
    (48, 27),    # Behind truck at (46.82, 26.44)
    (25, -35),   # Behind car at (23.97, -40.09)
    (35, 0),     # Forward, might be behind pedestrian at (34.82, -2.32)
]

for cell_x, cell_y in test_cells:
    cell_pos = np.array([cell_x, cell_y])

    print(f"\nCell at ({cell_x}, {cell_y}):")
    print(f"  Distance from ego: {np.linalg.norm(cell_pos - ego_state['position']):.2f}m")

    # Compute features
    features = compute_cell_features(cell_pos, ego_state, objects, CONFIG)

    print(f"  Occluded: {features['is_occluded']}")
    if features['occluder']:
        occ = features['occluder']
        occ_x, occ_y = occ['position'][:2]
        print(f"    Occluder: {occ['class']}")
        print(f"    Occluder pos: ({occ_x:.2f}, {occ_y:.2f})")
        print(f"    Occluder size: {occ['size']}")
        print(f"  Occlusion strength: {features['occlusion_strength']:.3f}")
        print(f"  Type diversity: {features['type_diversity']:.3f}")

    print(f"  Longitudinal urgency: {features['longitudinal_urgency']:.3f}")
    print(f"  Lateral risk: {features['lateral_risk']:.3f}")
    print(f"  Ego alignment: {features['ego_alignment']:.3f}")
    print(f"  Trajectory factor: {features['trajectory_factor']:.3f}")

    # Compute risk
    risk = compute_risk_score(features)
    print(f"  **FINAL RISK**: {risk:.4f}")

print("\n" + "=" * 80)
print("Objects summary:")
print("=" * 80)
for i, obj in enumerate(objects[:10]):
    x, y, z = obj['position']
    dist = np.sqrt(x**2 + y**2)
    area = obj['size'][0] * obj['size'][1]
    print(f"{i+1:2d}. {obj['class'][:25]:25s} pos=({x:6.2f},{y:6.2f}) dist={dist:5.1f}m area={area:4.1f}m²")
