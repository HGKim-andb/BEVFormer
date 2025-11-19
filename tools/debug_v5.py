#!/usr/bin/env python3
"""Debug V5 risk calculation"""

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

# Get a sample
scene = [s for s in nusc.scene if s['name'] == 'scene-0061'][0]
sample_tokens = []
current = scene['first_sample_token']
while current != '':
    sample_tokens.append(current)
    sample = nusc.get('sample', current)
    current = sample['next']

# Use 10th sample
sample_token = sample_tokens[10]
sample = nusc.get('sample', sample_token)

# Get ego state and objects
ego_state = get_ego_state(nusc, sample)
ego_pose = nusc.get('ego_pose', sample['data']['LIDAR_TOP'])
objects = get_detected_objects(nusc, sample, ego_pose)

print("=" * 80)
print(f"V5 DEBUGGING")
print("=" * 80)
print(f"Sample: {sample_token[:16]}")
print(f"Ego velocity: {ego_state['velocity']:.2f} m/s")
print(f"Ego heading: {ego_state['heading']:.2f} rad")
print(f"Number of objects: {len(objects)}")
print(f"Trajectory points: {len(ego_state['trajectory'])}")
print("=" * 80)

# Test cells
test_cells = [
    (30, 3),     # Forward, behind object
    (10, 0),     # Forward, close
    (5, 0),      # Forward, very close
]

for cell_x, cell_y in test_cells:
    cell_pos = np.array([cell_x, cell_y], dtype=float)

    print(f"\nCell at ({cell_x}, {cell_y}):")

    # Compute features
    features = compute_cell_features(cell_pos, ego_state, objects, CONFIG)

    print(f"  is_occluded: {features['is_occluded']}")
    print(f"  occluder_area: {features['occluder_area']:.2f} m²")
    print(f"  time_to_collision: {features['time_to_collision']:.2f} s")
    print(f"  distance_to_trajectory: {features['distance_to_trajectory']:.2f} m")
    print(f"  is_on_trajectory: {features['is_on_trajectory']}")
    print(f"  is_future: {features['is_future']}")
    print(f"  temporal_position: {features['temporal_position_on_trajectory']:.2f} m")

    # Compute risk
    risk = compute_risk_score(features, CONFIG)
    print(f"  **RISK**: {risk:.4f}")

    # Debug risk computation
    params = CONFIG['risk_params']

    if features['is_on_trajectory'] and features['is_future']:
        if features['is_occluded']:
            # O
            O = min(features['occluder_area'] / params['A_ref'], 1.0)
            print(f"    O = {O:.3f}")

            # U
            ttc = features['time_to_collision']
            T_safe = params['T_safe']
            T_critical = params['T_critical']
            if ttc >= T_safe:
                U = 0.0
            elif ttc <= T_critical:
                U = 1.0
            else:
                U = (T_safe - ttc) / (T_safe - T_critical)
            print(f"    U = {U:.3f} (TTC = {ttc:.2f}s)")

            # P
            d_traj = features['distance_to_trajectory']
            d_close = params['d_close']
            d_far = params['d_far']
            if d_traj <= d_close:
                P = 1.0
            elif d_traj >= d_far:
                P = 0.0
            else:
                P = (d_far - d_traj) / (d_far - d_close)
            print(f"    P = {P:.3f} (dist = {d_traj:.2f}m)")

            print(f"    O × U × P = {O:.3f} × {U:.3f} × {P:.3f} = {O*U*P:.4f}")
        else:
            print(f"    Not occluded!")
    else:
        print(f"    Filtered out (on_traj={features['is_on_trajectory']}, future={features['is_future']})")

print("\n" + "=" * 80)
