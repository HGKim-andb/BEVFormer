#!/usr/bin/env python3
"""Debug temporal trajectory feature"""

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
print(f"TEMPORAL TRAJECTORY DEBUGGING")
print("=" * 80)
print(f"Sample: {sample_token[:16]}")
print(f"Ego velocity: {ego_state['velocity']:.2f} m/s")
print(f"Ego heading: {ego_state['heading']:.2f} rad")
print("=" * 80)

# Test cells along the ego's heading direction
# Heading is 0.0 rad = pointing along +X axis
# So positive X = ahead, negative X = behind
heading = ego_state['heading']
ego_x, ego_y = ego_state['position']

# Create test cells: behind (-), at ego (0), and ahead (+)
test_positions = [
    ("Behind 20m", ego_x - 20, ego_y),
    ("Behind 10m", ego_x - 10, ego_y),
    ("Behind 5m", ego_x - 5, ego_y),
    ("At ego", ego_x, ego_y),
    ("Ahead 5m", ego_x + 5, ego_y),
    ("Ahead 10m", ego_x + 10, ego_y),
    ("Ahead 20m", ego_x + 20, ego_y),
]

print("\nTEMPORAL POSITION TEST (on trajectory):")
print("-" * 80)
print(f"{'Position':<15} {'Temporal':>12} {'Base Prox':>12} {'Weight':>10} {'Final Prox':>12} {'Risk':>10}")
print("-" * 80)

for label, cell_x, cell_y in test_positions:
    cell_pos = np.array([cell_x, cell_y])
    features = compute_cell_features(cell_pos, ego_state, objects, CONFIG)
    risk = compute_risk_score(features)

    temporal_pos = features.get('temporal_position_on_trajectory', 0.0)
    dist_to_traj = features['distance_to_trajectory']

    # Calculate what the base proximity would be
    if dist_to_traj < 2.0:
        base_proximity = 0.05
    elif dist_to_traj < 5.0:
        base_proximity = 0.05 * (5.0 - dist_to_traj) / 3.0
    elif dist_to_traj < 10.0:
        base_proximity = 0.03 * (10.0 - dist_to_traj) / 5.0
    else:
        base_proximity = 0.0

    # Calculate temporal weight
    if temporal_pos < 0:
        decay = np.exp(temporal_pos / 5.0)
        temporal_weight = decay * 0.2
    else:
        temporal_weight = 1.0

    final_proximity = base_proximity * temporal_weight

    print(f"{label:<15} {temporal_pos:>11.1f}m {base_proximity:>12.4f} {temporal_weight:>9.2f}x {final_proximity:>12.4f} {risk:>10.4f}")

print("\n" + "=" * 80)
print("INTERPRETATION:")
print("=" * 80)
print("Behind cells (temporal_pos < 0):")
print("  - Already passed by ego vehicle")
print("  - Apply exponential decay: weight = exp(pos/5) × 0.2")
print("  - Result: Very low proximity score (~0-20% of base)")
print()
print("Ahead cells (temporal_pos > 0):")
print("  - Future trajectory")
print("  - Apply full weight = 1.0")
print("  - Result: Full proximity score (100% of base)")
print("=" * 80)
