"""
V5 Simplified Feature Computation

This module contains the simplified compute_cell_features() function for V5.
It will replace the complex V4 version in risk_utils.py
"""

import numpy as np
from typing import Dict, List


def compute_cell_features_v5(cell_pos: np.ndarray,
                             ego_state: Dict,
                             objects: List[Dict],
                             config: Dict) -> Dict:
    """
    Compute features needed for V5 risk calculation (simplified)

    V5 only needs:
        - is_occluded
        - occluder_area
        - time_to_collision
        - distance_to_trajectory
        - is_on_trajectory
        - is_future

    Args:
        cell_pos: Cell position (x, y) in ego coordinates
        ego_state: Ego vehicle state dict
        objects: List of detected objects
        config: Configuration dict

    Returns:
        Feature dict with V5-required values
    """
    features = {}
    params = config['risk_params']

    # === Ego state ===
    ego_pos = ego_state['position']
    ego_velocity = ego_state['velocity']
    ego_heading = ego_state['heading']
    trajectory = ego_state['trajectory']

    # === 1. Ego Distance ===
    cell_to_ego = cell_pos - ego_pos
    distance = np.linalg.norm(cell_to_ego)
    features['ego_distance'] = distance

    # === 2. Time to Collision (TTC) ===
    if ego_velocity > 0.1:
        # TTC based on distance along heading direction
        heading_vec = np.array([np.cos(ego_heading), np.sin(ego_heading)])
        longitudinal_dist = np.dot(cell_to_ego, heading_vec)

        if longitudinal_dist > 0:  # Ahead of ego
            ttc = longitudinal_dist / ego_velocity
        else:
            ttc = float('inf')  # Behind ego
    else:
        ttc = float('inf')  # Not moving

    features['time_to_collision'] = ttc

    # === 3. Distance to Trajectory ===
    dist_to_traj = compute_distance_to_trajectory(cell_pos, trajectory)
    features['distance_to_trajectory'] = dist_to_traj

    # === 4. Trajectory Indicators ===
    # On trajectory if within d_traj_max
    d_traj_max = params['d_traj_max']
    features['is_on_trajectory'] = (dist_to_traj <= d_traj_max)

    # Future if ahead of ego (temporal position > 0)
    temporal_pos = compute_temporal_position(cell_pos, ego_state)
    features['is_future'] = (temporal_pos > 0)
    features['temporal_position_on_trajectory'] = temporal_pos

    # === 5. Occlusion ===
    occluder = find_occluding_object(cell_pos, ego_state, objects, config)

    if occluder is None:
        features['is_occluded'] = False
        features['occluder_area'] = 0.0
        features['occluder'] = None
    else:
        features['is_occluded'] = True

        # Object area (m²) = width × length
        size = occluder['size']
        width = size[0]
        length = size[1]
        features['occluder_area'] = width * length
        features['occluder'] = occluder

    return features


def compute_temporal_position(cell_pos: np.ndarray, ego_state: Dict) -> float:
    """
    Compute temporal position along ego heading

    Returns:
        temporal_pos: float
            > 0: ahead of ego (future)
            < 0: behind ego (past)
    """
    ego_pos = ego_state['position']
    ego_heading = ego_state['heading']

    # Vector from ego to cell
    to_cell = cell_pos - ego_pos

    # Ego heading vector
    heading_vec = np.array([
        np.cos(ego_heading),
        np.sin(ego_heading)
    ])

    # Project onto heading
    temporal_pos = np.dot(to_cell, heading_vec)

    return temporal_pos


def compute_distance_to_trajectory(cell_pos: np.ndarray, trajectory: np.ndarray) -> float:
    """
    Compute minimum perpendicular distance from cell to trajectory

    Args:
        cell_pos: (x, y)
        trajectory: np.array of (x, y) points, shape (N, 2)

    Returns:
        min_distance: float (meters)
    """
    if trajectory is None or len(trajectory) == 0:
        # No trajectory available
        return float('inf')

    min_dist = float('inf')

    # Handle single point trajectory
    if len(trajectory) == 1:
        return np.linalg.norm(cell_pos - trajectory[0])

    # Check distance to each trajectory segment
    for i in range(len(trajectory) - 1):
        p1 = trajectory[i]
        p2 = trajectory[i + 1]

        # Distance from point to line segment
        dist = point_to_segment_distance(cell_pos, p1, p2)
        min_dist = min(min_dist, dist)

    return min_dist


def point_to_segment_distance(point: np.ndarray,
                              seg_start: np.ndarray,
                              seg_end: np.ndarray) -> float:
    """
    Calculate distance from point to line segment

    Args:
        point: (x, y)
        seg_start: (x, y) - segment start
        seg_end: (x, y) - segment end

    Returns:
        distance: float
    """
    # Vector from seg_start to seg_end
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)

    if seg_len_sq < 1e-6:
        # Segment is essentially a point
        return np.linalg.norm(point - seg_start)

    # Project point onto segment
    point_vec = point - seg_start
    t = np.dot(point_vec, seg_vec) / seg_len_sq
    t = np.clip(t, 0, 1)

    # Closest point on segment
    closest = seg_start + t * seg_vec

    return np.linalg.norm(point - closest)


def find_occluding_object(cell_pos: np.ndarray,
                          ego_state: Dict,
                          objects: List[Dict],
                          config: Dict) -> Dict:
    """
    Find object that occludes this cell from ego's viewpoint

    NOTE: This function should be imported from the original risk_utils.py
    This is just a placeholder signature.

    Returns:
        occluder: dict or None
    """
    # This will use the existing implementation from risk_utils.py
    # Just import it: from risk_utils import find_occluding_object
    pass
