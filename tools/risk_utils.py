#!/usr/bin/env python3
"""
Risk Map Utility Functions

This module contains all utility functions for generating BEV Risk Maps
for occluded regions where objects might suddenly appear.
"""

import numpy as np
from pyquaternion import Quaternion
from typing import Dict, List, Tuple, Optional
import copy


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Version
    'version': 'v5_continuous_function',  # Risk calculation version

    # BEV Grid
    'bev_range': [-50, 50, -50, 50],  # [x_min, x_max, y_min, y_max] meters
    'bev_resolution': 0.5,             # meters per pixel
    'bev_h': 200,
    'bev_w': 200,

    # Distance thresholds (TTC based)
    'far_distance': 30.0,      # > 30m: low urgency
    'medium_distance': 15.0,   # 15-30m: medium urgency
    'close_distance': 8.0,     # 8-15m: high urgency

    # Lateral (좌우) thresholds
    'same_lane_threshold': 1.5,      # < 1.5m: same lane
    'adjacent_lane_threshold': 5.0,  # 1.5-5m: adjacent lanes
    'lane_width': 3.5,                # standard lane width

    # Occlusion
    'min_occluder_area': 1.0,  # m², minimum size to occlude

    # Collision course
    'collision_distance_threshold': 2.0,  # meters
    'prediction_horizon': 3.0,            # seconds

    # Risk calculation parameters (v5 continuous function method)
    'risk_params': {
        # Trajectory corridor (좌우 거리 필터)
        'd_traj_max': 20.0,      # meters, trajectory corridor width (15→20, 좌우 영향 감소)

        # Occlusion
        'A_ref': 10.0,          # m², reference area for occlusion normalization

        # Urgency (TTC-based) - 조정: 더 긴급하게
        'T_safe': 10.0,          # seconds, safe reaction time (3→10으로 증가)
        'T_critical': 2.0,      # seconds, critical reaction time (1→2로 증가)

        # Proximity (distance to trajectory) - 좌우 거리 영향 감소
        'd_close': 5.0,         # meters, on trajectory threshold (변경 없음)
        'd_far': 20.0,           # meters, off trajectory threshold (15→20, 좌우 영향 감소)
    },
}

# Occlusion profiles by object type
OCCLUSION_PROFILES = {
    'vehicle.bus': 1.0,
    'vehicle.truck': 0.9,
    'vehicle.trailer': 0.85,
    'vehicle.construction': 0.9,
    'vehicle.car': 0.6,
    'vehicle.motorcycle': 0.3,
    'vehicle.bicycle': 0.2,
    'human.pedestrian.adult': 0.3,
    'human.pedestrian.child': 0.1,
    'human.pedestrian.construction_worker': 0.3,
    'human.pedestrian.police_officer': 0.3,
}


# ============================================================================
# Coordinate Transformations
# ============================================================================

def grid_to_world(grid_x: int, grid_y: int, config: Dict) -> Tuple[float, float]:
    """
    Convert grid coordinates to world coordinates (ego-centric)

    Args:
        grid_x: Grid x index (0 to bev_w-1)
        grid_y: Grid y index (0 to bev_h-1)
        config: Configuration dict

    Returns:
        (x, y) in world coordinates (meters)
    """
    x_min, x_max, y_min, y_max = config['bev_range']
    resolution = config['bev_resolution']

    # Convert grid indices to world coordinates
    # grid (0,0) corresponds to (x_min, y_min)
    world_x = x_min + grid_x * resolution
    world_y = y_min + grid_y * resolution

    return world_x, world_y


def world_to_grid(world_x: float, world_y: float, config: Dict) -> Tuple[int, int]:
    """
    Convert world coordinates to grid coordinates

    Args:
        world_x: World x coordinate (meters)
        world_y: World y coordinate (meters)
        config: Configuration dict

    Returns:
        (grid_x, grid_y) or None if out of bounds
    """
    x_min, x_max, y_min, y_max = config['bev_range']
    resolution = config['bev_resolution']

    grid_x = int((world_x - x_min) / resolution)
    grid_y = int((world_y - y_min) / resolution)

    # Check bounds
    if 0 <= grid_x < config['bev_w'] and 0 <= grid_y < config['bev_h']:
        return grid_x, grid_y
    else:
        return None


# ============================================================================
# Ego State Extraction
# ============================================================================

def get_ego_state(nusc, sample: Dict) -> Dict:
    """
    Extract ego vehicle state from sample

    Args:
        nusc: NuScenes instance
        sample: Sample dict

    Returns:
        {
            'position': (x, y),
            'velocity': float,  # m/s
            'heading': float,   # radians
            'trajectory': np.array  # predicted path points
        }
    """
    # Get ego pose
    ego_pose = nusc.get('ego_pose', sample['data']['LIDAR_TOP'])

    # In ego-relative frame, ego is always at origin
    position = np.array([0.0, 0.0])  # Ego position in ego frame is always (0, 0)

    # Get rotation (quaternion -> heading)
    # In ego frame, heading is 0 (forward is +X direction)
    heading = 0.0  # Ego heading in ego frame is always 0

    # Calculate velocity from previous sample (in global frame, then convert to scalar)
    velocity = 0.0
    trajectory = np.array([position])  # Start with current position

    if sample['prev'] != '':
        try:
            prev_sample = nusc.get('sample', sample['prev'])
            prev_ego_pose = nusc.get('ego_pose', prev_sample['data']['LIDAR_TOP'])

            curr_position_global = np.array(ego_pose['translation'][:2])
            prev_position_global = np.array(prev_ego_pose['translation'][:2])

            # Time delta (samples are 0.5s apart in nuScenes)
            time_delta = 0.5

            # Velocity (scalar, in global frame)
            delta_position = curr_position_global - prev_position_global
            velocity = np.linalg.norm(delta_position) / time_delta
        except:
            velocity = 0.0

    # Predict trajectory (simple linear prediction along forward direction)
    # In ego frame, forward is +X direction (heading = 0)
    if velocity > 0.1:  # Only if moving
        # Forward direction in ego frame is [1, 0]
        heading_vec = np.array([1.0, 0.0])
        num_points = 10
        trajectory_points = []
        for i in range(num_points):
            t = (i + 1) * CONFIG['prediction_horizon'] / num_points
            future_pos = position + velocity * heading_vec * t
            trajectory_points.append(future_pos)
        trajectory = np.array(trajectory_points)

    return {
        'position': position,
        'velocity': velocity,
        'heading': heading,
        'trajectory': trajectory,
    }


# ============================================================================
# Object Detection Extraction
# ============================================================================

def get_detected_objects(nusc, sample: Dict, ego_pose: Dict) -> List[Dict]:
    """
    Extract all detected objects from sample in ego coordinates

    Args:
        nusc: NuScenes instance
        sample: Sample dict
        ego_pose: Ego pose dict (from nusc.get('ego_pose', ...))

    Returns:
        List of object dicts with:
            'position': (x, y, z) in ego frame
            'size': (width, length, height)
            'class': category name
            'velocity': (vx, vy) or None
            'rotation': yaw angle
    """
    objects = []

    # Get ego transformation
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])

    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)

        # Object position in world frame
        obj_translation_world = np.array(ann['translation'])

        # Transform to ego frame
        obj_translation_ego = obj_translation_world - ego_translation
        obj_translation_ego = ego_rotation.inverse.rotate(obj_translation_ego)

        # Object rotation
        obj_rotation_world = Quaternion(ann['rotation'])
        obj_rotation_ego = ego_rotation.inverse * obj_rotation_world
        obj_yaw = obj_rotation_ego.yaw_pitch_roll[0]

        # Size (width, length, height)
        size = ann['size']  # [width, length, height]

        # Category
        category = ann['category_name']

        # Velocity (optional)
        velocity = None
        try:
            velocity_world = nusc.box_velocity(ann_token)
            if not np.isnan(velocity_world).any():
                # Transform velocity to ego frame
                velocity_ego = ego_rotation.inverse.rotate(velocity_world)
                velocity = velocity_ego[:2]  # (vx, vy)
        except:
            pass

        objects.append({
            'position': obj_translation_ego,
            'size': size,
            'class': category,
            'velocity': velocity,
            'rotation': obj_yaw,
        })

    return objects


# ============================================================================
# Occlusion Detection (Ray Casting)
# ============================================================================

def find_occluding_object(cell_pos: np.ndarray,
                          ego_state: Dict,
                          objects: List[Dict],
                          config: Dict) -> Optional[Dict]:
    """
    Find if cell is occluded by any object using ray casting

    Args:
        cell_pos: (x, y) position in ego coordinates
        ego_state: Ego vehicle state dict
        objects: List of detected objects
        config: Configuration dict

    Returns:
        Occluding object dict or None
    """
    ego_pos = ego_state['position']

    # Ray from ego to cell
    ray_direction = cell_pos - ego_pos
    cell_distance = np.linalg.norm(ray_direction)

    if cell_distance < 0.1:
        return None

    ray_direction = ray_direction / cell_distance

    # Check each object
    closest_occluder = None
    closest_distance = float('inf')

    for obj in objects:
        obj_pos = obj['position'][:2]  # (x, y)
        obj_distance = np.linalg.norm(obj_pos - ego_pos)

        # Object must be closer than cell
        if obj_distance >= cell_distance - 1.0:
            continue

        # Object must be large enough to occlude
        obj_area = obj['size'][0] * obj['size'][1]
        if obj_area < config['min_occluder_area']:
            continue

        # Check if ray intersects object bounding box
        # Simple method: check if object center is near the ray
        ego_to_obj = obj_pos - ego_pos
        projection = np.dot(ego_to_obj, ray_direction)

        if projection < 0:
            continue

        # Perpendicular distance from ray to object center
        perpendicular = ego_to_obj - projection * ray_direction
        perp_distance = np.linalg.norm(perpendicular)

        # Use object size to determine if ray hits
        max_dim = max(obj['size'][0], obj['size'][1])
        if perp_distance < max_dim / 2:
            # Ray hits this object
            if obj_distance < closest_distance:
                closest_distance = obj_distance
                closest_occluder = obj

    return closest_occluder


# ============================================================================
# Lateral Distance Computation
# ============================================================================

def compute_lateral_distance(ego_state: Dict,
                             obj_position: np.ndarray) -> Tuple[float, int]:
    """
    Compute lateral (perpendicular) distance from ego trajectory

    Args:
        ego_state: Ego vehicle state dict
        obj_position: Object position (x, y)

    Returns:
        (lateral_distance, lateral_side)
        lateral_distance: Absolute perpendicular distance
        lateral_side: -1 (left) or 1 (right)
    """
    ego_pos = ego_state['position']
    heading = ego_state['heading']

    # Heading vector (forward direction)
    heading_vec = np.array([np.cos(heading), np.sin(heading)])

    # Perpendicular vector (left is positive in standard orientation)
    perp_vec = np.array([-np.sin(heading), np.cos(heading)])

    # Vector from ego to object
    ego_to_obj = obj_position - ego_pos

    # Project onto perpendicular vector
    lateral_dist = np.dot(ego_to_obj, perp_vec)

    # Determine side
    lateral_side = -1 if lateral_dist > 0 else 1  # left: -1, right: 1

    return abs(lateral_dist), lateral_side


# ============================================================================
# Collision Course Detection
# ============================================================================

def check_collision_course(ego_state: Dict,
                           obj_state: Dict,
                           horizon: float = 3.0,
                           threshold: float = 2.0) -> Tuple[bool, float]:
    """
    Check if ego and object are on collision course

    Args:
        ego_state: Ego vehicle state dict
        obj_state: Object state dict
        horizon: Prediction horizon in seconds
        threshold: Distance threshold for collision (meters)

    Returns:
        (is_collision, min_distance)
    """
    ego_pos = ego_state['position']
    ego_vel = ego_state['velocity']
    heading = ego_state['heading']
    heading_vec = np.array([np.cos(heading), np.sin(heading)])

    obj_pos = obj_state['position'][:2]
    obj_vel = obj_state.get('velocity')

    # Ego future position
    ego_future = ego_pos + ego_vel * heading_vec * horizon

    # Object future position
    if obj_vel is not None and not np.isnan(obj_vel).any():
        obj_future = obj_pos + obj_vel * horizon
    else:
        obj_future = obj_pos  # Assume stationary

    # Compute minimum distance between line segments
    # Ego path: (ego_pos, ego_future)
    # Obj path: (obj_pos, obj_future)
    min_dist = minimum_distance_between_segments(
        ego_pos, ego_future,
        obj_pos, obj_future
    )

    is_collision = min_dist < threshold

    return is_collision, min_dist


def minimum_distance_between_segments(p1: np.ndarray, p2: np.ndarray,
                                      p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Compute minimum distance between two line segments

    Args:
        p1, p2: Endpoints of first segment
        p3, p4: Endpoints of second segment

    Returns:
        Minimum distance
    """
    # Direction vectors
    d1 = p2 - p1
    d2 = p4 - p3

    # Check for point-to-point distances
    distances = [
        np.linalg.norm(p1 - p3),
        np.linalg.norm(p1 - p4),
        np.linalg.norm(p2 - p3),
        np.linalg.norm(p2 - p4),
    ]

    # Point-to-segment distances
    distances.extend([
        point_to_segment_distance(p1, p3, p4),
        point_to_segment_distance(p2, p3, p4),
        point_to_segment_distance(p3, p1, p2),
        point_to_segment_distance(p4, p1, p2),
    ])

    return min(distances)


def point_to_segment_distance(p: np.ndarray,
                              a: np.ndarray,
                              b: np.ndarray) -> float:
    """
    Distance from point p to line segment ab
    """
    ab = b - a
    ap = p - a

    ab_len_sq = np.dot(ab, ab)

    if ab_len_sq < 1e-6:
        return np.linalg.norm(ap)

    t = np.clip(np.dot(ap, ab) / ab_len_sq, 0, 1)
    projection = a + t * ab

    return np.linalg.norm(p - projection)


# ============================================================================
# Occlusion Strength & Type Diversity
# ============================================================================

def compute_occlusion_strength(occluder: Dict) -> float:
    """
    Compute occlusion strength based on object type and size

    Args:
        occluder: Object dict

    Returns:
        Strength value (0~1)
    """
    obj_class = occluder['class']

    # Get base strength from profile
    base_strength = 0.5  # Default
    for key, value in OCCLUSION_PROFILES.items():
        if key in obj_class:
            base_strength = value
            break

    # Adjust by actual size
    width, length, height = occluder['size']
    area = width * length

    # Expected areas by type (rough estimates)
    expected_areas = {
        'vehicle.bus': 12.0,
        'vehicle.truck': 10.0,
        'vehicle.car': 4.0,
        'vehicle.motorcycle': 2.0,
        'human.pedestrian': 0.5,
    }

    expected_area = 4.0  # Default
    for key, value in expected_areas.items():
        if key in obj_class:
            expected_area = value
            break

    # Size adjustment factor (up to 1.5x)
    size_factor = min(area / expected_area, 1.5)

    strength = base_strength * size_factor

    return np.clip(strength, 0.0, 1.0)


def compute_type_diversity(occluder: Dict) -> float:
    """
    Compute diversity of objects that could emerge from occlusion

    Args:
        occluder: Object dict

    Returns:
        Diversity value (0~1)
    """
    width, length, height = occluder['size']
    area = width * length

    if area > 8.0:  # Large (bus, truck)
        diversity = 1.0  # Can hide vehicles, pedestrians, bicycles
    elif area > 3.0:  # Medium (car)
        diversity = 0.7  # Can hide pedestrians, bicycles, motorcycles
    elif area > 1.0:  # Small
        diversity = 0.4  # Can hide pedestrians, bicycles
    else:  # Very small
        diversity = 0.1  # Hardly occludes anything

    return diversity


# ============================================================================
# Longitudinal Urgency
# ============================================================================

def compute_longitudinal_urgency(ttc: float,
                                 distance: float,
                                 lateral_offset: Optional[float] = None,
                                 config: Dict = CONFIG) -> float:
    """
    Compute urgency based on distance and time-to-collision

    Args:
        ttc: Time to collision (seconds)
        distance: Distance to cell (meters)
        lateral_offset: How much ego has passed the occluder (meters)
        config: Configuration dict

    Returns:
        Urgency value (0~1)
    """
    if ttc > 3.0 or distance > config['far_distance']:
        # Far, plenty of time
        return 0.4

    elif ttc > 1.5 or distance > config['medium_distance']:
        # Medium distance, caution needed
        return 0.7

    elif ttc > 0.8 or distance > config['close_distance']:
        # Close, dangerous
        return 0.9

    else:  # Very close (< 8m)
        # Use lateral offset to determine if already visible
        if lateral_offset is None:
            return 1.0

        if lateral_offset < 3.0:
            # Haven't passed yet - critical!
            return 1.0
        else:
            # Already passed - likely visible
            return 0.2


# ============================================================================
# Lateral Risk
# ============================================================================

def compute_lateral_risk(lateral_dist: float,
                        relative_velocity: float,
                        is_collision_course: bool,
                        config: Dict = CONFIG) -> float:
    """
    Compute risk based on lateral distance

    Args:
        lateral_dist: Lateral distance (meters)
        relative_velocity: Relative velocity (m/s)
        is_collision_course: Whether on collision course
        config: Configuration dict

    Returns:
        Risk value (0~1)
    """
    if lateral_dist < config['same_lane_threshold']:
        # Same lane
        if relative_velocity < 2.0:
            # Moving together
            risk = 0.3
        else:
            # Speed difference
            risk = 0.6

    elif lateral_dist < config['adjacent_lane_threshold']:
        # Adjacent lanes
        num_lanes = int(lateral_dist / config['lane_width'])

        if num_lanes == 1 or lateral_dist < 3.5:
            # Immediately adjacent lane
            risk = 0.9
            if is_collision_course:
                risk = 1.0  # Boost!
        else:
            # 2 lanes away
            risk = 0.5

    else:
        # Far away
        risk = 0.2 * np.exp(-(lateral_dist - 5.0) / 5.0)

    return np.clip(risk, 0.0, 1.0)


# ============================================================================
# Feature Computation
# ============================================================================

def compute_temporal_position_on_trajectory(cell_pos: np.ndarray,
                                           ego_state: Dict) -> float:
    """
    Compute cell's longitudinal position relative to ego vehicle

    This determines if a cell is on the past trajectory (behind ego)
    or future trajectory (ahead of ego).

    Args:
        cell_pos: Cell position (x, y) in ego coordinates
        ego_state: Ego vehicle state dict

    Returns:
        temporal_pos: Longitudinal distance along ego heading
                     < 0 for behind (past trajectory)
                     > 0 for ahead (future trajectory)
    """
    ego_pos = ego_state['position']
    ego_heading = ego_state['heading']

    # Vector from ego to cell
    to_cell = cell_pos - ego_pos

    # Heading vector (forward direction)
    heading_vec = np.array([np.cos(ego_heading), np.sin(ego_heading)])

    # Project onto heading (longitudinal component)
    temporal_pos = np.dot(to_cell, heading_vec)

    return temporal_pos


def compute_distance_to_trajectory_v5(cell_pos: np.ndarray,
                                      trajectory: np.ndarray,
                                      ego_state: Dict) -> float:
    """
    Compute minimum perpendicular distance from cell to trajectory

    Args:
        cell_pos: (x, y)
        trajectory: np.array of (x, y) points, shape (N, 2)
        ego_state: ego state dict

    Returns:
        min_distance: float (meters)
    """
    if trajectory is None or len(trajectory) == 0:
        # No trajectory available, use perpendicular distance to heading
        ego_pos = ego_state['position']
        ego_heading = ego_state['heading']

        ego_to_cell = cell_pos - ego_pos
        # Perpendicular distance to heading line
        dist_to_traj = abs(np.dot(ego_to_cell,
                                  np.array([-np.sin(ego_heading), np.cos(ego_heading)])))
        return dist_to_traj

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


def compute_cell_features(cell_pos: np.ndarray,
                         ego_state: Dict,
                         objects: List[Dict],
                         config: Dict = CONFIG) -> Dict:
    """
    Compute features for V5 risk calculation (simplified)

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
    trajectory = ego_state.get('trajectory', np.array([]))

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
    dist_to_traj = compute_distance_to_trajectory_v5(cell_pos, trajectory, ego_state)
    features['distance_to_trajectory'] = dist_to_traj

    # === 4. Trajectory Indicators ===
    # On trajectory if within d_traj_max
    d_traj_max = params['d_traj_max']
    features['is_on_trajectory'] = (dist_to_traj <= d_traj_max)

    # Future if ahead of ego (temporal position > 0)
    temporal_pos = compute_temporal_position_on_trajectory(cell_pos, ego_state)
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


# ============================================================================
# Risk Score Computation
# ============================================================================

def compute_risk_score(features: Dict, config: Dict = CONFIG) -> float:
    """
    Compute emergence risk using continuous functions (V5)

    Formula:
        R = I_traj × O × U × P

    where:
        I_traj: trajectory indicator (0 or 1) - hard filter
        O: occlusion factor (0~1) - continuous, linear
        U: urgency factor (0~1) - continuous, linear, TTC-based
        P: proximity weight (0~1) - continuous, linear

    Args:
        features: dict with keys:
            - is_occluded: bool
            - occluder_area: float (m²)
            - time_to_collision: float (s)
            - distance_to_trajectory: float (m)
            - is_on_trajectory: bool
            - is_future: bool (temporal_position > 0)
        config: configuration dict

    Returns:
        risk: float [0, 1]
    """
    params = config['risk_params']

    # ========================================
    # 1. Trajectory Indicator (Hard Filter)
    # ========================================
    # Only consider cells on/near trajectory AND in future
    if not (features.get('is_on_trajectory', False) and features.get('is_future', False)):
        return 0.0

    # ========================================
    # 2. Occlusion Factor (Continuous, Linear)
    # ========================================
    if not features.get('is_occluded', False):
        return 0.0

    # O = min(A_obj / A_ref, 1.0)
    A_ref = params['A_ref']
    occluder_area = features.get('occluder_area', 0.0)
    O = min(occluder_area / A_ref, 1.0)

    # ========================================
    # 3. Urgency Factor (Continuous, Linear)
    # ========================================
    # U = max(0, min(1, (T_safe - ttc) / (T_safe - T_critical)))

    ttc = features.get('time_to_collision', float('inf'))
    T_safe = params['T_safe']
    T_critical = params['T_critical']

    if ttc >= T_safe:
        U = 0.0
    elif ttc <= T_critical:
        U = 1.0
    else:
        # Linear interpolation
        U = (T_safe - ttc) / (T_safe - T_critical)

    # ========================================
    # 4. Proximity Weight (Continuous, Linear)
    # ========================================
    # P = max(0, min(1, (d_far - d_traj) / (d_far - d_close)))

    d_traj = features.get('distance_to_trajectory', float('inf'))
    d_close = params['d_close']
    d_far = params['d_far']

    if d_traj <= d_close:
        P = 1.0
    elif d_traj >= d_far:
        P = 0.0
    else:
        # Linear interpolation
        P = (d_far - d_traj) / (d_far - d_close)

    # ========================================
    # 5. Final Risk (Multiplicative)
    # ========================================
    risk = O * U * P

    return np.clip(risk, 0.0, 1.0)
