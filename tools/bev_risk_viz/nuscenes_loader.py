#!/usr/bin/env python3
"""
BEV Risk Map Generator - nuScenes Data Loader

Handles loading and processing of nuScenes dataset for risk visualization.
Supports scene selection, frame navigation, and occlusion region extraction.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pickle


@dataclass
class SceneFrame:
    """Data structure for a single scene frame"""
    scene_token: str
    sample_token: str
    timestamp: int
    ego_pose: Dict  # Translation and rotation
    camera_data: Dict  # Camera images and calibration
    annotations: List[Dict]  # Object annotations
    occlusion_mask: Optional[np.ndarray] = None
    occlusion_depth: Optional[np.ndarray] = None


class NuScenesLoader:
    """
    Loader for nuScenes dataset with occlusion-based risk data
    """

    def __init__(
        self,
        data_root: str = 'data/nuscenes',
        version: str = 'v1.0-mini',
        risk_data_path: Optional[str] = None
    ):
        """
        Initialize nuScenes data loader

        Args:
            data_root: Root directory of nuScenes dataset
            version: Dataset version ('v1.0-mini', 'v1.0-trainval', etc.)
            risk_data_path: Optional path to pre-computed risk/occlusion data
        """
        self.data_root = data_root
        self.version = version
        self.risk_data_path = risk_data_path

        # Will be initialized lazily
        self._nusc = None
        self._risk_data = None
        self._scene_list = None

    @property
    def nusc(self):
        """Lazy load nuScenes instance"""
        if self._nusc is None:
            try:
                from nuscenes.nuscenes import NuScenes
                self._nusc = NuScenes(
                    version=self.version,
                    dataroot=self.data_root,
                    verbose=True
                )
                print(f"✓ Loaded nuScenes {self.version}")
            except ImportError:
                raise ImportError(
                    "nuscenes-devkit not installed. "
                    "Install with: pip install nuscenes-devkit"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load nuScenes: {e}")
        return self._nusc

    @property
    def risk_data(self):
        """Lazy load pre-computed risk data"""
        if self._risk_data is None and self.risk_data_path is not None:
            if os.path.exists(self.risk_data_path):
                print(f"Loading risk data from {self.risk_data_path}")
                with open(self.risk_data_path, 'rb') as f:
                    self._risk_data = pickle.load(f)
                print(f"✓ Loaded {len(self._risk_data)} scenes with risk data")
            else:
                print(f"⚠ Risk data not found at {self.risk_data_path}")
                self._risk_data = {}
        return self._risk_data or {}

    def get_scene_list(self) -> List[Dict]:
        """
        Get list of available scenes

        Returns:
            List of scene dictionaries with metadata
        """
        if self._scene_list is None:
            self._scene_list = []
            for scene in self.nusc.scene:
                self._scene_list.append({
                    'token': scene['token'],
                    'name': scene['name'],
                    'description': scene['description'],
                    'num_samples': scene['nbr_samples'],
                    'first_sample': scene['first_sample_token'],
                    'last_sample': scene['last_sample_token'],
                })
        return self._scene_list

    def load_scene_by_name(self, scene_name: str) -> Dict:
        """
        Load scene by name

        Args:
            scene_name: Scene name (e.g., 'scene-0001')

        Returns:
            Scene dictionary
        """
        for scene in self.nusc.scene:
            if scene['name'] == scene_name:
                return scene
        raise ValueError(f"Scene '{scene_name}' not found")

    def load_scene_by_token(self, scene_token: str) -> Dict:
        """
        Load scene by token

        Args:
            scene_token: Scene token

        Returns:
            Scene dictionary
        """
        return self.nusc.get('scene', scene_token)

    def get_scene_frames(
        self,
        scene_token: str,
        max_frames: Optional[int] = None
    ) -> List[str]:
        """
        Get all sample tokens for a scene

        Args:
            scene_token: Scene token
            max_frames: Maximum number of frames to return (None for all)

        Returns:
            List of sample tokens in temporal order
        """
        scene = self.nusc.get('scene', scene_token)
        sample_tokens = []

        current_sample_token = scene['first_sample_token']
        while current_sample_token:
            sample_tokens.append(current_sample_token)
            if max_frames and len(sample_tokens) >= max_frames:
                break

            sample = self.nusc.get('sample', current_sample_token)
            current_sample_token = sample['next']

        return sample_tokens

    def load_frame(
        self,
        sample_token: str,
        load_images: bool = False,
        load_occlusion: bool = True
    ) -> SceneFrame:
        """
        Load a single frame with all associated data

        Args:
            sample_token: Sample token to load
            load_images: Whether to load camera images
            load_occlusion: Whether to load occlusion data (if available)

        Returns:
            SceneFrame object with loaded data
        """
        sample = self.nusc.get('sample', sample_token)

        # Get ego pose
        sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose_record = self.nusc.get('ego_pose', sample_data['ego_pose_token'])

        ego_pose = {
            'translation': ego_pose_record['translation'],
            'rotation': ego_pose_record['rotation'],
        }

        # Get camera data
        camera_data = {}
        camera_names = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if load_images:
            for cam_name in camera_names:
                if cam_name in sample['data']:
                    cam_token = sample['data'][cam_name]
                    cam_sample = self.nusc.get('sample_data', cam_token)
                    calibrated_sensor = self.nusc.get(
                        'calibrated_sensor',
                        cam_sample['calibrated_sensor_token']
                    )

                    camera_data[cam_name] = {
                        'token': cam_token,
                        'filename': os.path.join(self.data_root, cam_sample['filename']),
                        'calibration': {
                            'camera_intrinsic': calibrated_sensor['camera_intrinsic'],
                            'translation': calibrated_sensor['translation'],
                            'rotation': calibrated_sensor['rotation'],
                        }
                    }

        # Get annotations (objects)
        annotations = []
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            annotations.append({
                'token': ann_token,
                'category': ann['category_name'],
                'instance_token': ann['instance_token'],
                'translation': ann['translation'],
                'size': ann['size'],
                'rotation': ann['rotation'],
                'velocity': self.nusc.box_velocity(ann_token),
                'num_lidar_pts': ann['num_lidar_pts'],
                'num_radar_pts': ann['num_radar_pts'],
            })

        # Load occlusion data if available
        occlusion_mask = None
        occlusion_depth = None

        if load_occlusion and self.risk_data:
            scene = self.nusc.get('sample', sample_token)['scene_token']
            scene_risk_data = self.risk_data.get(scene, [])

            # Find matching frame in risk data
            for risk_entry in scene_risk_data:
                if risk_entry.get('sample_token') == sample_token:
                    occlusion_mask = risk_entry.get('occlusion_mask')
                    occlusion_depth = risk_entry.get('occlusion_depth')
                    break

        return SceneFrame(
            scene_token=sample['scene_token'],
            sample_token=sample_token,
            timestamp=sample['timestamp'],
            ego_pose=ego_pose,
            camera_data=camera_data,
            annotations=annotations,
            occlusion_mask=occlusion_mask,
            occlusion_depth=occlusion_depth,
        )

    def create_occlusion_mask_from_objects(
        self,
        annotations: List[Dict],
        bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50),
        bev_resolution: float = 0.5
    ) -> np.ndarray:
        """
        Create occlusion mask from object annotations

        Args:
            annotations: List of object annotations
            bev_range: BEV range (x_min, x_max, y_min, y_max) in meters
            bev_resolution: Grid resolution in meters

        Returns:
            Binary occlusion mask [H, W]
        """
        x_min, x_max, y_min, y_max = bev_range
        width = int((x_max - x_min) / bev_resolution)
        height = int((y_max - y_min) / bev_resolution)

        occlusion_mask = np.zeros((height, width), dtype=np.float32)

        # Create occlusion from large static objects
        occluding_categories = [
            'vehicle.car', 'vehicle.truck', 'vehicle.bus',
            'vehicle.construction', 'vehicle.trailer',
            'static_object.bicycle_rack'
        ]

        for ann in annotations:
            if any(cat in ann['category'] for cat in occluding_categories):
                # Get object position and size
                x, y, _ = ann['translation']
                w, l, h = ann['size']

                # Convert to grid coordinates
                grid_x = int((x - x_min) / bev_resolution)
                grid_y = int((y - y_min) / bev_resolution)

                # Approximate object footprint (simplified as rectangle)
                grid_w = max(1, int(w / bev_resolution))
                grid_l = max(1, int(l / bev_resolution))

                # Mark occluded region (behind the object from ego perspective)
                # Simple implementation: shadow extends behind object
                if 0 <= grid_x < width and 0 <= grid_y < height:
                    # Mark object position
                    y_start = max(0, grid_y - grid_l // 2)
                    y_end = min(height, grid_y + grid_l // 2)
                    x_start = max(0, grid_x - grid_w // 2)
                    x_end = min(width, grid_x + grid_w // 2)

                    occlusion_mask[y_start:y_end, x_start:x_end] = 1.0

                    # Extend shadow behind object (away from ego at origin)
                    if x != 0 or y != 0:
                        shadow_length = int(5.0 / bev_resolution)  # 5 meters
                        angle = np.arctan2(y, x)

                        for i in range(1, shadow_length):
                            shadow_x = grid_x + int(i * np.cos(angle))
                            shadow_y = grid_y + int(i * np.sin(angle))

                            if 0 <= shadow_x < width and 0 <= shadow_y < height:
                                shadow_strength = 1.0 - (i / shadow_length)
                                occlusion_mask[shadow_y, shadow_x] = max(
                                    occlusion_mask[shadow_y, shadow_x],
                                    shadow_strength
                                )

        return occlusion_mask

    def get_ego_velocity(self, sample_token: str) -> Tuple[float, float]:
        """
        Get ego vehicle velocity for a sample

        Args:
            sample_token: Sample token

        Returns:
            Tuple of (speed, heading) where speed is in m/s and heading is in radians
        """
        sample = self.nusc.get('sample', sample_token)

        # Get two consecutive ego poses to compute velocity
        if sample['next']:
            next_sample = self.nusc.get('sample', sample['next'])

            sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            next_sample_data = self.nusc.get('sample_data', next_sample['data']['LIDAR_TOP'])

            ego_pose1 = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
            ego_pose2 = self.nusc.get('ego_pose', next_sample_data['ego_pose_token'])

            # Calculate displacement
            pos1 = np.array(ego_pose1['translation'][:2])  # x, y
            pos2 = np.array(ego_pose2['translation'][:2])

            # Time difference (nuScenes samples at ~2Hz)
            dt = (next_sample['timestamp'] - sample['timestamp']) / 1e6  # Convert to seconds

            if dt > 0:
                velocity_vector = (pos2 - pos1) / dt
                speed = np.linalg.norm(velocity_vector)
                heading = np.arctan2(velocity_vector[1], velocity_vector[0])
                return speed, heading

        return 0.0, 0.0
