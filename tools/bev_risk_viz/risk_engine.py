#!/usr/bin/env python3
"""
BEV Risk Map Generator - Core Risk Calculation Engine

This module implements the occlusion-based emergence risk calculation
with four risk factors: Trajectory Alignment (θ), Occlusion Severity (O),
Temporal Urgency (T), and Proximity (P).

Risk formula: R = f(θ, O, T, P) with configurable weights
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class RiskConfig:
    """Configuration for risk calculation parameters"""
    # Factor weights (α, β, γ, δ)
    weight_trajectory: float = 0.3  # α - Trajectory Alignment weight
    weight_occlusion: float = 0.3   # β - Occlusion Severity weight
    weight_temporal: float = 0.2    # γ - Temporal Urgency weight
    weight_proximity: float = 0.2   # δ - Proximity weight

    # BEV grid configuration
    bev_x_range: Tuple[float, float] = (-50.0, 50.0)  # meters
    bev_y_range: Tuple[float, float] = (-50.0, 50.0)  # meters
    bev_resolution: float = 0.5  # meters per grid cell

    # Ego vehicle parameters
    ego_velocity: float = 10.0  # m/s
    ego_heading: float = 0.0    # radians (0 = pointing up/forward)

    # Risk calculation parameters
    max_proximity_distance: float = 50.0  # meters
    time_horizon: float = 3.0  # seconds for temporal urgency
    occlusion_depth_threshold: float = 2.0  # meters


class RiskCalculationEngine:
    """
    Core engine for calculating occlusion-based emergence risk in BEV space
    """

    def __init__(self, config: RiskConfig = None):
        """
        Initialize the risk calculation engine

        Args:
            config: Risk configuration parameters
        """
        self.config = config or RiskConfig()

        # Calculate BEV grid dimensions
        self.bev_width = int((self.config.bev_x_range[1] - self.config.bev_x_range[0]) /
                            self.config.bev_resolution)
        self.bev_height = int((self.config.bev_y_range[1] - self.config.bev_y_range[0]) /
                             self.config.bev_resolution)

        # Create BEV grid coordinates (in meters)
        self.grid_x, self.grid_y = self._create_bev_grid()

    def _create_bev_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create BEV grid coordinates

        Returns:
            Tuple of (grid_x, grid_y) arrays with shape (bev_height, bev_width)
        """
        x_coords = np.linspace(
            self.config.bev_x_range[0] + self.config.bev_resolution / 2,
            self.config.bev_x_range[1] - self.config.bev_resolution / 2,
            self.bev_width
        )
        y_coords = np.linspace(
            self.config.bev_y_range[0] + self.config.bev_resolution / 2,
            self.config.bev_y_range[1] - self.config.bev_resolution / 2,
            self.bev_height
        )

        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        return grid_x, grid_y

    def calculate_trajectory_alignment(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        ego_velocity: float,
        ego_heading: float
    ) -> np.ndarray:
        """
        Calculate Trajectory Alignment factor (θ)

        Measures how aligned each grid cell is with the ego vehicle's trajectory.
        Higher values indicate cells directly in the vehicle's path.

        Args:
            grid_x: X coordinates of grid cells
            grid_y: Y coordinates of grid cells
            ego_velocity: Ego vehicle velocity (m/s)
            ego_heading: Ego vehicle heading angle (radians)

        Returns:
            Trajectory alignment scores [0, 1] for each grid cell
        """
        # Ego vehicle direction vector
        ego_dir_x = np.cos(ego_heading)
        ego_dir_y = np.sin(ego_heading)

        # Vectors from ego (0,0) to each grid cell
        cell_vectors_x = grid_x
        cell_vectors_y = grid_y

        # Normalize cell vectors
        cell_distances = np.sqrt(cell_vectors_x**2 + cell_vectors_y**2)
        cell_distances = np.maximum(cell_distances, 1e-6)  # Avoid division by zero

        norm_cell_x = cell_vectors_x / cell_distances
        norm_cell_y = cell_vectors_y / cell_distances

        # Dot product gives alignment (-1 to 1)
        alignment = norm_cell_x * ego_dir_x + norm_cell_y * ego_dir_y

        # Map to [0, 1], with forward direction = 1, backward = 0
        theta = np.maximum(alignment, 0)

        # Apply velocity scaling - faster = more weight on trajectory
        velocity_scale = np.tanh(ego_velocity / 10.0)  # Normalize around 10 m/s
        theta = theta * (0.5 + 0.5 * velocity_scale)

        return theta

    def calculate_occlusion_severity(
        self,
        occlusion_mask: np.ndarray,
        occlusion_depth: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate Occlusion Severity factor (O)

        Measures the severity of occlusion in each grid cell.

        Args:
            occlusion_mask: Binary mask of occluded regions [H, W]
            occlusion_depth: Optional depth/thickness of occlusion [H, W]

        Returns:
            Occlusion severity scores [0, 1] for each grid cell
        """
        # Basic occlusion mask
        O = occlusion_mask.astype(np.float32)

        # If depth information is available, scale by depth
        if occlusion_depth is not None:
            # Normalize depth to [0, 1] based on threshold
            depth_normalized = np.minimum(
                occlusion_depth / self.config.occlusion_depth_threshold,
                1.0
            )
            O = O * depth_normalized

        return O

    def calculate_temporal_urgency(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        ego_velocity: float,
        time_horizon: float
    ) -> np.ndarray:
        """
        Calculate Temporal Urgency factor (T)

        Measures how soon the ego vehicle might reach each grid cell.
        Higher values for cells that will be reached sooner.

        Args:
            grid_x: X coordinates of grid cells
            grid_y: Y coordinates of grid cells
            ego_velocity: Ego vehicle velocity (m/s)
            time_horizon: Time horizon for urgency calculation (seconds)

        Returns:
            Temporal urgency scores [0, 1] for each grid cell
        """
        # Calculate distance to each cell
        distances = np.sqrt(grid_x**2 + grid_y**2)

        # Time to reach each cell (assuming constant velocity)
        if ego_velocity < 0.1:  # Vehicle nearly stationary
            return np.zeros_like(distances)

        time_to_reach = distances / (ego_velocity + 1e-6)

        # Urgency is higher for shorter times, decay exponentially
        T = np.exp(-time_to_reach / time_horizon)

        return T

    def calculate_proximity(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        max_distance: float
    ) -> np.ndarray:
        """
        Calculate Proximity factor (P)

        Measures how close each grid cell is to the ego vehicle.

        Args:
            grid_x: X coordinates of grid cells
            grid_y: Y coordinates of grid cells
            max_distance: Maximum distance for normalization

        Returns:
            Proximity scores [0, 1] for each grid cell
        """
        # Calculate distance to each cell
        distances = np.sqrt(grid_x**2 + grid_y**2)

        # Inverse distance, normalized
        P = 1.0 - np.minimum(distances / max_distance, 1.0)

        # Apply exponential decay for smoother falloff
        P = np.exp(-2.0 * distances / max_distance)

        return P

    def calculate_risk_map(
        self,
        occlusion_mask: np.ndarray,
        occlusion_depth: Optional[np.ndarray] = None,
        ego_velocity: Optional[float] = None,
        ego_heading: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate comprehensive risk map with all factors

        Args:
            occlusion_mask: Binary mask of occluded regions [H, W]
            occlusion_depth: Optional depth/thickness of occlusion [H, W]
            ego_velocity: Override ego velocity (uses config default if None)
            ego_heading: Override ego heading (uses config default if None)

        Returns:
            Dictionary containing:
                - 'risk_map': Final risk scores [0, 1] for each grid cell
                - 'theta': Trajectory alignment factor
                - 'O': Occlusion severity factor
                - 'T': Temporal urgency factor
                - 'P': Proximity factor
                - 'risk_breakdown': Individual weighted contributions
        """
        # Use provided values or fall back to config
        velocity = ego_velocity if ego_velocity is not None else self.config.ego_velocity
        heading = ego_heading if ego_heading is not None else self.config.ego_heading

        # Resize occlusion mask if needed
        if occlusion_mask.shape != (self.bev_height, self.bev_width):
            from scipy.ndimage import zoom
            scale_y = self.bev_height / occlusion_mask.shape[0]
            scale_x = self.bev_width / occlusion_mask.shape[1]
            occlusion_mask = zoom(occlusion_mask, (scale_y, scale_x), order=0)

            if occlusion_depth is not None:
                occlusion_depth = zoom(occlusion_depth, (scale_y, scale_x), order=1)

        # Calculate individual risk factors
        theta = self.calculate_trajectory_alignment(
            self.grid_x, self.grid_y, velocity, heading
        )

        O = self.calculate_occlusion_severity(occlusion_mask, occlusion_depth)

        T = self.calculate_temporal_urgency(
            self.grid_x, self.grid_y, velocity, self.config.time_horizon
        )

        P = self.calculate_proximity(
            self.grid_x, self.grid_y, self.config.max_proximity_distance
        )

        # Calculate weighted risk score
        # R = α*θ + β*O + γ*T + δ*P (normalized by sum of weights)
        weights_sum = (self.config.weight_trajectory +
                      self.config.weight_occlusion +
                      self.config.weight_temporal +
                      self.config.weight_proximity)

        risk_map = (
            self.config.weight_trajectory * theta +
            self.config.weight_occlusion * O +
            self.config.weight_temporal * T +
            self.config.weight_proximity * P
        ) / weights_sum

        # Ensure risk is in [0, 1]
        risk_map = np.clip(risk_map, 0.0, 1.0)

        return {
            'risk_map': risk_map,
            'theta': theta,
            'O': O,
            'T': T,
            'P': P,
            'risk_breakdown': {
                'trajectory_contribution': self.config.weight_trajectory * theta / weights_sum,
                'occlusion_contribution': self.config.weight_occlusion * O / weights_sum,
                'temporal_contribution': self.config.weight_temporal * T / weights_sum,
                'proximity_contribution': self.config.weight_proximity * P / weights_sum,
            }
        }

    def update_config(self, **kwargs):
        """
        Update configuration parameters

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

        # Recreate grid if spatial parameters changed
        if any(k in kwargs for k in ['bev_x_range', 'bev_y_range', 'bev_resolution']):
            self.bev_width = int((self.config.bev_x_range[1] - self.config.bev_x_range[0]) /
                                self.config.bev_resolution)
            self.bev_height = int((self.config.bev_y_range[1] - self.config.bev_y_range[0]) /
                                 self.config.bev_resolution)
            self.grid_x, self.grid_y = self._create_bev_grid()
