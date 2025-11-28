#!/usr/bin/env python3
"""
BEV Risk Map Generator - Configuration Loader

Handles loading and validation of YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


class ConfigLoader:
    """
    Loader and validator for YAML configuration files
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader

        Args:
            config_path: Path to YAML configuration file
                        (uses default if None)
        """
        if config_path is None:
            # Use default config in same directory
            config_path = Path(__file__).parent / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file

        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dots)

        Args:
            key: Configuration key (e.g., 'bev_grid.resolution')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_risk_config(self):
        """
        Get risk calculation configuration as RiskConfig object

        Returns:
            RiskConfig instance
        """
        try:
            from risk_engine import RiskConfig
        except ImportError:
            from tools.bev_risk_viz.risk_engine import RiskConfig

        weights = self.config['risk_weights']
        bev_grid = self.config['bev_grid']
        ego = self.config['ego_vehicle']
        params = self.config['risk_parameters']

        return RiskConfig(
            weight_trajectory=weights['trajectory_alignment'],
            weight_occlusion=weights['occlusion_severity'],
            weight_temporal=weights['temporal_urgency'],
            weight_proximity=weights['proximity'],
            bev_x_range=(bev_grid['x_range']['min'], bev_grid['x_range']['max']),
            bev_y_range=(bev_grid['y_range']['min'], bev_grid['y_range']['max']),
            bev_resolution=bev_grid['resolution'],
            ego_velocity=ego['velocity'],
            ego_heading=ego['heading'],
            max_proximity_distance=params['max_proximity_distance'],
            time_horizon=params['time_horizon'],
            occlusion_depth_threshold=params['occlusion_depth_threshold']
        )

    def update(self, key: str, value: Any):
        """
        Update configuration value

        Args:
            key: Configuration key (supports nested keys with dots)
            value: New value
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, output_path: Optional[str] = None):
        """
        Save configuration to YAML file

        Args:
            output_path: Output file path (overwrites source if None)
        """
        if output_path is None:
            output_path = self.config_path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Configuration saved to {output_path}")

    def validate(self) -> bool:
        """
        Validate configuration values

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate risk weights sum to a reasonable value
        weights = self.config['risk_weights']
        weight_sum = (
            weights['trajectory_alignment'] +
            weights['occlusion_severity'] +
            weights['temporal_urgency'] +
            weights['proximity']
        )

        if weight_sum <= 0:
            raise ValueError("Sum of risk weights must be positive")

        # Validate BEV grid
        bev_grid = self.config['bev_grid']
        if bev_grid['x_range']['min'] >= bev_grid['x_range']['max']:
            raise ValueError("Invalid X range: min must be less than max")
        if bev_grid['y_range']['min'] >= bev_grid['y_range']['max']:
            raise ValueError("Invalid Y range: min must be less than max")
        if bev_grid['resolution'] <= 0:
            raise ValueError("BEV resolution must be positive")

        # Validate ego vehicle parameters
        ego = self.config['ego_vehicle']
        if ego['velocity'] < 0:
            raise ValueError("Ego velocity cannot be negative")

        # Validate risk parameters
        params = self.config['risk_parameters']
        if params['max_proximity_distance'] <= 0:
            raise ValueError("Max proximity distance must be positive")
        if params['time_horizon'] <= 0:
            raise ValueError("Time horizon must be positive")
        if params['occlusion_depth_threshold'] <= 0:
            raise ValueError("Occlusion depth threshold must be positive")

        print("✓ Configuration validation passed")
        return True

    def print_summary(self):
        """Print configuration summary"""
        print("=" * 70)
        print("BEV RISK VISUALIZATION - CONFIGURATION SUMMARY")
        print("=" * 70)

        weights = self.config['risk_weights']
        print("\nRisk Factor Weights:")
        print(f"  α (Trajectory):  {weights['trajectory_alignment']:.2f}")
        print(f"  β (Occlusion):   {weights['occlusion_severity']:.2f}")
        print(f"  γ (Temporal):    {weights['temporal_urgency']:.2f}")
        print(f"  δ (Proximity):   {weights['proximity']:.2f}")

        bev_grid = self.config['bev_grid']
        x_range = (bev_grid['x_range']['min'], bev_grid['x_range']['max'])
        y_range = (bev_grid['y_range']['min'], bev_grid['y_range']['max'])
        print(f"\nBEV Grid:")
        print(f"  X Range:      {x_range[0]:.1f} to {x_range[1]:.1f} m")
        print(f"  Y Range:      {y_range[0]:.1f} to {y_range[1]:.1f} m")
        print(f"  Resolution:   {bev_grid['resolution']:.2f} m/cell")
        grid_size = (
            int((x_range[1] - x_range[0]) / bev_grid['resolution']),
            int((y_range[1] - y_range[0]) / bev_grid['resolution'])
        )
        print(f"  Grid Size:    {grid_size[0]} × {grid_size[1]} cells")

        ego = self.config['ego_vehicle']
        print(f"\nEgo Vehicle:")
        print(f"  Velocity:     {ego['velocity']:.1f} m/s")
        print(f"  Heading:      {ego['heading']:.2f} rad")

        data_src = self.config['data_source']
        print(f"\nData Source:")
        print(f"  nuScenes:     {data_src['nuscenes']['data_root']}")
        print(f"  Version:      {data_src['nuscenes']['version']}")

        export = self.config['export']
        print(f"\nExport:")
        print(f"  Output Dir:   {export['output_dir']}")
        print(f"  Formats:      {', '.join(export['default_formats'])}")

        print("=" * 70)


def load_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Convenience function to load configuration

    Args:
        config_path: Path to YAML configuration file

    Returns:
        ConfigLoader instance
    """
    loader = ConfigLoader(config_path)
    loader.validate()
    return loader


if __name__ == "__main__":
    # Test configuration loader
    import sys

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = None

    loader = load_config(config_path)
    loader.print_summary()
