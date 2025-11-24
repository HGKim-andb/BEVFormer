"""
BEV Risk Map Generator (BEV-RiskViz)

A comprehensive tool for visualizing and analyzing occlusion-based
emergence risk in Bird's-Eye View for autonomous driving.

Main Components:
    - RiskCalculationEngine: Core risk computation with 4 factors
    - NuScenesLoader: Dataset integration
    - RiskVisualizer: Visualization utilities
    - RiskDataExporter: Export in multiple formats
    - ConfigLoader: YAML configuration management

Usage:
    # Python API
    from tools.bev_risk_viz import RiskCalculationEngine, RiskConfig
    from tools.bev_risk_viz import RiskVisualizer, RiskDataExporter

    # CLI
    python tools/bev_risk_viz/cli.py --mode demo --export png,pdf

    # GUI
    streamlit run tools/bev_risk_viz/gui_app.py

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "BEVFormer Team"

# Import main classes for convenient access
from tools.bev_risk_viz.risk_engine import (
    RiskCalculationEngine,
    RiskConfig
)

from tools.bev_risk_viz.visualizer import RiskVisualizer

from tools.bev_risk_viz.exporter import RiskDataExporter

from tools.bev_risk_viz.config_loader import (
    ConfigLoader,
    load_config
)

try:
    from tools.bev_risk_viz.nuscenes_loader import (
        NuScenesLoader,
        SceneFrame
    )
except ImportError:
    # nuScenes not available
    NuScenesLoader = None
    SceneFrame = None

__all__ = [
    'RiskCalculationEngine',
    'RiskConfig',
    'RiskVisualizer',
    'RiskDataExporter',
    'ConfigLoader',
    'load_config',
    'NuScenesLoader',
    'SceneFrame',
]
