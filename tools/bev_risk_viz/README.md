# BEV Risk Map Generator (BEV-RiskViz)

**Occlusion-Based Emergence Risk Visualization for Autonomous Driving**

## Overview

BEV-RiskViz is a comprehensive tool for visualizing and analyzing risk maps in Bird's-Eye View (BEV) perspective for autonomous driving scenarios. It calculates occlusion-based emergence risk using four key factors and provides interactive visualization with parameter adjustment capabilities.

### Key Features

- **4-Factor Risk Calculation Engine**
  - θ (Trajectory Alignment): Alignment with ego vehicle trajectory
  - O (Occlusion Severity): Severity of occluded regions
  - T (Temporal Urgency): Time to potential collision
  - P (Proximity): Distance to ego vehicle

- **Multiple Input Modes**
  - nuScenes dataset integration
  - Custom scenario creation
  - Synthetic demo scenarios

- **Interactive GUI**
  - Real-time parameter adjustment
  - Multi-view visualizations
  - Factor breakdown analysis

- **Comprehensive Export**
  - PNG images (high-resolution heatmaps)
  - NumPy arrays (.npy, .npz)
  - CSV data files
  - PDF reports with analysis

## Installation

### Prerequisites

```bash
# Python 3.9 or higher
python --version

# Install dependencies
pip install -r tools/bev_risk_viz/requirements.txt
```

### Required Packages

```
numpy>=1.21.0
matplotlib>=3.5.0
opencv-python>=4.5.0
pyyaml>=6.0
streamlit>=1.20.0
scipy>=1.7.0
nuscenes-devkit>=1.1.9  # Optional, for nuScenes support
```

## Quick Start

### 1. Command-Line Usage

```bash
# Run with default demo scenario
python tools/bev_risk_viz/cli.py --demo

# Process nuScenes scene
python tools/bev_risk_viz/cli.py \
    --mode nuscenes \
    --scene scene-0001 \
    --output-dir visualizations/risk_maps

# Custom configuration
python tools/bev_risk_viz/cli.py \
    --config tools/bev_risk_viz/config.yaml \
    --demo \
    --export png,pdf
```

### 2. Interactive GUI

```bash
# Launch Streamlit app
streamlit run tools/bev_risk_viz/gui_app.py

# Access at http://localhost:8501
```

### 3. Python API

```python
from tools.bev_risk_viz.risk_engine import RiskCalculationEngine, RiskConfig
from tools.bev_risk_viz.visualizer import RiskVisualizer
from tools.bev_risk_viz.exporter import RiskDataExporter
import numpy as np

# Create configuration
config = RiskConfig(
    weight_trajectory=0.3,
    weight_occlusion=0.3,
    weight_temporal=0.2,
    weight_proximity=0.2,
    bev_x_range=(-50, 50),
    bev_y_range=(-50, 50),
    bev_resolution=0.5,
    ego_velocity=10.0,
    ego_heading=0.0
)

# Initialize engine
engine = RiskCalculationEngine(config)

# Create synthetic occlusion mask
H, W = engine.bev_height, engine.bev_width
occlusion_mask = np.zeros((H, W))
occlusion_mask[40:60, 60:80] = 1.0  # Simple occlusion region

# Calculate risk map
risk_results = engine.calculate_risk_map(occlusion_mask)

# Visualize
visualizer = RiskVisualizer()
fig, ax = plt.subplots(figsize=(10, 10))
visualizer.plot_risk_heatmap(
    risk_results['risk_map'],
    title='My Risk Map'
)
plt.show()

# Export
exporter = RiskDataExporter(output_dir='exports')
exporter.export_png(risk_results['risk_map'], 'my_risk_map')
exporter.export_pdf_report(risk_results, 'my_risk_report')
```

## Architecture

```
tools/bev_risk_viz/
├── risk_engine.py       # Core risk calculation engine
├── nuscenes_loader.py   # nuScenes dataset integration
├── visualizer.py        # Visualization utilities
├── exporter.py          # Export functionality
├── gui_app.py           # Streamlit interactive GUI
├── cli.py               # Command-line interface
├── config_loader.py     # YAML configuration loader
├── config.yaml          # Default configuration
├── README.md            # This file
└── __init__.py          # Package initialization
```

## Configuration

All parameters can be configured via YAML file ([config.yaml](config.yaml)):

### Risk Factor Weights

```yaml
risk_weights:
  trajectory_alignment: 0.3  # α
  occlusion_severity: 0.3    # β
  temporal_urgency: 0.2      # γ
  proximity: 0.2             # δ
```

### BEV Grid

```yaml
bev_grid:
  x_range:
    min: -50.0
    max: 50.0
  y_range:
    min: -50.0
    max: 50.0
  resolution: 0.5
```

### Ego Vehicle

```yaml
ego_vehicle:
  velocity: 10.0  # m/s
  heading: 0.0    # radians
```

## Usage Examples

### Example 1: Simple Occlusion Scenario

```python
from tools.bev_risk_viz.config_loader import load_config
from tools.bev_risk_viz.risk_engine import RiskCalculationEngine
from tools.bev_risk_viz.visualizer import RiskVisualizer

# Load configuration
config_loader = load_config()
config_loader.print_summary()

# Get risk config
risk_config = config_loader.get_risk_config()

# Create engine
engine = RiskCalculationEngine(risk_config)

# Create simple occlusion (vehicle in front)
import numpy as np
occlusion = np.zeros((engine.bev_height, engine.bev_width))
cy, cx = engine.bev_height // 2, engine.bev_width // 2
occlusion[cy-20:cy+20, cx+10:cx+30] = 1.0

# Calculate risk
results = engine.calculate_risk_map(occlusion)

# Visualize all factors
visualizer = RiskVisualizer()
fig = visualizer.plot_factor_breakdown(results)
plt.savefig('risk_breakdown.png', dpi=150)
```

### Example 2: nuScenes Integration

```python
from tools.bev_risk_viz.nuscenes_loader import NuScenesLoader
from tools.bev_risk_viz.risk_engine import RiskCalculationEngine, RiskConfig
from tools.bev_risk_viz.visualizer import RiskVisualizer

# Initialize loader
loader = NuScenesLoader(
    data_root='data/nuscenes',
    version='v1.0-mini'
)

# Get available scenes
scenes = loader.get_scene_list()
print(f"Found {len(scenes)} scenes")

# Load first scene
scene = scenes[0]
frames = loader.get_scene_frames(scene['token'], max_frames=10)

# Process first frame
frame_data = loader.load_frame(frames[0], load_images=False)

# Create occlusion from objects
occlusion_mask = loader.create_occlusion_mask_from_objects(
    frame_data.annotations,
    bev_range=(-50, 50, -50, 50),
    bev_resolution=0.5
)

# Get ego velocity
ego_speed, ego_heading = loader.get_ego_velocity(frames[0])

# Calculate risk with actual ego motion
config = RiskConfig()
engine = RiskCalculationEngine(config)
results = engine.calculate_risk_map(
    occlusion_mask,
    ego_velocity=ego_speed,
    ego_heading=ego_heading
)

# Visualize with objects
visualizer = RiskVisualizer()
visualizer.plot_with_objects(
    results['risk_map'],
    frame_data.annotations
)
plt.show()
```

### Example 3: Parameter Sensitivity Analysis

```python
import matplotlib.pyplot as plt
from tools.bev_risk_viz.risk_engine import RiskCalculationEngine, RiskConfig
from tools.bev_risk_viz.visualizer import RiskVisualizer

# Create occlusion scenario
occlusion = create_test_occlusion()  # Your occlusion data

# Test different trajectory weights
weights = [0.0, 0.3, 0.6, 1.0]
results_list = []

for w_traj in weights:
    config = RiskConfig(
        weight_trajectory=w_traj,
        weight_occlusion=1.0 - w_traj
    )
    engine = RiskCalculationEngine(config)
    results = engine.calculate_risk_map(occlusion)
    results_list.append(results['risk_map'])

# Visualize comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
visualizer = RiskVisualizer()

for idx, (w, risk_map) in enumerate(zip(weights, results_list)):
    ax = axes[idx // 2, idx % 2]
    visualizer.plot_risk_heatmap(
        risk_map,
        title=f'Trajectory Weight = {w:.1f}',
        ax=ax
    )

plt.tight_layout()
plt.savefig('parameter_sensitivity.png', dpi=150)
```

### Example 4: Batch Processing

```python
from tools.bev_risk_viz.nuscenes_loader import NuScenesLoader
from tools.bev_risk_viz.risk_engine import RiskCalculationEngine, RiskConfig
from tools.bev_risk_viz.exporter import RiskDataExporter

# Setup
loader = NuScenesLoader(data_root='data/nuscenes', version='v1.0-mini')
engine = RiskCalculationEngine(RiskConfig())
exporter = RiskDataExporter(output_dir='batch_exports')

# Get scene frames
scene = loader.get_scene_list()[0]
frames = loader.get_scene_frames(scene['token'], max_frames=50)

# Process all frames
risk_results_list = []

for i, frame_token in enumerate(frames):
    print(f"Processing frame {i+1}/{len(frames)}...")

    # Load frame
    frame_data = loader.load_frame(frame_token)

    # Create occlusion
    occlusion = loader.create_occlusion_mask_from_objects(
        frame_data.annotations
    )

    # Calculate risk
    results = engine.calculate_risk_map(occlusion)
    risk_results_list.append(results)

# Batch export
exporter.export_batch(
    risk_results_list,
    base_filename=f"{scene['name']}_risk",
    formats=['png', 'npy']
)

print(f"✓ Processed {len(frames)} frames")
```

### Example 5: Animation Creation

```python
from tools.bev_risk_viz.visualizer import RiskVisualizer
from tools.bev_risk_viz.exporter import RiskDataExporter

# Assume you have risk_results_list from batch processing
risk_maps = [r['risk_map'] for r in risk_results_list]

# Create animation
visualizer = RiskVisualizer()
visualizer.create_animation(
    risk_maps,
    output_path='scene_risk_animation.gif',
    fps=2,
    title_prefix='Scene Frame'
)

# Or export individual frames
exporter = RiskDataExporter()
exporter.export_animation_frames(
    risk_maps,
    filename_prefix='scene_animation',
    dpi=100
)
```

## Command-Line Interface

```bash
# General usage
python tools/bev_risk_viz/cli.py [OPTIONS]

# Options:
#   --mode {demo,nuscenes,custom}   Input mode
#   --config PATH                   Path to config.yaml
#   --demo-scenario NAME            Demo scenario name
#   --scene NAME                    nuScenes scene name
#   --output-dir PATH               Output directory
#   --export FORMAT                 Export formats (png,npy,csv,pdf)
#   --show                          Display visualizations
#   --batch                         Process all frames in scene

# Examples:

# 1. Run demo with all exports
python tools/bev_risk_viz/cli.py \
    --mode demo \
    --demo-scenario "Multi-Vehicle Intersection" \
    --export png,pdf,npy \
    --show

# 2. Process nuScenes scene with batch mode
python tools/bev_risk_viz/cli.py \
    --mode nuscenes \
    --scene scene-0001 \
    --batch \
    --output-dir visualizations/scene-0001 \
    --export png

# 3. Custom configuration
python tools/bev_risk_viz/cli.py \
    --config my_config.yaml \
    --mode demo \
    --export pdf
```

## Risk Formula

The final risk score for each grid cell is calculated as:

```
R = (α·θ + β·O + γ·T + δ·P) / (α + β + γ + δ)
```

Where:
- **θ (Trajectory Alignment)**: `θ = max(0, cos(angle_to_cell)) · velocity_scale`
- **O (Occlusion Severity)**: `O = occlusion_mask · depth_factor`
- **T (Temporal Urgency)**: `T = exp(-time_to_reach / time_horizon)`
- **P (Proximity)**: `P = exp(-2 · distance / max_distance)`

All factors are normalized to [0, 1].

## Output Formats

### PNG Images
High-resolution risk heatmaps with customizable colormaps (green → yellow → red).

### NumPy Arrays
- `.npy`: Single risk map array
- `.npz`: Compressed archive with risk map and all factors

### CSV Files
Tabular data with optional spatial coordinates:
```csv
X,Y,Risk
-50.00,-50.00,0.123456
-49.50,-50.00,0.234567
...
```

### PDF Reports
Comprehensive multi-page reports including:
- Page 1: Metadata and configuration
- Page 2: Main risk heatmap
- Page 3: Factor breakdown (θ, O, T, P)
- Page 4: Statistical analysis and distributions

## Performance Considerations

- **Grid Resolution**: Lower resolution (0.5-1.0m) is faster, higher (0.1-0.2m) is more detailed
- **Batch Processing**: Use multiprocessing for large datasets (configure in YAML)
- **Memory**: 200×200 grid at 0.5m resolution ≈ 160KB per risk map

## Troubleshooting

### Issue: nuScenes not loading
```python
# Check nuscenes-devkit installation
pip install nuscenes-devkit

# Verify data path
ls data/nuscenes/  # Should contain v1.0-mini or v1.0-trainval
```

### Issue: Memory error with large grids
```yaml
# Reduce resolution in config.yaml
bev_grid:
  resolution: 1.0  # Increase from 0.5 to 1.0
```

### Issue: Slow visualization
```python
# Use lower DPI for faster rendering
visualizer = RiskVisualizer(dpi=100)  # Default is 150
```

## Citation

If you use BEV-RiskViz in your research, please cite:

```bibtex
@software{bev_riskviz,
  title={BEV-RiskViz: Bird's-Eye View Risk Visualization for Autonomous Driving},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/BEVFormer}
}
```

## License

This tool is part of the BEVFormer project. See the main repository for license information.

## Contact

For questions, issues, or contributions:
- GitHub Issues: [https://github.com/yourusername/BEVFormer/issues](link)
- Email: your.email@example.com

## Acknowledgments

- Built on top of [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- Uses [nuScenes dataset](https://www.nuscenes.org/)
- Visualization powered by Matplotlib and Streamlit
