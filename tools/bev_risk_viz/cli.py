#!/usr/bin/env python3
"""
BEV Risk Map Generator - Command Line Interface

Command-line tool for BEV risk visualization and analysis.

Usage:
    python tools/bev_risk_viz/cli.py --mode demo --export png,pdf
    python tools/bev_risk_viz/cli.py --mode nuscenes --scene scene-0001 --batch
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.bev_risk_viz.config_loader import load_config
from tools.bev_risk_viz.risk_engine import RiskCalculationEngine
from tools.bev_risk_viz.nuscenes_loader import NuScenesLoader
from tools.bev_risk_viz.visualizer import RiskVisualizer
from tools.bev_risk_viz.exporter import RiskDataExporter


def create_demo_occlusion(scenario_name: str, H: int, W: int) -> np.ndarray:
    """
    Create synthetic occlusion mask for demo scenarios

    Args:
        scenario_name: Name of demo scenario
        H: Height of BEV grid
        W: Width of BEV grid

    Returns:
        Occlusion mask [H, W]
    """
    occlusion_mask = np.zeros((H, W), dtype=np.float32)
    cx, cy = W // 2, H // 2

    if scenario_name == "Simple Occlusion":
        occlusion_mask[cy - 20:cy + 20, cx + 10:cx + 30] = 1.0

    elif scenario_name == "Multi-Vehicle Intersection":
        occlusion_mask[cy - 30:cy - 10, cx + 20:cx + 40] = 1.0
        occlusion_mask[cy + 10:cy + 30, cx - 40:cx - 20] = 1.0
        occlusion_mask[cy - 10:cy + 10, cx + 50:cx + 70] = 1.0

    elif scenario_name == "Parking Lot Exit":
        for i in range(5):
            y_pos = cy + 20 + i * 25
            if y_pos < H - 10:
                occlusion_mask[y_pos:y_pos + 15, cx - 25:cx - 15] = 1.0
                occlusion_mask[y_pos:y_pos + 15, cx + 15:cx + 25] = 1.0

    elif scenario_name == "Highway Merge":
        occlusion_mask[cy - 40:cy - 20, cx + 30:cx + 50] = 1.0
        occlusion_mask[cy + 20:cy + 40, cx + 20:cx + 40] = 1.0

    elif scenario_name == "Pedestrian Crossing":
        occlusion_mask[cy - 50:cy + 50, cx + 15:cx + 25] = 1.0

    else:
        print(f"Warning: Unknown scenario '{scenario_name}', using simple occlusion")
        occlusion_mask[cy - 20:cy + 20, cx + 10:cx + 30] = 1.0

    return occlusion_mask


def run_demo_mode(args, config_loader):
    """Run in demo mode with synthetic scenario"""
    print("\n" + "=" * 70)
    print("BEV RISK VISUALIZATION - DEMO MODE")
    print("=" * 70)

    # Get configuration
    risk_config = config_loader.get_risk_config()
    engine = RiskCalculationEngine(risk_config)

    # Create demo occlusion
    scenario = args.demo_scenario or "Simple Occlusion"
    print(f"\nScenario: {scenario}")

    occlusion_mask = create_demo_occlusion(
        scenario,
        engine.bev_height,
        engine.bev_width
    )

    # Calculate risk
    print("Calculating risk map...")
    risk_results = engine.calculate_risk_map(occlusion_mask)

    # Print statistics
    risk_map = risk_results['risk_map']
    print(f"\nRisk Statistics:")
    print(f"  Max Risk:  {risk_map.max():.4f}")
    print(f"  Mean Risk: {risk_map.mean():.6f}")
    print(f"  Std Dev:   {risk_map.std():.6f}")

    # Visualization
    visualizer = RiskVisualizer()
    exporter = RiskDataExporter(output_dir=args.output_dir)

    bev_range = (
        risk_config.bev_x_range[0],
        risk_config.bev_x_range[1],
        risk_config.bev_y_range[0],
        risk_config.bev_y_range[1]
    )

    # Export in requested formats
    export_formats = args.export.split(',') if args.export else []

    if 'png' in export_formats or args.show:
        print("\nGenerating PNG visualization...")
        exporter.export_png(
            risk_map,
            f"{scenario.replace(' ', '_')}_risk_map",
            visualizer=visualizer,
            bev_range=bev_range,
            title=f"Risk Map - {scenario}"
        )

    if 'npy' in export_formats:
        print("Exporting NumPy array...")
        exporter.export_numpy(
            risk_results,
            f"{scenario.replace(' ', '_')}_risk_data"
        )

    if 'csv' in export_formats:
        print("Exporting CSV...")
        exporter.export_csv(
            risk_map,
            f"{scenario.replace(' ', '_')}_risk_map",
            bev_range=bev_range
        )

    if 'pdf' in export_formats:
        print("Generating PDF report...")
        exporter.export_pdf_report(
            risk_results,
            f"{scenario.replace(' ', '_')}_report",
            visualizer=visualizer,
            config=vars(risk_config),
            metadata={'Scenario': scenario},
            bev_range=bev_range
        )

    if args.show:
        print("\nDisplaying visualization...")
        fig = visualizer.plot_factor_breakdown(
            risk_results,
            bev_range=bev_range,
            title=f"Risk Analysis - {scenario}"
        )
        plt.show()

    print("\n✓ Demo mode completed successfully")


def run_nuscenes_mode(args, config_loader):
    """Run in nuScenes mode"""
    print("\n" + "=" * 70)
    print("BEV RISK VISUALIZATION - NUSCENES MODE")
    print("=" * 70)

    # Initialize nuScenes loader
    nuscenes_config = config_loader.get('data_source.nuscenes')
    print(f"\nLoading nuScenes dataset from {nuscenes_config['data_root']}...")

    try:
        loader = NuScenesLoader(
            data_root=nuscenes_config['data_root'],
            version=nuscenes_config['version']
        )
    except Exception as e:
        print(f"Error loading nuScenes: {e}")
        return

    # Get risk engine
    risk_config = config_loader.get_risk_config()
    engine = RiskCalculationEngine(risk_config)
    visualizer = RiskVisualizer()
    exporter = RiskDataExporter(output_dir=args.output_dir)

    # Load scene
    scene_name = args.scene
    if not scene_name:
        scenes = loader.get_scene_list()
        print(f"\nAvailable scenes ({len(scenes)}):")
        for i, scene in enumerate(scenes[:10]):
            print(f"  {i + 1}. {scene['name']}: {scene['description']}")
        print("\nPlease specify a scene with --scene")
        return

    print(f"\nLoading scene: {scene_name}")
    try:
        scene = loader.load_scene_by_name(scene_name)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Get frames
    max_frames = None if args.batch else 1
    frames = loader.get_scene_frames(scene['token'], max_frames=max_frames)
    print(f"Found {len(frames)} frames")

    bev_range = (
        risk_config.bev_x_range[0],
        risk_config.bev_x_range[1],
        risk_config.bev_y_range[0],
        risk_config.bev_y_range[1]
    )

    # Process frames
    for i, frame_token in enumerate(frames):
        print(f"\nProcessing frame {i + 1}/{len(frames)}...")

        # Load frame data
        frame_data = loader.load_frame(frame_token, load_images=False)

        # Create occlusion from objects
        occlusion_mask = loader.create_occlusion_mask_from_objects(
            frame_data.annotations,
            bev_range=bev_range,
            bev_resolution=risk_config.bev_resolution
        )

        # Get ego velocity
        ego_speed, ego_heading = loader.get_ego_velocity(frame_token)

        # Calculate risk
        risk_results = engine.calculate_risk_map(
            occlusion_mask,
            ego_velocity=ego_speed,
            ego_heading=ego_heading
        )

        # Export
        filename_base = f"{scene_name}_frame_{i:04d}"

        export_formats = args.export.split(',') if args.export else ['png']

        if 'png' in export_formats:
            exporter.export_png(
                risk_results['risk_map'],
                filename_base,
                visualizer=visualizer,
                bev_range=bev_range,
                title=f"{scene_name} - Frame {i + 1}"
            )

        if 'npy' in export_formats:
            exporter.export_numpy(risk_results, filename_base)

        if 'csv' in export_formats:
            exporter.export_csv(risk_results['risk_map'], filename_base, bev_range=bev_range)

        if 'pdf' in export_formats and i == 0:  # Only first frame for PDF
            exporter.export_pdf_report(
                risk_results,
                f"{scene_name}_report",
                visualizer=visualizer,
                metadata={
                    'Scene': scene_name,
                    'Frame': f"{i + 1}/{len(frames)}",
                    'Ego Speed': f"{ego_speed:.2f} m/s"
                },
                bev_range=bev_range
            )

    print(f"\n✓ Processed {len(frames)} frames successfully")


def main():
    parser = argparse.ArgumentParser(
        description='BEV Risk Map Generator - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo mode with PDF export
  python tools/bev_risk_viz/cli.py --mode demo --export png,pdf

  # nuScenes single frame
  python tools/bev_risk_viz/cli.py --mode nuscenes --scene scene-0001

  # nuScenes batch processing
  python tools/bev_risk_viz/cli.py --mode nuscenes --scene scene-0001 --batch --export png,npy
        """
    )

    parser.add_argument(
        '--mode',
        choices=['demo', 'nuscenes', 'custom'],
        default='demo',
        help='Input mode (default: demo)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--demo-scenario',
        type=str,
        choices=[
            'Simple Occlusion',
            'Multi-Vehicle Intersection',
            'Parking Lot Exit',
            'Highway Merge',
            'Pedestrian Crossing'
        ],
        default='Simple Occlusion',
        help='Demo scenario name'
    )

    parser.add_argument(
        '--scene',
        type=str,
        help='nuScenes scene name (e.g., scene-0001)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='exports',
        help='Output directory for exports (default: exports)'
    )

    parser.add_argument(
        '--export',
        type=str,
        default='png',
        help='Export formats (comma-separated: png,npy,csv,pdf)'
    )

    parser.add_argument(
        '--show',
        action='store_true',
        help='Display visualizations interactively'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all frames in scene (nuScenes mode only)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    config_loader = load_config(args.config)

    if args.verbose:
        config_loader.print_summary()

    # Run appropriate mode
    if args.mode == 'demo':
        run_demo_mode(args, config_loader)
    elif args.mode == 'nuscenes':
        run_nuscenes_mode(args, config_loader)
    elif args.mode == 'custom':
        print("Custom mode not yet implemented")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
