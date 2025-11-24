#!/usr/bin/env python3
"""
BEV Risk Map Generator - Quick Start Example

This script demonstrates basic usage of the BEV Risk Map Generator.
Run this to verify your installation and see example outputs.

Usage:
    python tools/bev_risk_viz/example_usage.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.bev_risk_viz.risk_engine import RiskCalculationEngine, RiskConfig
from tools.bev_risk_viz.visualizer import RiskVisualizer
from tools.bev_risk_viz.exporter import RiskDataExporter
from tools.bev_risk_viz.config_loader import load_config


def example_1_simple_usage():
    """Example 1: Simple risk map generation"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Simple Risk Map Generation")
    print("=" * 70)

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
        ego_heading=0.0  # Pointing forward (north)
    )

    # Initialize engine
    engine = RiskCalculationEngine(config)
    print(f"✓ Created BEV grid: {engine.bev_width} × {engine.bev_height} cells")

    # Create simple occlusion scenario (vehicle directly ahead)
    H, W = engine.bev_height, engine.bev_width
    occlusion_mask = np.zeros((H, W), dtype=np.float32)

    # Place occluding vehicle at (x=0, y=15m) in front of ego
    center_x = W // 2
    center_y = H // 2 + 30  # 15 meters ahead
    occlusion_mask[center_y - 10:center_y + 10, center_x - 10:center_x + 10] = 1.0

    print("✓ Created occlusion scenario: vehicle ahead")

    # Calculate risk map
    risk_results = engine.calculate_risk_map(occlusion_mask)
    print("✓ Calculated risk map")

    # Print statistics
    risk_map = risk_results['risk_map']
    print(f"\nRisk Statistics:")
    print(f"  Max Risk:    {risk_map.max():.4f}")
    print(f"  Mean Risk:   {risk_map.mean():.6f}")
    print(f"  Std Dev:     {risk_map.std():.6f}")

    # Visualize
    visualizer = RiskVisualizer()
    fig, ax = plt.subplots(figsize=(10, 10))
    visualizer.plot_risk_heatmap(
        risk_map,
        bev_range=(-50, 50, -50, 50),
        title="Example 1: Simple Occlusion Risk Map",
        ax=ax
    )

    output_path = "example_1_simple.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    plt.close()


def example_2_factor_breakdown():
    """Example 2: Analyzing individual risk factors"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Risk Factor Breakdown")
    print("=" * 70)

    # Load default configuration
    config_loader = load_config()
    risk_config = config_loader.get_risk_config()
    engine = RiskCalculationEngine(risk_config)

    # Create multi-vehicle scenario
    H, W = engine.bev_height, engine.bev_width
    occlusion_mask = np.zeros((H, W), dtype=np.float32)
    cx, cy = W // 2, H // 2

    # Three vehicles at different positions
    occlusion_mask[cy + 20:cy + 40, cx - 15:cx + 15] = 1.0  # Front
    occlusion_mask[cy - 10:cy + 10, cx + 40:cx + 60] = 1.0  # Right
    occlusion_mask[cy - 10:cy + 10, cx - 60:cx - 40] = 1.0  # Left

    print("✓ Created multi-vehicle scenario")

    # Calculate risk with all factors
    risk_results = engine.calculate_risk_map(occlusion_mask)
    print("✓ Calculated risk factors")

    # Visualize breakdown
    visualizer = RiskVisualizer()
    fig = visualizer.plot_factor_breakdown(
        risk_results,
        bev_range=(-50, 50, -50, 50),
        title="Example 2: Risk Factor Analysis"
    )

    output_path = "example_2_breakdown.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved factor breakdown to {output_path}")
    plt.close()


def example_3_parameter_comparison():
    """Example 3: Comparing different parameter settings"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Parameter Comparison")
    print("=" * 70)

    # Create occlusion scenario
    base_config = RiskConfig(bev_resolution=0.5)
    engine = RiskCalculationEngine(base_config)

    H, W = engine.bev_height, engine.bev_width
    occlusion_mask = np.zeros((H, W), dtype=np.float32)
    cx, cy = W // 2, H // 2
    occlusion_mask[cy - 20:cy + 20, cx + 15:cx + 35] = 1.0

    # Configuration 1: High trajectory weight
    config1 = RiskConfig(
        weight_trajectory=0.7,
        weight_occlusion=0.1,
        weight_temporal=0.1,
        weight_proximity=0.1,
        ego_velocity=15.0
    )
    engine1 = RiskCalculationEngine(config1)
    results1 = engine1.calculate_risk_map(occlusion_mask)

    # Configuration 2: High occlusion weight
    config2 = RiskConfig(
        weight_trajectory=0.1,
        weight_occlusion=0.7,
        weight_temporal=0.1,
        weight_proximity=0.1,
        ego_velocity=15.0
    )
    engine2 = RiskCalculationEngine(config2)
    results2 = engine2.calculate_risk_map(occlusion_mask)

    print("✓ Calculated risk with different parameter sets")

    # Visualize comparison
    visualizer = RiskVisualizer()
    fig = visualizer.plot_comparison(
        results1['risk_map'],
        results2['risk_map'],
        labels=("High Trajectory Weight", "High Occlusion Weight"),
        bev_range=(-50, 50, -50, 50),
        title="Example 3: Parameter Impact on Risk"
    )

    output_path = "example_3_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison to {output_path}")
    plt.close()


def example_4_export_formats():
    """Example 4: Exporting in different formats"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Export Formats")
    print("=" * 70)

    # Create risk scenario
    config = RiskConfig()
    engine = RiskCalculationEngine(config)

    H, W = engine.bev_height, engine.bev_width
    occlusion_mask = np.zeros((H, W), dtype=np.float32)
    cx, cy = W // 2, H // 2
    occlusion_mask[cy:cy + 30, cx + 10:cx + 40] = 1.0

    risk_results = engine.calculate_risk_map(occlusion_mask)
    print("✓ Calculated risk map")

    # Export in multiple formats
    exporter = RiskDataExporter(output_dir="example_exports")
    visualizer = RiskVisualizer()

    # PNG
    exporter.export_png(
        risk_results['risk_map'],
        "example_4_risk_map",
        visualizer=visualizer
    )

    # NumPy
    exporter.export_numpy(
        risk_results,
        "example_4_risk_data",
        include_factors=True
    )

    # CSV
    exporter.export_csv(
        risk_results['risk_map'],
        "example_4_risk_map",
        include_coordinates=True
    )

    # PDF Report
    exporter.export_pdf_report(
        risk_results,
        "example_4_risk_report",
        visualizer=visualizer,
        config={
            'trajectory_weight': config.weight_trajectory,
            'occlusion_weight': config.weight_occlusion,
            'temporal_weight': config.weight_temporal,
            'proximity_weight': config.weight_proximity,
        },
        metadata={
            'Example': '4 - Export Formats',
            'Description': 'Demonstration of export capabilities'
        }
    )

    print("✓ Exported in all formats (PNG, NPY, CSV, PDF)")


def example_5_velocity_impact():
    """Example 5: Effect of ego vehicle velocity on risk"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Velocity Impact on Risk")
    print("=" * 70)

    # Create occlusion
    base_config = RiskConfig(bev_resolution=0.5)
    engine = RiskCalculationEngine(base_config)

    H, W = engine.bev_height, engine.bev_width
    occlusion_mask = np.zeros((H, W), dtype=np.float32)
    cx, cy = W // 2, H // 2
    occlusion_mask[cy + 10:cy + 40, cx - 15:cx + 15] = 1.0

    # Test different velocities
    velocities = [0.0, 5.0, 15.0, 25.0]  # m/s
    risk_maps = []

    for vel in velocities:
        config = RiskConfig(
            weight_trajectory=0.4,
            weight_temporal=0.4,
            weight_occlusion=0.1,
            weight_proximity=0.1,
            ego_velocity=vel
        )
        engine = RiskCalculationEngine(config)
        results = engine.calculate_risk_map(occlusion_mask, ego_velocity=vel)
        risk_maps.append(results['risk_map'])

        max_risk = results['risk_map'].max()
        print(f"  Velocity {vel:5.1f} m/s → Max Risk: {max_risk:.4f}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    visualizer = RiskVisualizer()

    for idx, (vel, risk_map) in enumerate(zip(velocities, risk_maps)):
        ax = axes[idx // 2, idx % 2]
        visualizer.plot_risk_heatmap(
            risk_map,
            bev_range=(-50, 50, -50, 50),
            title=f"Velocity = {vel:.1f} m/s",
            ax=ax
        )

    plt.suptitle("Example 5: Impact of Ego Velocity on Risk", fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = "example_5_velocity_impact.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved velocity comparison to {output_path}")
    plt.close()


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("BEV RISK MAP GENERATOR - EXAMPLE USAGE")
    print("=" * 70)
    print("\nThis script demonstrates the main features of BEV-RiskViz.")
    print("Five examples will be generated with visualizations.\n")

    try:
        example_1_simple_usage()
        example_2_factor_breakdown()
        example_3_parameter_comparison()
        example_4_export_formats()
        example_5_velocity_impact()

        print("\n" + "=" * 70)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - example_1_simple.png")
        print("  - example_2_breakdown.png")
        print("  - example_3_comparison.png")
        print("  - example_4_* (multiple formats in example_exports/)")
        print("  - example_5_velocity_impact.png")
        print("\nNext steps:")
        print("  1. Try the interactive GUI: streamlit run tools/bev_risk_viz/gui_app.py")
        print("  2. Use CLI tool: python tools/bev_risk_viz/cli.py --help")
        print("  3. Read documentation: tools/bev_risk_viz/README.md")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
