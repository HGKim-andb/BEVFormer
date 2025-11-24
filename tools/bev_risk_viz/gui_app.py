#!/usr/bin/env python3
"""
BEV Risk Map Generator - Interactive GUI Application

Streamlit-based interactive application for BEV risk visualization
with real-time parameter adjustment and scenario exploration.

Usage:
    streamlit run tools/bev_risk_viz/gui_app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.bev_risk_viz.risk_engine import RiskCalculationEngine, RiskConfig
from tools.bev_risk_viz.nuscenes_loader import NuScenesLoader
from tools.bev_risk_viz.visualizer import RiskVisualizer


def init_session_state():
    """Initialize Streamlit session state"""
    if 'loader' not in st.session_state:
        st.session_state.loader = None
    if 'engine' not in st.session_state:
        st.session_state.engine = None
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = RiskVisualizer(style='default')
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = None
    if 'scene_frames' not in st.session_state:
        st.session_state.scene_frames = []


def main():
    st.set_page_config(
        page_title="BEV Risk Map Generator",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🚗 BEV Risk Map Generator (BEV-RiskViz)")
    st.markdown("""
    **Occlusion-based Emergence Risk Visualization for Autonomous Driving**

    This tool visualizes risk maps in Bird's-Eye View (BEV) based on four risk factors:
    - **θ (Trajectory Alignment)**: Alignment with ego vehicle trajectory
    - **O (Occlusion Severity)**: Severity of occluded regions
    - **T (Temporal Urgency)**: Time to potential collision
    - **P (Proximity)**: Distance to ego vehicle
    """)

    init_session_state()

    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Data source selection
        st.subheader("Data Source")
        data_mode = st.radio(
            "Select input mode:",
            ["nuScenes Dataset", "Custom Scenario", "Demo (Synthetic)"]
        )

        if data_mode == "nuScenes Dataset":
            render_nuscenes_config()
        elif data_mode == "Custom Scenario":
            render_custom_scenario_config()
        else:
            render_demo_config()

        st.divider()

        # Risk factor weights
        st.subheader("Risk Factor Weights")

        weight_trajectory = st.slider(
            "α - Trajectory Alignment",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Weight for trajectory alignment factor"
        )

        weight_occlusion = st.slider(
            "β - Occlusion Severity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Weight for occlusion severity factor"
        )

        weight_temporal = st.slider(
            "γ - Temporal Urgency",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Weight for temporal urgency factor"
        )

        weight_proximity = st.slider(
            "δ - Proximity",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Weight for proximity factor"
        )

        st.divider()

        # Ego vehicle parameters
        st.subheader("Ego Vehicle Parameters")

        ego_velocity = st.slider(
            "Velocity (m/s)",
            min_value=0.0,
            max_value=30.0,
            value=10.0,
            step=0.5,
            help="Ego vehicle velocity"
        )

        ego_heading = st.slider(
            "Heading (degrees)",
            min_value=-180,
            max_value=180,
            value=0,
            step=5,
            help="Ego vehicle heading (0° = North/Forward)"
        )

        st.divider()

        # BEV grid configuration
        st.subheader("BEV Grid Configuration")

        bev_range_x = st.slider(
            "X Range (meters)",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            help="BEV range in X direction (±value)"
        )

        bev_range_y = st.slider(
            "Y Range (meters)",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            help="BEV range in Y direction (±value)"
        )

        bev_resolution = st.select_slider(
            "Grid Resolution (meters)",
            options=[0.1, 0.2, 0.5, 1.0],
            value=0.5,
            help="Size of each grid cell"
        )

    # Create risk engine with current configuration
    config = RiskConfig(
        weight_trajectory=weight_trajectory,
        weight_occlusion=weight_occlusion,
        weight_temporal=weight_temporal,
        weight_proximity=weight_proximity,
        bev_x_range=(-bev_range_x, bev_range_x),
        bev_y_range=(-bev_range_y, bev_range_y),
        bev_resolution=bev_resolution,
        ego_velocity=ego_velocity,
        ego_heading=np.radians(ego_heading)
    )

    st.session_state.engine = RiskCalculationEngine(config)

    # Main content area
    if data_mode == "Demo (Synthetic)":
        render_demo_visualization()
    elif data_mode == "Custom Scenario":
        render_custom_visualization()
    elif data_mode == "nuScenes Dataset":
        render_nuscenes_visualization()


def render_nuscenes_config():
    """Render nuScenes dataset configuration"""
    st.text_input(
        "Data Root",
        value="data/nuscenes",
        key="nuscenes_root",
        help="Path to nuScenes dataset"
    )

    st.selectbox(
        "Version",
        ["v1.0-mini", "v1.0-trainval"],
        key="nuscenes_version"
    )

    if st.button("Load Dataset"):
        with st.spinner("Loading nuScenes dataset..."):
            try:
                loader = NuScenesLoader(
                    data_root=st.session_state.nuscenes_root,
                    version=st.session_state.nuscenes_version
                )
                st.session_state.loader = loader
                st.success(f"✓ Loaded {len(loader.get_scene_list())} scenes")
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")


def render_custom_scenario_config():
    """Render custom scenario configuration"""
    st.info("Draw occlusion regions and place objects manually")

    st.number_input(
        "Number of occluding objects",
        min_value=0,
        max_value=20,
        value=3,
        key="num_objects"
    )


def render_demo_config():
    """Render demo mode configuration"""
    st.selectbox(
        "Demo Scenario",
        [
            "Simple Occlusion",
            "Multi-Vehicle Intersection",
            "Parking Lot Exit",
            "Highway Merge",
            "Pedestrian Crossing"
        ],
        key="demo_scenario"
    )


def render_demo_visualization():
    """Render demo visualization with synthetic data"""
    scenario = st.session_state.get("demo_scenario", "Simple Occlusion")

    # Create synthetic occlusion mask based on scenario
    engine = st.session_state.engine
    H, W = engine.bev_height, engine.bev_width

    occlusion_mask = create_demo_scenario(scenario, H, W, engine.config)

    # Calculate risk map
    risk_results = engine.calculate_risk_map(occlusion_mask)

    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Risk Map",
        "🔍 Factor Breakdown",
        "📈 Statistics",
        "💾 Export"
    ])

    with tab1:
        st.subheader("BEV Risk Heatmap")
        fig, ax = plt.subplots(figsize=(10, 10))
        st.session_state.visualizer.plot_risk_heatmap(
            risk_results['risk_map'],
            bev_range=(
                engine.config.bev_x_range[0],
                engine.config.bev_x_range[1],
                engine.config.bev_y_range[0],
                engine.config.bev_y_range[1]
            ),
            title=f"Risk Map - {scenario}",
            ax=ax
        )
        st.pyplot(fig)

        # Add occlusion overlay
        if st.checkbox("Show Occlusion Overlay"):
            fig2, ax2 = plt.subplots(figsize=(10, 10))
            st.session_state.visualizer.plot_occlusion_overlay(
                risk_results['risk_map'],
                occlusion_mask,
                bev_range=(
                    engine.config.bev_x_range[0],
                    engine.config.bev_x_range[1],
                    engine.config.bev_y_range[0],
                    engine.config.bev_y_range[1]
                ),
                ax=ax2
            )
            st.pyplot(fig2)

    with tab2:
        st.subheader("Risk Factor Breakdown")
        fig = st.session_state.visualizer.plot_factor_breakdown(
            risk_results,
            bev_range=(
                engine.config.bev_x_range[0],
                engine.config.bev_x_range[1],
                engine.config.bev_y_range[0],
                engine.config.bev_y_range[1]
            ),
            title="Risk Factor Analysis"
        )
        st.pyplot(fig)

    with tab3:
        st.subheader("Risk Statistics")

        col1, col2, col3 = st.columns(3)

        risk_map = risk_results['risk_map']

        with col1:
            st.metric("Max Risk", f"{risk_map.max():.3f}")
            st.metric("Mean Risk", f"{risk_map.mean():.4f}")

        with col2:
            st.metric("Std Deviation", f"{risk_map.std():.4f}")
            st.metric("Median Risk", f"{np.median(risk_map):.4f}")

        with col3:
            high_risk_pct = (risk_map > 0.7).sum() / risk_map.size * 100
            med_risk_pct = ((risk_map >= 0.3) & (risk_map <= 0.7)).sum() / risk_map.size * 100
            low_risk_pct = (risk_map < 0.3).sum() / risk_map.size * 100

            st.metric("High Risk (>0.7)", f"{high_risk_pct:.1f}%")
            st.metric("Medium Risk", f"{med_risk_pct:.1f}%")
            st.metric("Low Risk (<0.3)", f"{low_risk_pct:.1f}%")

        # Risk histogram
        st.subheader("Risk Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(risk_map.flatten(), bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Risk Level')
        ax.set_ylabel('Frequency')
        ax.set_title('Risk Distribution Histogram')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with tab4:
        st.subheader("Export Options")

        export_format = st.selectbox(
            "Export Format",
            ["PNG Image", "NumPy Array (.npy)", "CSV Data", "PDF Report"]
        )

        if st.button("Export"):
            export_visualization(risk_results, export_format, scenario)


def render_custom_visualization():
    """Render custom scenario visualization"""
    st.info("Custom scenario mode - implement interactive drawing here")


def render_nuscenes_visualization():
    """Render nuScenes dataset visualization"""
    if st.session_state.loader is None:
        st.warning("⚠️ Please load the nuScenes dataset first (see sidebar)")
        return

    loader = st.session_state.loader
    scenes = loader.get_scene_list()

    # Scene selection
    scene_names = [s['name'] for s in scenes]
    selected_scene = st.selectbox("Select Scene", scene_names)

    if selected_scene:
        # Load scene frames
        scene = loader.load_scene_by_name(selected_scene)
        frames = loader.get_scene_frames(scene['token'], max_frames=50)

        # Frame navigation
        frame_idx = st.slider(
            "Frame",
            min_value=0,
            max_value=len(frames) - 1,
            value=0
        )

        # Load frame data
        frame_data = loader.load_frame(frames[frame_idx], load_images=False)

        # Create occlusion mask from objects
        occlusion_mask = loader.create_occlusion_mask_from_objects(
            frame_data.annotations,
            bev_range=(
                st.session_state.engine.config.bev_x_range[0],
                st.session_state.engine.config.bev_x_range[1],
                st.session_state.engine.config.bev_y_range[0],
                st.session_state.engine.config.bev_y_range[1]
            ),
            bev_resolution=st.session_state.engine.config.bev_resolution
        )

        # Get ego velocity
        ego_speed, ego_heading = loader.get_ego_velocity(frames[frame_idx])

        # Calculate risk
        risk_results = st.session_state.engine.calculate_risk_map(
            occlusion_mask,
            ego_velocity=ego_speed,
            ego_heading=ego_heading
        )

        # Display
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Risk Map")
            fig, ax = plt.subplots(figsize=(8, 8))
            st.session_state.visualizer.plot_with_objects(
                risk_results['risk_map'],
                frame_data.annotations,
                bev_range=(
                    st.session_state.engine.config.bev_x_range[0],
                    st.session_state.engine.config.bev_x_range[1],
                    st.session_state.engine.config.bev_y_range[0],
                    st.session_state.engine.config.bev_y_range[1]
                ),
                ax=ax
            )
            st.pyplot(fig)

        with col2:
            st.subheader("Scene Information")
            st.write(f"**Scene:** {selected_scene}")
            st.write(f"**Frame:** {frame_idx + 1}/{len(frames)}")
            st.write(f"**Timestamp:** {frame_data.timestamp}")
            st.write(f"**Ego Speed:** {ego_speed:.2f} m/s")
            st.write(f"**Ego Heading:** {np.degrees(ego_heading):.1f}°")
            st.write(f"**Objects:** {len(frame_data.annotations)}")


def create_demo_scenario(scenario: str, H: int, W: int, config: RiskConfig) -> np.ndarray:
    """
    Create synthetic occlusion mask for demo scenarios

    Args:
        scenario: Scenario name
        H: Height of BEV grid
        W: Width of BEV grid
        config: Risk configuration

    Returns:
        Occlusion mask [H, W]
    """
    occlusion_mask = np.zeros((H, W), dtype=np.float32)

    cx, cy = W // 2, H // 2

    if scenario == "Simple Occlusion":
        # Single large occluding object
        occlusion_mask[cy - 20:cy + 20, cx + 10:cx + 30] = 1.0

    elif scenario == "Multi-Vehicle Intersection":
        # Multiple vehicles creating occlusions
        occlusion_mask[cy - 30:cy - 10, cx + 20:cx + 40] = 1.0
        occlusion_mask[cy + 10:cy + 30, cx - 40:cx - 20] = 1.0
        occlusion_mask[cy - 10:cy + 10, cx + 50:cx + 70] = 1.0

    elif scenario == "Parking Lot Exit":
        # Parked cars on both sides
        for i in range(5):
            y_pos = cy + 20 + i * 25
            if y_pos < H - 10:
                occlusion_mask[y_pos:y_pos + 15, cx - 25:cx - 15] = 1.0
                occlusion_mask[y_pos:y_pos + 15, cx + 15:cx + 25] = 1.0

    elif scenario == "Highway Merge":
        # Vehicle on merge lane
        occlusion_mask[cy - 40:cy - 20, cx + 30:cx + 50] = 1.0
        occlusion_mask[cy + 20:cy + 40, cx + 20:cx + 40] = 1.0

    elif scenario == "Pedestrian Crossing":
        # Building/wall creating occlusion
        occlusion_mask[cy - 50:cy + 50, cx + 15:cx + 25] = 1.0

    return occlusion_mask


def export_visualization(risk_results: Dict, format_type: str, scenario_name: str):
    """Export visualization in specified format"""
    output_dir = Path("exports")
    output_dir.mkdir(exist_ok=True)

    if format_type == "PNG Image":
        output_path = output_dir / f"{scenario_name}_risk_map.png"
        fig, ax = plt.subplots(figsize=(10, 10))
        st.session_state.visualizer.plot_risk_heatmap(
            risk_results['risk_map'],
            ax=ax
        )
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        st.success(f"✓ Exported to {output_path}")

    elif format_type == "NumPy Array (.npy)":
        output_path = output_dir / f"{scenario_name}_risk_map.npy"
        np.save(output_path, risk_results['risk_map'])
        st.success(f"✓ Exported to {output_path}")

    elif format_type == "CSV Data":
        output_path = output_dir / f"{scenario_name}_risk_map.csv"
        np.savetxt(output_path, risk_results['risk_map'], delimiter=',')
        st.success(f"✓ Exported to {output_path}")

    elif format_type == "PDF Report":
        st.info("PDF report generation coming soon!")


if __name__ == "__main__":
    main()
