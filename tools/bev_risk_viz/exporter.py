#!/usr/bin/env python3
"""
BEV Risk Map Generator - Export Functionality

Handles exporting risk visualizations and data in various formats:
- PNG images
- NumPy arrays (.npy)
- CSV data files
- PDF reports with comprehensive analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime
import csv


class RiskDataExporter:
    """
    Exporter for risk map data and visualizations
    """

    def __init__(self, output_dir: str = "exports"):
        """
        Initialize exporter

        Args:
            output_dir: Directory for exported files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_png(
        self,
        risk_map: np.ndarray,
        filename: str,
        visualizer=None,
        bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50),
        dpi: int = 150,
        title: str = "BEV Risk Map"
    ) -> str:
        """
        Export risk map as PNG image

        Args:
            risk_map: Risk map array [H, W]
            filename: Output filename (without extension)
            visualizer: RiskVisualizer instance (creates default if None)
            bev_range: BEV range (x_min, x_max, y_min, y_max)
            dpi: DPI for output image
            title: Title for the visualization

        Returns:
            Path to exported file
        """
        if visualizer is None:
            from tools.bev_risk_viz.visualizer import RiskVisualizer
            visualizer = RiskVisualizer()

        output_path = os.path.join(self.output_dir, f"{filename}.png")

        fig, ax = plt.subplots(figsize=(12, 10))
        visualizer.plot_risk_heatmap(
            risk_map,
            bev_range=bev_range,
            title=title,
            show_colorbar=True,
            show_grid=True,
            ax=ax
        )

        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        print(f"✓ Exported PNG to {output_path}")
        return output_path

    def export_numpy(
        self,
        risk_results: Dict[str, np.ndarray],
        filename: str,
        include_factors: bool = True
    ) -> str:
        """
        Export risk data as NumPy array file

        Args:
            risk_results: Dictionary with risk map and factors
            filename: Output filename (without extension)
            include_factors: Whether to include individual factors

        Returns:
            Path to exported file
        """
        output_path = os.path.join(self.output_dir, f"{filename}.npz")

        if include_factors:
            np.savez_compressed(
                output_path,
                risk_map=risk_results['risk_map'],
                theta=risk_results.get('theta'),
                O=risk_results.get('O'),
                T=risk_results.get('T'),
                P=risk_results.get('P')
            )
        else:
            np.save(
                os.path.join(self.output_dir, f"{filename}.npy"),
                risk_results['risk_map']
            )
            output_path = os.path.join(self.output_dir, f"{filename}.npy")

        print(f"✓ Exported NumPy data to {output_path}")
        return output_path

    def export_csv(
        self,
        risk_map: np.ndarray,
        filename: str,
        include_coordinates: bool = True,
        bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50)
    ) -> str:
        """
        Export risk map as CSV file

        Args:
            risk_map: Risk map array [H, W]
            filename: Output filename (without extension)
            include_coordinates: Whether to include spatial coordinates
            bev_range: BEV range (x_min, x_max, y_min, y_max)

        Returns:
            Path to exported file
        """
        output_path = os.path.join(self.output_dir, f"{filename}.csv")

        H, W = risk_map.shape
        x_min, x_max, y_min, y_max = bev_range

        with open(output_path, 'w', newline='') as csvfile:
            if include_coordinates:
                writer = csv.writer(csvfile)
                writer.writerow(['X', 'Y', 'Risk'])

                # Generate coordinates
                x_coords = np.linspace(x_min, x_max, W)
                y_coords = np.linspace(y_min, y_max, H)

                for i in range(H):
                    for j in range(W):
                        writer.writerow([
                            f"{x_coords[j]:.2f}",
                            f"{y_coords[i]:.2f}",
                            f"{risk_map[i, j]:.6f}"
                        ])
            else:
                # Just save the array
                np.savetxt(csvfile, risk_map, delimiter=',', fmt='%.6f')

        print(f"✓ Exported CSV to {output_path}")
        return output_path

    def export_pdf_report(
        self,
        risk_results: Dict[str, np.ndarray],
        filename: str,
        visualizer=None,
        config: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50)
    ) -> str:
        """
        Export comprehensive PDF report with analysis

        Args:
            risk_results: Dictionary with risk map and factors
            filename: Output filename (without extension)
            visualizer: RiskVisualizer instance
            config: Configuration parameters used
            metadata: Additional metadata (scene name, timestamp, etc.)
            bev_range: BEV range (x_min, x_max, y_min, y_max)

        Returns:
            Path to exported file
        """
        if visualizer is None:
            from tools.bev_risk_viz.visualizer import RiskVisualizer
            visualizer = RiskVisualizer()

        output_path = os.path.join(self.output_dir, f"{filename}.pdf")

        with PdfPages(output_path) as pdf:
            # Page 1: Title and metadata
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.5, 0.9, 'BEV Risk Map Analysis Report',
                    ha='center', va='top', fontsize=24, fontweight='bold')

            fig.text(0.5, 0.8, 'Occlusion-Based Emergence Risk Assessment',
                    ha='center', va='top', fontsize=16)

            # Metadata
            metadata_text = f"""
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Tool: BEV-RiskViz (BEV Risk Visualization Tool)
Purpose: Autonomous driving risk assessment based on occlusion analysis
"""

            if metadata:
                metadata_text += "\nScenario Information:\n"
                for key, value in metadata.items():
                    metadata_text += f"  {key}: {value}\n"

            fig.text(0.1, 0.6, metadata_text, fontsize=11, family='monospace',
                    verticalalignment='top')

            # Configuration
            if config:
                config_text = "\nRisk Calculation Parameters:\n"
                for key, value in config.items():
                    config_text += f"  {key}: {value}\n"
                fig.text(0.1, 0.3, config_text, fontsize=10, family='monospace',
                        verticalalignment='top')

            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Page 2: Main risk map
            fig, ax = plt.subplots(figsize=(11, 9))
            visualizer.plot_risk_heatmap(
                risk_results['risk_map'],
                bev_range=bev_range,
                title='BEV Risk Heatmap',
                show_colorbar=True,
                show_grid=True,
                ax=ax
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Page 3: Factor breakdown
            fig = visualizer.plot_factor_breakdown(
                risk_results,
                bev_range=bev_range,
                title='Risk Factor Analysis'
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Page 4: Statistical analysis
            fig = self._create_statistics_page(risk_results)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Set PDF metadata
            d = pdf.infodict()
            d['Title'] = 'BEV Risk Map Analysis Report'
            d['Author'] = 'BEV-RiskViz'
            d['Subject'] = 'Autonomous Driving Risk Assessment'
            d['Keywords'] = 'BEV, Risk Map, Occlusion, Autonomous Driving'
            d['CreationDate'] = datetime.now()

        print(f"✓ Exported PDF report to {output_path}")
        return output_path

    def _create_statistics_page(self, risk_results: Dict[str, np.ndarray]) -> plt.Figure:
        """
        Create statistics page for PDF report

        Args:
            risk_results: Dictionary with risk map and factors

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(11, 8.5))

        # Title
        fig.text(0.5, 0.95, 'Statistical Analysis',
                ha='center', fontsize=18, fontweight='bold')

        risk_map = risk_results['risk_map']

        # Basic statistics
        stats_text = f"""
OVERALL RISK STATISTICS
{'=' * 50}

Risk Map Dimensions: {risk_map.shape[0]} × {risk_map.shape[1]} cells
Total Cells: {risk_map.size:,}

Risk Value Statistics:
  Maximum Risk:      {risk_map.max():.6f}
  Minimum Risk:      {risk_map.min():.6f}
  Mean Risk:         {risk_map.mean():.6f}
  Median Risk:       {np.median(risk_map):.6f}
  Std Deviation:     {risk_map.std():.6f}

Risk Distribution:
  High Risk (>0.7):   {(risk_map > 0.7).sum():6,} cells ({(risk_map > 0.7).sum() / risk_map.size * 100:5.2f}%)
  Medium Risk (0.3-0.7): {((risk_map >= 0.3) & (risk_map <= 0.7)).sum():6,} cells ({((risk_map >= 0.3) & (risk_map <= 0.7)).sum() / risk_map.size * 100:5.2f}%)
  Low Risk (<0.3):    {(risk_map < 0.3).sum():6,} cells ({(risk_map < 0.3).sum() / risk_map.size * 100:5.2f}%)

Percentiles:
  25th percentile:   {np.percentile(risk_map, 25):.6f}
  50th percentile:   {np.percentile(risk_map, 50):.6f}
  75th percentile:   {np.percentile(risk_map, 75):.6f}
  95th percentile:   {np.percentile(risk_map, 95):.6f}
  99th percentile:   {np.percentile(risk_map, 99):.6f}
"""

        fig.text(0.1, 0.85, stats_text, fontsize=10, family='monospace',
                verticalalignment='top')

        # Risk distribution histogram
        ax1 = fig.add_subplot(2, 2, 3)
        ax1.hist(risk_map.flatten(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.set_xlabel('Risk Level', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title('Risk Distribution Histogram', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Risk distribution by region
        ax2 = fig.add_subplot(2, 2, 4)
        risk_categories = ['Low\n(<0.3)', 'Medium\n(0.3-0.7)', 'High\n(>0.7)']
        risk_counts = [
            (risk_map < 0.3).sum(),
            ((risk_map >= 0.3) & (risk_map <= 0.7)).sum(),
            (risk_map > 0.7).sum()
        ]
        colors_bar = ['green', 'yellow', 'red']
        ax2.bar(risk_categories, risk_counts, color=colors_bar, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Number of Cells', fontsize=10)
        ax2.set_title('Risk Category Distribution', fontsize=11, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        return fig

    def export_animation_frames(
        self,
        risk_maps: List[np.ndarray],
        filename_prefix: str,
        visualizer=None,
        bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50),
        dpi: int = 100
    ) -> List[str]:
        """
        Export sequence of risk maps as individual PNG frames

        Args:
            risk_maps: List of risk map arrays
            filename_prefix: Prefix for output filenames
            visualizer: RiskVisualizer instance
            bev_range: BEV range
            dpi: DPI for output images

        Returns:
            List of paths to exported files
        """
        if visualizer is None:
            from tools.bev_risk_viz.visualizer import RiskVisualizer
            visualizer = RiskVisualizer()

        output_paths = []

        for i, risk_map in enumerate(risk_maps):
            filename = f"{filename_prefix}_frame_{i:04d}"
            output_path = self.export_png(
                risk_map,
                filename,
                visualizer=visualizer,
                bev_range=bev_range,
                dpi=dpi,
                title=f'Frame {i + 1}/{len(risk_maps)}'
            )
            output_paths.append(output_path)

        print(f"✓ Exported {len(risk_maps)} animation frames")
        return output_paths

    def export_batch(
        self,
        risk_results_list: List[Dict],
        base_filename: str,
        formats: List[str] = ['png', 'npy'],
        visualizer=None
    ) -> Dict[str, List[str]]:
        """
        Export multiple risk maps in batch

        Args:
            risk_results_list: List of risk result dictionaries
            base_filename: Base filename for outputs
            formats: List of formats to export ('png', 'npy', 'csv', 'pdf')
            visualizer: RiskVisualizer instance

        Returns:
            Dictionary mapping format to list of output paths
        """
        output_paths = {fmt: [] for fmt in formats}

        for i, risk_results in enumerate(risk_results_list):
            filename = f"{base_filename}_{i:04d}"

            if 'png' in formats:
                path = self.export_png(
                    risk_results['risk_map'],
                    filename,
                    visualizer=visualizer
                )
                output_paths['png'].append(path)

            if 'npy' in formats:
                path = self.export_numpy(risk_results, filename)
                output_paths['npy'].append(path)

            if 'csv' in formats:
                path = self.export_csv(risk_results['risk_map'], filename)
                output_paths['csv'].append(path)

            if 'pdf' in formats:
                path = self.export_pdf_report(
                    risk_results,
                    filename,
                    visualizer=visualizer,
                    metadata={'Batch Index': i}
                )
                output_paths['pdf'].append(path)

        print(f"✓ Batch export completed: {len(risk_results_list)} items")
        return output_paths
