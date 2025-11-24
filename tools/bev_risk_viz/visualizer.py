#!/usr/bin/env python3
"""
BEV Risk Map Generator - Visualization Module

Handles all visualization outputs including BEV risk heatmaps,
occlusion overlays, trajectory predictions, and animations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
import cv2
from typing import Dict, List, Optional, Tuple
import os


class RiskVisualizer:
    """
    Visualizer for BEV risk maps and related outputs
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 10),
        dpi: int = 150,
        style: str = 'dark_background'
    ):
        """
        Initialize the visualizer

        Args:
            figsize: Figure size in inches
            dpi: DPI for output images
            style: Matplotlib style ('dark_background' or 'default')
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = style

        # Create custom colormap (green -> yellow -> red)
        self.risk_cmap = self._create_risk_colormap()

    def _create_risk_colormap(self) -> LinearSegmentedColormap:
        """
        Create custom colormap for risk visualization
        Green (low) -> Yellow (medium) -> Red (high)

        Returns:
            Custom colormap
        """
        colors = [
            (0.0, '#00FF00'),  # Green (no risk)
            (0.3, '#7FFF00'),  # Yellow-green
            (0.5, '#FFFF00'),  # Yellow
            (0.7, '#FF7F00'),  # Orange
            (1.0, '#FF0000'),  # Red (high risk)
        ]

        cmap = LinearSegmentedColormap.from_list(
            'risk_map',
            [(pos, color) for pos, color in colors]
        )
        return cmap

    def plot_risk_heatmap(
        self,
        risk_map: np.ndarray,
        bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50),
        title: str = 'BEV Risk Map',
        show_colorbar: bool = True,
        show_grid: bool = True,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot BEV risk heatmap

        Args:
            risk_map: Risk map array [H, W]
            bev_range: BEV range (x_min, x_max, y_min, y_max) in meters
            title: Plot title
            show_colorbar: Whether to show colorbar
            show_grid: Whether to show grid lines
            ax: Matplotlib axes (creates new if None)

        Returns:
            Matplotlib axes object
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=self.figsize)

        x_min, x_max, y_min, y_max = bev_range

        # Plot risk map
        im = ax.imshow(
            risk_map,
            cmap=self.risk_cmap,
            vmin=0.0,
            vmax=1.0,
            extent=[x_min, x_max, y_min, y_max],
            origin='lower',
            interpolation='bilinear',
            alpha=0.9
        )

        # Add ego vehicle marker at origin
        ax.plot(0, 0, 'w*', markersize=20, markeredgecolor='black', markeredgewidth=1.5)
        ax.text(0, -2, 'EGO', ha='center', va='top', color='white',
                fontweight='bold', fontsize=10)

        # Styling
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')

        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Risk Level', rotation=270, labelpad=20, fontsize=11)

        return ax

    def plot_occlusion_overlay(
        self,
        risk_map: np.ndarray,
        occlusion_mask: np.ndarray,
        bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50),
        title: str = 'Risk Map with Occlusion Overlay',
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot risk map with occlusion regions overlaid

        Args:
            risk_map: Risk map array [H, W]
            occlusion_mask: Binary occlusion mask [H, W]
            bev_range: BEV range (x_min, x_max, y_min, y_max) in meters
            title: Plot title
            ax: Matplotlib axes (creates new if None)

        Returns:
            Matplotlib axes object
        """
        ax = self.plot_risk_heatmap(risk_map, bev_range, title, True, True, ax)

        # Overlay occlusion regions with blue transparent mask
        x_min, x_max, y_min, y_max = bev_range
        occlusion_rgba = np.zeros((*occlusion_mask.shape, 4))
        occlusion_rgba[..., 2] = occlusion_mask  # Blue channel
        occlusion_rgba[..., 3] = occlusion_mask * 0.3  # Alpha channel

        ax.imshow(
            occlusion_rgba,
            extent=[x_min, x_max, y_min, y_max],
            origin='lower',
            interpolation='nearest'
        )

        return ax

    def plot_factor_breakdown(
        self,
        risk_results: Dict[str, np.ndarray],
        bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50),
        title: str = 'Risk Factor Breakdown'
    ) -> plt.Figure:
        """
        Plot all risk factors in a grid layout

        Args:
            risk_results: Dictionary with 'risk_map', 'theta', 'O', 'T', 'P'
            bev_range: BEV range (x_min, x_max, y_min, y_max) in meters
            title: Overall title

        Returns:
            Matplotlib figure
        """
        with plt.style.context(self.style):
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(title, fontsize=16, fontweight='bold')

            factors = [
                ('risk_map', 'Final Risk Map', self.risk_cmap),
                ('theta', 'Trajectory Alignment (θ)', 'viridis'),
                ('O', 'Occlusion Severity (O)', 'Blues'),
                ('T', 'Temporal Urgency (T)', 'YlOrRd'),
                ('P', 'Proximity (P)', 'plasma'),
            ]

            x_min, x_max, y_min, y_max = bev_range

            for idx, (key, factor_title, cmap) in enumerate(factors):
                row = idx // 3
                col = idx % 3
                ax = axes[row, col]

                if key in risk_results:
                    data = risk_results[key]
                    im = ax.imshow(
                        data,
                        cmap=cmap,
                        vmin=0.0,
                        vmax=1.0,
                        extent=[x_min, x_max, y_min, y_max],
                        origin='lower',
                        interpolation='bilinear'
                    )

                    # Add ego marker
                    ax.plot(0, 0, 'w*', markersize=15, markeredgecolor='black')

                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Y (m)')
                    ax.set_title(factor_title, fontweight='bold')
                    ax.set_aspect('equal')
                    ax.grid(True, alpha=0.2, linestyle='--')

                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Use last subplot for statistics
            ax_stats = axes[1, 2]
            ax_stats.axis('off')

            # Display statistics
            stats_text = "Risk Statistics:\n\n"
            risk_map = risk_results['risk_map']
            stats_text += f"Max Risk: {risk_map.max():.3f}\n"
            stats_text += f"Mean Risk: {risk_map.mean():.4f}\n"
            stats_text += f"Std Dev: {risk_map.std():.4f}\n"
            stats_text += f"High Risk Area (>0.7): {(risk_map > 0.7).sum() / risk_map.size * 100:.1f}%\n"
            stats_text += f"Medium Risk (0.3-0.7): {((risk_map >= 0.3) & (risk_map <= 0.7)).sum() / risk_map.size * 100:.1f}%\n"
            stats_text += f"Low Risk (<0.3): {(risk_map < 0.3).sum() / risk_map.size * 100:.1f}%\n"

            ax_stats.text(
                0.1, 0.5, stats_text,
                fontsize=12,
                verticalalignment='center',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            )

            plt.tight_layout()

        return fig

    def plot_with_objects(
        self,
        risk_map: np.ndarray,
        annotations: List[Dict],
        bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50),
        title: str = 'Risk Map with Objects',
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot risk map with object bounding boxes

        Args:
            risk_map: Risk map array [H, W]
            annotations: List of object annotations
            bev_range: BEV range (x_min, x_max, y_min, y_max) in meters
            title: Plot title
            ax: Matplotlib axes (creates new if None)

        Returns:
            Matplotlib axes object
        """
        ax = self.plot_risk_heatmap(risk_map, bev_range, title, True, True, ax)

        # Color mapping for different object categories
        category_colors = {
            'vehicle': 'cyan',
            'pedestrian': 'yellow',
            'bicycle': 'magenta',
            'motorcycle': 'orange',
            'default': 'white'
        }

        # Draw object bounding boxes
        for ann in annotations:
            x, y, _ = ann['translation']
            w, l, h = ann['size']

            # Determine color based on category
            color = 'white'
            for cat_key, cat_color in category_colors.items():
                if cat_key in ann['category']:
                    color = cat_color
                    break

            # Draw rectangle (simplified, not considering rotation)
            rect = patches.Rectangle(
                (x - w/2, y - l/2),
                w, l,
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)

            # Add label
            category_short = ann['category'].split('.')[-1]
            ax.text(
                x, y + l/2 + 1,
                category_short,
                color=color,
                fontsize=8,
                ha='center',
                va='bottom',
                fontweight='bold'
            )

        return ax

    def create_animation(
        self,
        risk_maps: List[np.ndarray],
        bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50),
        output_path: str = 'risk_animation.gif',
        fps: int = 2,
        title_prefix: str = 'Frame'
    ):
        """
        Create animated GIF of risk maps over time

        Args:
            risk_maps: List of risk map arrays
            bev_range: BEV range (x_min, x_max, y_min, y_max) in meters
            output_path: Output file path
            fps: Frames per second
            title_prefix: Prefix for frame titles
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)

            def update(frame_idx):
                ax.clear()
                self.plot_risk_heatmap(
                    risk_maps[frame_idx],
                    bev_range,
                    title=f'{title_prefix} {frame_idx + 1}/{len(risk_maps)}',
                    show_colorbar=True,
                    show_grid=True,
                    ax=ax
                )

            anim = FuncAnimation(
                fig,
                update,
                frames=len(risk_maps),
                interval=1000 // fps,
                repeat=True
            )

            # Save as GIF
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer, dpi=self.dpi)
            plt.close()

            print(f"✓ Animation saved to {output_path}")

    def save_figure(self, fig: plt.Figure, output_path: str):
        """
        Save figure to file

        Args:
            fig: Matplotlib figure
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved visualization to {output_path}")

    def plot_comparison(
        self,
        risk_map1: np.ndarray,
        risk_map2: np.ndarray,
        labels: Tuple[str, str] = ('Configuration 1', 'Configuration 2'),
        bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50),
        title: str = 'Risk Map Comparison'
    ) -> plt.Figure:
        """
        Plot side-by-side comparison of two risk maps

        Args:
            risk_map1: First risk map
            risk_map2: Second risk map
            labels: Labels for the two maps
            bev_range: BEV range (x_min, x_max, y_min, y_max) in meters
            title: Overall title

        Returns:
            Matplotlib figure
        """
        with plt.style.context(self.style):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(title, fontsize=16, fontweight='bold')

            # Plot first map
            self.plot_risk_heatmap(
                risk_map1,
                bev_range,
                labels[0],
                show_colorbar=True,
                ax=axes[0]
            )

            # Plot second map
            self.plot_risk_heatmap(
                risk_map2,
                bev_range,
                labels[1],
                show_colorbar=True,
                ax=axes[1]
            )

            # Plot difference
            diff = risk_map2 - risk_map1
            x_min, x_max, y_min, y_max = bev_range

            im = axes[2].imshow(
                diff,
                cmap='RdBu_r',
                vmin=-1.0,
                vmax=1.0,
                extent=[x_min, x_max, y_min, y_max],
                origin='lower',
                interpolation='bilinear'
            )

            axes[2].plot(0, 0, 'k*', markersize=15)
            axes[2].set_xlabel('X (meters)')
            axes[2].set_ylabel('Y (meters)')
            axes[2].set_title('Difference (Map 2 - Map 1)', fontweight='bold')
            axes[2].set_aspect('equal')
            axes[2].grid(True, alpha=0.3, linestyle='--')

            cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            cbar.set_label('Risk Difference', rotation=270, labelpad=20)

            plt.tight_layout()

        return fig
