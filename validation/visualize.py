"""
Visualization Tools for Risk-Guided BEVFormer

Provides visualization for:
- Risk maps (GT vs Predicted)
- Detection results with risk overlay
- Attention weights
- Multi-view comparisons
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import cv2
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class RiskVisualizer:
    """Visualizer for risk maps and detection results"""

    def __init__(self, output_dir='visualizations/risk_guided_bevformer'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color maps
        self.risk_cmap = 'hot'  # Red for high risk
        self.attention_cmap = 'viridis'

    def visualize_risk_comparison(self, gt_risk, pred_risk, sample_token, save=True):
        """
        Visualize GT vs Predicted risk maps side by side.

        Args:
            gt_risk: Ground truth risk map [200, 200] or [1, 200, 200]
            pred_risk: Predicted risk map [200, 200] or [1, 200, 200]
            sample_token: Sample identifier
            save: Whether to save figure
        """
        # Ensure 2D
        if isinstance(gt_risk, torch.Tensor):
            gt_risk = gt_risk.cpu().numpy()
        if isinstance(pred_risk, torch.Tensor):
            pred_risk = pred_risk.cpu().numpy()

        if gt_risk.ndim == 3:
            gt_risk = gt_risk[0]
        if pred_risk.ndim == 3:
            pred_risk = pred_risk[0]

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # GT Risk
        im1 = axes[0].imshow(gt_risk, cmap=self.risk_cmap, vmin=0, vmax=1)
        axes[0].set_title('Ground Truth Risk Map', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X (pixels)')
        axes[0].set_ylabel('Y (pixels)')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(im1, ax=axes[0], label='Risk [0, 1]')

        # Add ego vehicle marker
        axes[0].scatter([100], [100], c='cyan', s=100, marker='o',
                       edgecolors='white', linewidths=2, label='Ego Vehicle')
        axes[0].legend()

        # Predicted Risk
        im2 = axes[1].imshow(pred_risk, cmap=self.risk_cmap, vmin=0, vmax=1)
        axes[1].set_title('Predicted Risk Map', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X (pixels)')
        axes[1].set_ylabel('Y (pixels)')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(im2, ax=axes[1], label='Risk [0, 1]')

        axes[1].scatter([100], [100], c='cyan', s=100, marker='o',
                       edgecolors='white', linewidths=2, label='Ego Vehicle')
        axes[1].legend()

        # Difference Map
        diff = np.abs(gt_risk - pred_risk)
        im3 = axes[2].imshow(diff, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[2].set_title('Absolute Difference', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('X (pixels)')
        axes[2].set_ylabel('Y (pixels)')
        axes[2].grid(True, alpha=0.3)
        plt.colorbar(im3, ax=axes[2], label='|GT - Pred|')

        # Add metrics
        mae = diff.mean()
        mse = (diff ** 2).mean()
        max_error = diff.max()

        metrics_text = f'MAE: {mae:.4f}\nMSE: {mse:.4f}\nMax Error: {max_error:.4f}'
        axes[2].text(0.02, 0.98, metrics_text, transform=axes[2].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10, family='monospace')

        plt.suptitle(f'Risk Map Comparison - Sample: {sample_token[:8]}...',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save:
            save_path = self.output_dir / f'risk_comparison_{sample_token[:8]}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def visualize_risk_with_detections(self, risk_map, bboxes_3d, scores, labels,
                                       sample_token, title='Risk Map with Detections', save=True):
        """
        Visualize risk map with 3D bounding boxes projected to BEV.

        Args:
            risk_map: Risk map [200, 200]
            bboxes_3d: 3D bounding boxes in BEV coordinates
            scores: Detection scores
            labels: Class labels
            sample_token: Sample identifier
            title: Plot title
            save: Whether to save
        """
        if isinstance(risk_map, torch.Tensor):
            risk_map = risk_map.cpu().numpy()
        if risk_map.ndim == 3:
            risk_map = risk_map[0]

        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot risk map
        im = ax.imshow(risk_map, cmap=self.risk_cmap, vmin=0, vmax=1, alpha=0.7)
        plt.colorbar(im, ax=ax, label='Risk [0, 1]')

        # Overlay detections
        # Convert from meters to pixels
        # BEV range: [-50, 50] meters, 200 pixels -> 0.5m per pixel
        def meters_to_pixels(x, y):
            px = (x + 50) / 0.5
            py = (y + 50) / 0.5
            return px, py

        # Draw bounding boxes
        if bboxes_3d is not None and len(bboxes_3d) > 0:
            for bbox, score, label in zip(bboxes_3d, scores, labels):
                if score < 0.3:  # Skip low-confidence detections
                    continue

                # bbox: [x, y, z, l, w, h, theta, ...]
                cx, cy = bbox[0], bbox[1]
                length, width = bbox[3], bbox[4]
                theta = bbox[6] if len(bbox) > 6 else 0

                # Convert to pixels
                cx_px, cy_px = meters_to_pixels(cx, cy)
                l_px = length / 0.5
                w_px = width / 0.5

                # Create rectangle
                rect = patches.Rectangle(
                    (cx_px - l_px/2, cy_px - w_px/2),
                    l_px, w_px,
                    angle=np.degrees(theta),
                    linewidth=2,
                    edgecolor='lime',
                    facecolor='none',
                    label=f'Class {label}'
                )
                ax.add_patch(rect)

                # Add score text
                ax.text(cx_px, cy_px, f'{score:.2f}',
                       color='white', fontsize=8, ha='center',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        # Add ego vehicle
        ax.scatter([100], [100], c='cyan', s=200, marker='o',
                  edgecolors='white', linewidths=3, label='Ego Vehicle', zorder=10)

        # Add forward direction arrow
        ax.arrow(100, 100, 0, -30, head_width=5, head_length=8,
                fc='cyan', ec='white', linewidth=2, zorder=10)

        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        plt.tight_layout()

        if save:
            save_path = self.output_dir / f'risk_detections_{sample_token[:8]}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def visualize_attention_weights(self, attention_weights, risk_map, sample_token, save=True):
        """
        Visualize attention weights alongside risk map.

        Args:
            attention_weights: Attention weight map [50, 50] or [1, 50, 50]
            risk_map: Risk map [200, 200]
            sample_token: Sample identifier
            save: Whether to save
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()
        if isinstance(risk_map, torch.Tensor):
            risk_map = risk_map.cpu().numpy()

        if attention_weights.ndim == 3:
            attention_weights = attention_weights[0]
        if risk_map.ndim == 3:
            risk_map = risk_map[0]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Attention weights
        im1 = axes[0].imshow(attention_weights, cmap=self.attention_cmap)
        axes[0].set_title('Risk-Guided Attention Weights', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X (BEV grid)')
        axes[0].set_ylabel('Y (BEV grid)')
        plt.colorbar(im1, ax=axes[0], label='Attention Weight')

        # Risk map
        im2 = axes[1].imshow(risk_map, cmap=self.risk_cmap, vmin=0, vmax=1)
        axes[1].set_title('Risk Map', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X (pixels)')
        axes[1].set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=axes[1], label='Risk [0, 1]')

        plt.suptitle(f'Attention Analysis - Sample: {sample_token[:8]}...',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save:
            save_path = self.output_dir / f'attention_{sample_token[:8]}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def visualize_multi_sample_comparison(self, samples, save=True):
        """
        Visualize multiple samples in a grid.

        Args:
            samples: List of dicts with keys:
                - 'gt_risk': Ground truth risk map
                - 'pred_risk': Predicted risk map
                - 'sample_token': Sample ID
            save: Whether to save
        """
        num_samples = len(samples)
        fig = plt.figure(figsize=(18, 6 * num_samples))
        gs = GridSpec(num_samples, 3, figure=fig, hspace=0.3, wspace=0.3)

        for i, sample in enumerate(samples):
            gt_risk = sample['gt_risk']
            pred_risk = sample['pred_risk']
            token = sample['sample_token']

            # Ensure numpy
            if isinstance(gt_risk, torch.Tensor):
                gt_risk = gt_risk.cpu().numpy()
            if isinstance(pred_risk, torch.Tensor):
                pred_risk = pred_risk.cpu().numpy()

            if gt_risk.ndim == 3:
                gt_risk = gt_risk[0]
            if pred_risk.ndim == 3:
                pred_risk = pred_risk[0]

            # GT
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.imshow(gt_risk, cmap=self.risk_cmap, vmin=0, vmax=1)
            ax1.set_title(f'GT - {token[:8]}...', fontsize=12)
            ax1.axis('off')

            # Pred
            ax2 = fig.add_subplot(gs[i, 1])
            ax2.imshow(pred_risk, cmap=self.risk_cmap, vmin=0, vmax=1)
            ax2.set_title(f'Pred - {token[:8]}...', fontsize=12)
            ax2.axis('off')

            # Diff
            ax3 = fig.add_subplot(gs[i, 2])
            diff = np.abs(gt_risk - pred_risk)
            im = ax3.imshow(diff, cmap='RdYlGn_r', vmin=0, vmax=1)
            ax3.set_title(f'Diff (MAE={diff.mean():.4f})', fontsize=12)
            ax3.axis('off')
            plt.colorbar(im, ax=ax3, fraction=0.046)

        plt.suptitle('Multi-Sample Risk Comparison', fontsize=16, fontweight='bold')

        if save:
            save_path = self.output_dir / 'multi_sample_comparison.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_metrics_over_time(self, metrics_history, save=True):
        """
        Plot training metrics over time.

        Args:
            metrics_history: Dict with keys like 'loss_risk', 'loss_cls', etc.
                Each value is a list of values over iterations/epochs.
            save: Whether to save
        """
        num_metrics = len(metrics_history)
        fig, axes = plt.subplots((num_metrics + 1) // 2, 2, figsize=(16, 4 * ((num_metrics + 1) // 2)))
        axes = axes.flatten() if num_metrics > 1 else [axes]

        for idx, (metric_name, values) in enumerate(metrics_history.items()):
            axes[idx].plot(values, linewidth=2)
            axes[idx].set_title(metric_name, fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Iteration')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(num_metrics, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Training Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save:
            save_path = self.output_dir / 'metrics_over_time.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig


def test_visualizer():
    """Test the visualizer with synthetic data"""
    print("\n" + "="*80)
    print("TEST: Visualizer")
    print("="*80)

    visualizer = RiskVisualizer(output_dir='visualizations/test')

    # Create synthetic data
    gt_risk = np.random.rand(200, 200) * 0.8
    pred_risk = gt_risk + np.random.randn(200, 200) * 0.1
    pred_risk = np.clip(pred_risk, 0, 1)

    sample_token = 'test_sample_12345'

    print("\n📋 Testing risk comparison...")
    fig1 = visualizer.visualize_risk_comparison(gt_risk, pred_risk, sample_token)
    plt.close(fig1)

    print("\n📋 Testing risk with detections...")
    # Fake detections
    bboxes_3d = np.array([
        [10, 5, 0, 4, 2, 2, 0.5],  # x, y, z, l, w, h, theta
        [-10, -5, 0, 3, 2, 2, -0.3],
    ])
    scores = np.array([0.9, 0.7])
    labels = np.array([1, 2])

    fig2 = visualizer.visualize_risk_with_detections(
        gt_risk, bboxes_3d, scores, labels, sample_token
    )
    plt.close(fig2)

    print("\n📋 Testing multi-sample comparison...")
    samples = [
        {'gt_risk': gt_risk, 'pred_risk': pred_risk, 'sample_token': f'sample_{i}'}
        for i in range(3)
    ]
    fig3 = visualizer.visualize_multi_sample_comparison(samples)
    plt.close(fig3)

    print("\n✅ Visualizer Test PASSED!\n")
    return True


if __name__ == '__main__':
    test_visualizer()
