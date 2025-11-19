#!/usr/bin/env python3
"""
Risk Label Analysis Script

This script analyzes generated risk labels to provide statistics and insights
about the risk distribution, spatial patterns, and quality of the labels.
"""

import numpy as np
import pickle
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from typing import Dict, List
from collections import defaultdict


def load_labels(pkl_path: str) -> Dict:
    """Load labels from pickle file"""
    print(f"Loading labels from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        labels = pickle.load(f)
    return labels


def analyze_risk_distribution(labels_dict: Dict) -> Dict:
    """
    Analyze risk value distribution across all samples

    Args:
        labels_dict: Dict of scene_token -> list of labels

    Returns:
        Statistics dict
    """
    all_max_risks = []
    all_mean_risks = []
    all_high_risk_cells = []
    all_medium_risk_cells = []
    all_low_risk_cells = []
    all_risk_values = []

    total_samples = 0
    total_scenes = len(labels_dict)

    for scene_labels in labels_dict.values():
        for label in scene_labels:
            total_samples += 1

            # Collect metadata
            all_max_risks.append(label['metadata']['max_risk'])
            all_mean_risks.append(label['metadata']['mean_risk'])
            all_high_risk_cells.append(label['metadata']['high_risk_cells'])
            all_medium_risk_cells.append(label['metadata']['medium_risk_cells'])
            all_low_risk_cells.append(label['metadata']['low_risk_cells'])

            # Collect all risk values for histogram
            risk_map = label['risk_map']
            all_risk_values.extend(risk_map.flatten())

    stats = {
        'total_scenes': total_scenes,
        'total_samples': total_samples,

        # Max risk statistics
        'max_risk_mean': float(np.mean(all_max_risks)),
        'max_risk_std': float(np.std(all_max_risks)),
        'max_risk_min': float(np.min(all_max_risks)),
        'max_risk_max': float(np.max(all_max_risks)),
        'max_risk_median': float(np.median(all_max_risks)),

        # Mean risk statistics
        'mean_risk_mean': float(np.mean(all_mean_risks)),
        'mean_risk_std': float(np.std(all_mean_risks)),
        'mean_risk_min': float(np.min(all_mean_risks)),
        'mean_risk_max': float(np.max(all_mean_risks)),
        'mean_risk_median': float(np.median(all_mean_risks)),

        # Cell count statistics
        'high_risk_cells_mean': float(np.mean(all_high_risk_cells)),
        'high_risk_cells_std': float(np.std(all_high_risk_cells)),
        'medium_risk_cells_mean': float(np.mean(all_medium_risk_cells)),
        'medium_risk_cells_std': float(np.std(all_medium_risk_cells)),
        'low_risk_cells_mean': float(np.mean(all_low_risk_cells)),
        'low_risk_cells_std': float(np.std(all_low_risk_cells)),

        # Distribution
        'samples_with_max_risk_gt_0.7': int(sum(1 for r in all_max_risks if r > 0.7)),
        'samples_with_max_risk_gt_0.5': int(sum(1 for r in all_max_risks if r > 0.5)),
        'samples_with_max_risk_gt_0.3': int(sum(1 for r in all_max_risks if r > 0.3)),

        # Percentages
        'pct_samples_max_risk_gt_0.7': float(100 * sum(1 for r in all_max_risks if r > 0.7) / len(all_max_risks)),
        'pct_samples_max_risk_gt_0.5': float(100 * sum(1 for r in all_max_risks if r > 0.5) / len(all_max_risks)),
        'pct_samples_max_risk_gt_0.3': float(100 * sum(1 for r in all_max_risks if r > 0.3) / len(all_max_risks)),

        # Raw data for plotting
        'all_max_risks': all_max_risks,
        'all_mean_risks': all_mean_risks,
        'all_risk_values': np.array(all_risk_values),
    }

    return stats


def analyze_spatial_distribution(labels_dict: Dict) -> np.ndarray:
    """
    Analyze spatial distribution of risk across BEV

    Args:
        labels_dict: Dict of scene_token -> list of labels

    Returns:
        Average risk map across all samples
    """
    # Accumulate risk maps
    total_risk_map = None
    count = 0

    for scene_labels in labels_dict.values():
        for label in scene_labels:
            risk_map = label['risk_map']
            if total_risk_map is None:
                total_risk_map = np.zeros_like(risk_map, dtype=np.float64)
            total_risk_map += risk_map
            count += 1

    # Average
    avg_risk_map = total_risk_map / count

    return avg_risk_map


def find_top_scenes(labels_dict: Dict, n: int = 10) -> Dict:
    """
    Find scenes with highest average risk

    Args:
        labels_dict: Dict of scene_token -> list of labels
        n: Number of top scenes to return

    Returns:
        Dict with top scenes info
    """
    scene_stats = []

    for scene_token, scene_labels in labels_dict.items():
        if len(scene_labels) == 0:
            continue

        scene_name = scene_labels[0]['scene_name']
        avg_max_risk = np.mean([label['metadata']['max_risk'] for label in scene_labels])
        avg_mean_risk = np.mean([label['metadata']['mean_risk'] for label in scene_labels])
        avg_high_risk_cells = np.mean([label['metadata']['high_risk_cells'] for label in scene_labels])

        scene_stats.append({
            'scene_token': scene_token,
            'scene_name': scene_name,
            'num_samples': len(scene_labels),
            'avg_max_risk': avg_max_risk,
            'avg_mean_risk': avg_mean_risk,
            'avg_high_risk_cells': avg_high_risk_cells,
        })

    # Sort by avg_max_risk
    scene_stats.sort(key=lambda x: x['avg_max_risk'], reverse=True)

    return {
        'top_by_max_risk': scene_stats[:n],
        'bottom_by_max_risk': scene_stats[-n:],
    }


def plot_risk_distribution(stats: Dict, output_path: Path):
    """
    Plot risk distribution histograms

    Args:
        stats: Statistics dict from analyze_risk_distribution
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Max risk distribution
    ax = axes[0, 0]
    ax.hist(stats['all_max_risks'], bins=50, color='red', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Max Risk per Sample')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Maximum Risk Values')
    ax.axvline(stats['max_risk_mean'], color='blue', linestyle='--',
               label=f'Mean: {stats["max_risk_mean"]:.3f}')
    ax.axvline(stats['max_risk_median'], color='green', linestyle='--',
               label=f'Median: {stats["max_risk_median"]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Mean risk distribution
    ax = axes[0, 1]
    ax.hist(stats['all_mean_risks'], bins=50, color='orange', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Mean Risk per Sample')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Average Risk Values')
    ax.axvline(stats['mean_risk_mean'], color='blue', linestyle='--',
               label=f'Mean: {stats["mean_risk_mean"]:.3f}')
    ax.axvline(stats['mean_risk_median'], color='green', linestyle='--',
               label=f'Median: {stats["mean_risk_median"]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. All risk values (histogram)
    ax = axes[1, 0]
    # Only plot non-zero values for better visualization
    non_zero_risks = stats['all_risk_values'][stats['all_risk_values'] > 0]
    if len(non_zero_risks) > 0:
        ax.hist(non_zero_risks, bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Risk Value')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of All Non-Zero Risk Values')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No non-zero risk values', ha='center', va='center',
                transform=ax.transAxes)

    # 4. Risk value ranges (bar chart)
    ax = axes[1, 1]
    total_samples = stats['total_samples']
    categories = ['> 0.7', '> 0.5', '> 0.3']
    counts = [
        stats['samples_with_max_risk_gt_0.7'],
        stats['samples_with_max_risk_gt_0.5'],
        stats['samples_with_max_risk_gt_0.3'],
    ]
    percentages = [
        stats['pct_samples_max_risk_gt_0.7'],
        stats['pct_samples_max_risk_gt_0.5'],
        stats['pct_samples_max_risk_gt_0.3'],
    ]

    bars = ax.bar(categories, counts, color=['red', 'orange', 'yellow'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Max Risk Threshold')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Samples by Maximum Risk Level')

    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_spatial_heatmap(avg_risk_map: np.ndarray, output_path: Path):
    """
    Plot average spatial risk distribution as heatmap

    Args:
        avg_risk_map: Average risk map [200, 200]
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    im = ax.imshow(avg_risk_map, cmap='hot', vmin=0, vmax=avg_risk_map.max(),
                   extent=[-50, 50, -50, 50], origin='lower', aspect='equal')

    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Average Risk Distribution Across All Samples', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Average Risk', fontsize=11)

    # Mark ego position
    ax.plot(0, 0, 'b*', markersize=20, markeredgecolor='white', markeredgewidth=2,
            label='Ego Position')

    # Add grid
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)

    # Add legend
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_analysis_report(stats: Dict, top_scenes: Dict):
    """
    Print analysis report to console

    Args:
        stats: Statistics dict
        top_scenes: Top scenes dict
    """
    print("\n" + "=" * 80)
    print("RISK LABEL ANALYSIS REPORT")
    print("=" * 80)

    print(f"\nDataset Overview:")
    print(f"  Total scenes:  {stats['total_scenes']}")
    print(f"  Total samples: {stats['total_samples']}")

    print(f"\nRisk Statistics (per sample):")
    print(f"  Maximum Risk:")
    print(f"    Mean:   {stats['max_risk_mean']:.4f} ± {stats['max_risk_std']:.4f}")
    print(f"    Median: {stats['max_risk_median']:.4f}")
    print(f"    Range:  [{stats['max_risk_min']:.4f}, {stats['max_risk_max']:.4f}]")

    print(f"\n  Average Risk:")
    print(f"    Mean:   {stats['mean_risk_mean']:.4f} ± {stats['mean_risk_std']:.4f}")
    print(f"    Median: {stats['mean_risk_median']:.4f}")
    print(f"    Range:  [{stats['mean_risk_min']:.4f}, {stats['mean_risk_max']:.4f}]")

    print(f"\nHigh-Risk Cells (per sample):")
    print(f"  High (> 0.7):   {stats['high_risk_cells_mean']:.1f} ± {stats['high_risk_cells_std']:.1f}")
    print(f"  Medium (0.3-0.7): {stats['medium_risk_cells_mean']:.1f} ± {stats['medium_risk_cells_std']:.1f}")
    print(f"  Low (0.0-0.3):  {stats['low_risk_cells_mean']:.1f} ± {stats['low_risk_cells_std']:.1f}")

    print(f"\nRisk Distribution:")
    print(f"  Samples with max_risk > 0.7: {stats['samples_with_max_risk_gt_0.7']} ({stats['pct_samples_max_risk_gt_0.7']:.1f}%)")
    print(f"  Samples with max_risk > 0.5: {stats['samples_with_max_risk_gt_0.5']} ({stats['pct_samples_max_risk_gt_0.5']:.1f}%)")
    print(f"  Samples with max_risk > 0.3: {stats['samples_with_max_risk_gt_0.3']} ({stats['pct_samples_max_risk_gt_0.3']:.1f}%)")

    print(f"\nTop 5 Scenes by Average Max Risk:")
    for i, scene in enumerate(top_scenes['top_by_max_risk'][:5], 1):
        print(f"  {i}. {scene['scene_name']}")
        print(f"     Avg max risk: {scene['avg_max_risk']:.3f}")
        print(f"     Avg mean risk: {scene['avg_mean_risk']:.3f}")
        print(f"     Samples: {scene['num_samples']}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze risk labels')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to labels pickle file')
    parser.add_argument('--output_dir', type=str, default='analysis/risk_labels',
                        help='Output directory for analysis results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("RISK LABEL ANALYSIS")
    print("=" * 80)
    print(f"Labels:      {args.labels}")
    print(f"Output dir:  {output_dir}")
    print("=" * 80 + "\n")

    # Load labels
    labels_dict = load_labels(args.labels)

    # Analyze risk distribution
    print("Analyzing risk distribution...")
    stats = analyze_risk_distribution(labels_dict)

    # Analyze spatial distribution
    print("Analyzing spatial distribution...")
    avg_risk_map = analyze_spatial_distribution(labels_dict)

    # Find top scenes
    print("Finding top scenes...")
    top_scenes = find_top_scenes(labels_dict, n=10)

    # Save statistics
    print("Saving statistics...")
    # Remove raw data arrays before saving JSON
    stats_for_json = {k: v for k, v in stats.items()
                      if k not in ['all_max_risks', 'all_mean_risks', 'all_risk_values']}
    stats_path = output_dir / 'risk_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats_for_json, f, indent=2)
    print(f"✓ Saved statistics to {stats_path}")

    # Save top scenes
    top_scenes_path = output_dir / 'top_scenes.json'
    with open(top_scenes_path, 'w') as f:
        json.dump(top_scenes, f, indent=2)
    print(f"✓ Saved top scenes to {top_scenes_path}")

    # Save average risk map
    avg_risk_map_path = output_dir / 'avg_risk_map.npy'
    np.save(avg_risk_map_path, avg_risk_map)
    print(f"✓ Saved average risk map to {avg_risk_map_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_risk_distribution(stats, output_dir / 'risk_distribution.png')
    print(f"✓ Saved risk distribution plot to {output_dir / 'risk_distribution.png'}")

    plot_spatial_heatmap(avg_risk_map, output_dir / 'spatial_heatmap.png')
    print(f"✓ Saved spatial heatmap to {output_dir / 'spatial_heatmap.png'}")

    # Print report
    print_analysis_report(stats, top_scenes)

    print("\n✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - risk_statistics.json")
    print(f"  - top_scenes.json")
    print(f"  - avg_risk_map.npy")
    print(f"  - risk_distribution.png")
    print(f"  - spatial_heatmap.png")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
