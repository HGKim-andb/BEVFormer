#!/usr/bin/env python3
"""
Comprehensive dataset analysis tool

Usage:
    python tools/analyze_dataset.py \
        --labels data/emergence_risk_v5_full/risk_labels_train.pkl \
        --output_dir analysis_results
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_labels(labels_path):
    """Load risk labels"""
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    return labels


def analyze_overall_stats(labels):
    """Overall dataset statistics"""
    all_samples = [s for scene in labels.values() for s in scene]

    max_risks = [s['metadata']['max_risk'] for s in all_samples]
    mean_risks = [s['metadata']['mean_risk'] for s in all_samples]
    high_cells = [s['metadata']['high_risk_cells'] for s in all_samples]
    medium_cells = [s['metadata'].get('medium_risk_cells', 0) for s in all_samples]
    low_cells = [s['metadata'].get('low_risk_cells', 0) for s in all_samples]

    stats = {
        'total_scenes': len(labels),
        'total_samples': len(all_samples),
        'max_risk_mean': np.mean(max_risks),
        'max_risk_std': np.std(max_risks),
        'max_risk_min': np.min(max_risks),
        'max_risk_max': np.max(max_risks),
        'mean_risk_mean': np.mean(mean_risks),
        'mean_risk_std': np.std(mean_risks),
        'high_cells_mean': np.mean(high_cells),
        'high_cells_std': np.std(high_cells),
        'medium_cells_mean': np.mean(medium_cells),
        'low_cells_mean': np.mean(low_cells),
        'samples_gt_0.7': sum(1 for r in max_risks if r > 0.7),
        'samples_gt_0.5': sum(1 for r in max_risks if r > 0.5),
        'samples_gt_0.3': sum(1 for r in max_risks if r > 0.3),
    }

    return stats, all_samples


def analyze_by_scene(labels):
    """Per-scene statistics"""
    scene_stats = []

    for scene_token, scene_samples in labels.items():
        max_risks = [s['metadata']['max_risk'] for s in scene_samples]
        mean_risks = [s['metadata']['mean_risk'] for s in scene_samples]
        high_cells = [s['metadata']['high_risk_cells'] for s in scene_samples]

        scene_stats.append({
            'scene_token': scene_token,
            'scene_name': scene_samples[0]['scene_name'],
            'num_samples': len(scene_samples),
            'avg_max_risk': np.mean(max_risks),
            'std_max_risk': np.std(max_risks),
            'avg_mean_risk': np.mean(mean_risks),
            'avg_high_cells': np.mean(high_cells),
            'high_risk_ratio': sum(1 for r in max_risks if r > 0.7) / len(max_risks),
            'samples_gt_0.7': sum(1 for r in max_risks if r > 0.7),
        })

    # Sort by average max risk
    scene_stats.sort(key=lambda x: x['avg_max_risk'], reverse=True)

    return scene_stats


def plot_distributions(all_samples, output_dir):
    """Plot various distributions"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_risks = [s['metadata']['max_risk'] for s in all_samples]
    mean_risks = [s['metadata']['mean_risk'] for s in all_samples]
    high_cells = [s['metadata']['high_risk_cells'] for s in all_samples]

    # Risk distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Max risk histogram
    axes[0, 0].hist(max_risks, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0.7, color='red', linestyle='--', label='High-risk threshold')
    axes[0, 0].set_xlabel('Max Risk')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'Max Risk Distribution (n={len(max_risks)})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Cumulative distribution
    axes[0, 1].hist(max_risks, bins=50, cumulative=True, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0.7, color='red', linestyle='--', label='High-risk threshold')
    axes[0, 1].set_xlabel('Max Risk')
    axes[0, 1].set_ylabel('Cumulative Count')
    axes[0, 1].set_title('Cumulative Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Mean risk (log scale)
    mean_risks_nonzero = [r for r in mean_risks if r > 0]
    axes[1, 0].hist(mean_risks_nonzero, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Mean Risk')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Mean Risk Distribution (non-zero, n={len(mean_risks_nonzero)})')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # High-risk cells
    high_cells_nonzero = [c for c in high_cells if c > 0]
    axes[1, 1].hist(high_cells_nonzero, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('High-Risk Cell Count')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'High-Risk Cells Distribution (non-zero, n={len(high_cells_nonzero)})')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'risk_distributions.png', dpi=150)
    print(f'✓ Saved: {output_dir / "risk_distributions.png"}')

    plt.close()


def plot_scene_comparison(scene_stats, output_dir, top_n=20):
    """Plot scene comparison"""
    output_dir = Path(output_dir)

    # Top N scenes
    top_scenes = scene_stats[:top_n]

    scene_names = [s['scene_name'] for s in top_scenes]
    avg_risks = [s['avg_max_risk'] for s in top_scenes]
    high_ratios = [s['high_risk_ratio'] * 100 for s in top_scenes]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Average max risk
    axes[0].barh(range(len(scene_names)), avg_risks, color='steelblue')
    axes[0].set_yticks(range(len(scene_names)))
    axes[0].set_yticklabels(scene_names, fontsize=8)
    axes[0].set_xlabel('Average Max Risk')
    axes[0].set_title(f'Top {top_n} Highest Risk Scenes')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].axvline(0.7, color='red', linestyle='--', label='High-risk threshold')
    axes[0].legend()

    # High-risk ratio
    axes[1].barh(range(len(scene_names)), high_ratios, color='coral')
    axes[1].set_yticks(range(len(scene_names)))
    axes[1].set_yticklabels(scene_names, fontsize=8)
    axes[1].set_xlabel('High-Risk Sample Ratio (%)')
    axes[1].set_title(f'Top {top_n} Scenes by High-Risk Ratio')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_dir / 'scene_comparison.png', dpi=150)
    print(f'✓ Saved: {output_dir / "scene_comparison.png"}')

    plt.close()


def save_statistics(stats, scene_stats, output_dir):
    """Save statistics to text file"""
    output_dir = Path(output_dir)

    with open(output_dir / 'statistics.txt', 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('RISK DATASET ANALYSIS REPORT\n')
        f.write('=' * 80 + '\n\n')

        # Overall stats
        f.write('OVERALL STATISTICS\n')
        f.write('-' * 80 + '\n')
        f.write(f"Total scenes:  {stats['total_scenes']}\n")
        f.write(f"Total samples: {stats['total_samples']}\n\n")

        f.write('Risk Statistics:\n')
        f.write(f"  Max risk (mean): {stats['max_risk_mean']:.3f} ± {stats['max_risk_std']:.3f}\n")
        f.write(f"  Max risk (range): [{stats['max_risk_min']:.3f}, {stats['max_risk_max']:.3f}]\n")
        f.write(f"  Mean risk (mean): {stats['mean_risk_mean']:.4f} ± {stats['mean_risk_std']:.4f}\n\n")

        f.write('Cell Statistics:\n')
        f.write(f"  High-risk cells (avg): {stats['high_cells_mean']:.1f} ± {stats['high_cells_std']:.1f}\n")
        f.write(f"  Medium-risk cells (avg): {stats['medium_cells_mean']:.1f}\n")
        f.write(f"  Low-risk cells (avg): {stats['low_cells_mean']:.1f}\n\n")

        f.write('Risk Distribution:\n')
        pct_0_7 = 100 * stats['samples_gt_0.7'] / stats['total_samples']
        pct_0_5 = 100 * stats['samples_gt_0.5'] / stats['total_samples']
        pct_0_3 = 100 * stats['samples_gt_0.3'] / stats['total_samples']
        f.write(f"  Samples > 0.7: {stats['samples_gt_0.7']} ({pct_0_7:.1f}%)\n")
        f.write(f"  Samples > 0.5: {stats['samples_gt_0.5']} ({pct_0_5:.1f}%)\n")
        f.write(f"  Samples > 0.3: {stats['samples_gt_0.3']} ({pct_0_3:.1f}%)\n\n")

        # Scene stats
        f.write('=' * 80 + '\n')
        f.write('TOP 20 HIGHEST RISK SCENES\n')
        f.write('=' * 80 + '\n\n')

        for i, scene in enumerate(scene_stats[:20], 1):
            f.write(f"{i:2d}. {scene['scene_name']:<20}\n")
            f.write(f"    Samples: {scene['num_samples']:3d} | ")
            f.write(f"Avg max risk: {scene['avg_max_risk']:.3f} ± {scene['std_max_risk']:.3f} | ")
            f.write(f"High-risk ratio: {scene['high_risk_ratio']*100:.1f}%\n")

        f.write('\n')
        f.write('=' * 80 + '\n')
        f.write('BOTTOM 20 LOWEST RISK SCENES\n')
        f.write('=' * 80 + '\n\n')

        for i, scene in enumerate(scene_stats[-20:], 1):
            f.write(f"{i:2d}. {scene['scene_name']:<20}\n")
            f.write(f"    Samples: {scene['num_samples']:3d} | ")
            f.write(f"Avg max risk: {scene['avg_max_risk']:.3f} ± {scene['std_max_risk']:.3f} | ")
            f.write(f"High-risk ratio: {scene['high_risk_ratio']*100:.1f}%\n")

    print(f'✓ Saved: {output_dir / "statistics.txt"}')


def main():
    parser = argparse.ArgumentParser(description='Analyze risk dataset')
    parser.add_argument('--labels_train', type=str, required=True,
                        help='Path to train risk labels pickle file')
    parser.add_argument('--labels_val', type=str, default=None,
                        help='Path to val risk labels pickle file (optional)')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                        help='Output directory for analysis results')

    args = parser.parse_args()

    print('=' * 80)
    print('RISK DATASET ANALYSIS')
    print('=' * 80)
    print(f'Train labels: {args.labels_train}')
    if args.labels_val:
        print(f'Val labels:   {args.labels_val}')
    print(f'Output:       {args.output_dir}')
    print()

    # Load train labels
    print('Loading train labels...')
    labels_train = load_labels(args.labels_train)

    # Analyze train
    print('Analyzing train set...')
    stats_train, samples_train = analyze_overall_stats(labels_train)
    scene_stats_train = analyze_by_scene(labels_train)

    # Plot train
    print('Generating train plots...')
    output_train = Path(args.output_dir) / 'train'
    plot_distributions(samples_train, output_train)
    plot_scene_comparison(scene_stats_train, output_train)

    # Save train stats
    print('Saving train statistics...')
    save_statistics(stats_train, scene_stats_train, output_train)

    # Load and analyze val if provided
    if args.labels_val:
        import os
        if os.path.exists(args.labels_val):
            print()
            print('Loading val labels...')
            labels_val = load_labels(args.labels_val)

            print('Analyzing val set...')
            stats_val, samples_val = analyze_overall_stats(labels_val)
            scene_stats_val = analyze_by_scene(labels_val)

            # Plot val
            print('Generating val plots...')
            output_val = Path(args.output_dir) / 'val'
            plot_distributions(samples_val, output_val)
            plot_scene_comparison(scene_stats_val, output_val)

            # Save val stats
            print('Saving val statistics...')
            save_statistics(stats_val, scene_stats_val, output_val)
        else:
            print(f'⚠️  Val labels not found: {args.labels_val}')

    print()
    print('=' * 80)
    print('ANALYSIS COMPLETE')
    print('=' * 80)
    print(f'Results saved to: {args.output_dir}/')
    print('  Train results: {}/train/'.format(args.output_dir))
    print('    - risk_distributions.png')
    print('    - scene_comparison.png')
    print('    - statistics.txt')
    if args.labels_val and os.path.exists(args.labels_val):
        print('  Val results: {}/val/'.format(args.output_dir))
        print('    - risk_distributions.png')
        print('    - scene_comparison.png')
        print('    - statistics.txt')
    print('=' * 80)


if __name__ == '__main__':
    main()
