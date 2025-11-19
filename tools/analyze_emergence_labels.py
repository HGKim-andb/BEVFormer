#!/usr/bin/env python3
"""
Emergence Label Analysis Script

This script analyzes the generated emergence labels and provides:
1. Statistical summary
2. Distribution analysis
3. Quality validation
4. Visualizations
"""

import numpy as np
import pickle
import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def load_labels(pkl_path):
    """Load labels from pickle file"""
    print(f"Loading labels from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        labels = pickle.load(f)
    return labels


def extract_statistics(labels_dict, split_name):
    """
    Extract statistics from labels

    Args:
        labels_dict: Dict of scene_token -> list of labels
        split_name: 'train' or 'val'

    Returns:
        Statistics dict
    """
    stats = {
        'split': split_name,
        'total_scenes': len(labels_dict),
        'total_samples': 0,
        'positive_samples': 0,
        'total_emergences': 0,
    }

    # Per-frame distribution
    frame_dist = defaultdict(int)
    # Category distribution
    category_dist = defaultdict(int)
    # Distance statistics
    distances = []
    # Spatial distribution (accumulate positions)
    spatial_heatmap = np.zeros((200, 200))

    # Process all labels
    for scene_token, scene_labels in labels_dict.items():
        for label in scene_labels:
            stats['total_samples'] += 1

            if label['num_emergences'] > 0:
                stats['positive_samples'] += 1
                stats['total_emergences'] += label['num_emergences']

                # Process each emergence
                for info in label['emergence_info']:
                    # Frame distribution
                    frame_dist[f"t+{info['frame']}"] += 1

                    # Category distribution
                    category = info['category']
                    if 'bicycle' in category:
                        simple_cat = 'bicycle'
                    elif 'motorcycle' in category:
                        simple_cat = 'motorcycle'
                    elif 'vehicle' in category:
                        simple_cat = 'vehicle'
                    elif 'pedestrian' in category:
                        simple_cat = 'pedestrian'
                    else:
                        simple_cat = 'other'

                    category_dist[simple_cat] += 1

                    # Distance
                    distances.append(info['distance'])

                    # Spatial heatmap
                    gx, gy = info['grid_pos']
                    if 0 <= gx < 200 and 0 <= gy < 200:
                        spatial_heatmap[gy, gx] += 1

    # Compute ratios and averages
    if stats['total_samples'] > 0:
        stats['positive_ratio'] = stats['positive_samples'] / stats['total_samples']
    else:
        stats['positive_ratio'] = 0

    if stats['positive_samples'] > 0:
        stats['avg_emergences_per_positive'] = stats['total_emergences'] / stats['positive_samples']
    else:
        stats['avg_emergences_per_positive'] = 0

    # Frame distribution
    stats['frame_distribution'] = dict(frame_dist)

    # Category distribution
    stats['category_distribution'] = dict(category_dist)

    # Distance statistics
    if distances:
        stats['distance_stats'] = {
            'mean': float(np.mean(distances)),
            'median': float(np.median(distances)),
            'min': float(np.min(distances)),
            'max': float(np.max(distances)),
            'std': float(np.std(distances)),
            'percentile_25': float(np.percentile(distances, 25)),
            'percentile_75': float(np.percentile(distances, 75)),
        }
    else:
        stats['distance_stats'] = {}

    # Store raw data for visualization
    stats['_raw_distances'] = distances
    stats['_spatial_heatmap'] = spatial_heatmap

    return stats


def validate_statistics(stats):
    """
    Validate statistics and check for potential issues

    Args:
        stats: Statistics dict

    Returns:
        List of issues (empty if all good)
    """
    issues = []

    # Check positive ratio
    pos_ratio = stats['positive_ratio']
    if not (0.05 <= pos_ratio <= 0.20):
        issues.append(f"⚠️  Positive ratio {pos_ratio:.1%} is outside expected range (5-20%)")
    else:
        print(f"✅ Positive ratio {pos_ratio:.1%} is within expected range")

    # Check frame distribution
    frame_dist = stats['frame_distribution']
    if 't+1' in frame_dist and 't+2' in frame_dist:
        if frame_dist['t+1'] < frame_dist['t+2']:
            issues.append("⚠️  t+1 should have more emergences than t+2 (objects appear sooner)")
        else:
            print(f"✅ Frame distribution is reasonable (t+1 > t+2)")

    # Check distance statistics
    if 'distance_stats' in stats and stats['distance_stats']:
        mean_dist = stats['distance_stats']['mean']
        if mean_dist < 5 or mean_dist > 35:
            issues.append(f"⚠️  Mean distance {mean_dist:.1f}m is unusual (expected 10-30m)")
        else:
            print(f"✅ Mean distance {mean_dist:.1f}m is reasonable")

    # Check if we have any emergences at all
    if stats['total_emergences'] == 0:
        issues.append("❌ No emergences found! Check the label generation logic.")

    return issues


def print_statistics(train_stats, val_stats):
    """Print formatted statistics report"""
    print("\n" + "="*80)
    print("EMERGENCE LABEL STATISTICS")
    print("="*80)

    for stats in [train_stats, val_stats]:
        split = stats['split'].upper()
        print(f"\n{split} SET:")
        print("-" * 80)
        print(f"Total scenes:                    {stats['total_scenes']}")
        print(f"Total samples:                   {stats['total_samples']}")
        print(f"Samples with emergence:          {stats['positive_samples']} ({stats['positive_ratio']*100:.2f}%)")
        print(f"Total emergence events:          {stats['total_emergences']}")
        print(f"Avg emergences per positive:     {stats['avg_emergences_per_positive']:.2f}")

        # Frame distribution
        print(f"\nPer-frame distribution:")
        frame_dist = stats['frame_distribution']
        for frame in ['t+1', 't+2', 't+3']:
            count = frame_dist.get(frame, 0)
            pct = count / stats['total_emergences'] * 100 if stats['total_emergences'] > 0 else 0
            print(f"  {frame}: {count:5d} ({pct:5.1f}%)")

        # Category distribution
        print(f"\nCategory distribution:")
        cat_dist = stats['category_distribution']
        for cat in ['pedestrian', 'vehicle', 'bicycle', 'motorcycle']:
            count = cat_dist.get(cat, 0)
            pct = count / stats['total_emergences'] * 100 if stats['total_emergences'] > 0 else 0
            print(f"  {cat:12s}: {count:5d} ({pct:5.1f}%)")

        # Distance statistics
        if stats['distance_stats']:
            print(f"\nDistance statistics (meters):")
            ds = stats['distance_stats']
            print(f"  Mean:   {ds['mean']:6.2f}")
            print(f"  Median: {ds['median']:6.2f}")
            print(f"  Std:    {ds['std']:6.2f}")
            print(f"  Min:    {ds['min']:6.2f}")
            print(f"  Max:    {ds['max']:6.2f}")
            print(f"  25%:    {ds['percentile_25']:6.2f}")
            print(f"  75%:    {ds['percentile_75']:6.2f}")

    print("\n" + "="*80 + "\n")


def create_visualizations(train_stats, val_stats, output_path):
    """
    Create visualization plots

    Args:
        train_stats: Training statistics
        val_stats: Validation statistics
        output_path: Path to save the figure
    """
    # Combine stats for overall view
    stats = train_stats  # Use train for main visualizations

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Frame distribution (top-left)
    ax = axes[0, 0]
    frame_dist = stats['frame_distribution']
    frames = ['t+1', 't+2', 't+3']
    counts = [frame_dist.get(f, 0) for f in frames]

    bars = ax.bar(frames, counts, color=['#FF6B6B', '#FFA500', '#FFD700'])
    ax.set_title('Emergence by Future Frame', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlabel('Future Frame', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=11)

    # 2. Category distribution (top-right)
    ax = axes[0, 1]
    cat_dist = stats['category_distribution']
    categories = ['pedestrian', 'vehicle', 'bicycle', 'motorcycle']
    cat_counts = [cat_dist.get(c, 0) for c in categories]
    cat_labels = [f"{c.capitalize()}\n({cnt})" for c, cnt in zip(categories, cat_counts)]

    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    wedges, texts, autotexts = ax.pie(cat_counts, labels=cat_labels, autopct='%1.1f%%',
                                        colors=colors, startangle=90)
    ax.set_title('Emergence by Category', fontsize=14, fontweight='bold')

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    # 3. Distance distribution (bottom-left)
    ax = axes[1, 0]
    distances = stats['_raw_distances']

    if distances:
        n, bins, patches = ax.hist(distances, bins=30, color='#74B9FF', edgecolor='black', alpha=0.7)

        # Add mean line
        mean_dist = stats['distance_stats']['mean']
        ax.axvline(mean_dist, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_dist:.1f}m')

        # Add median line
        median_dist = stats['distance_stats']['median']
        ax.axvline(median_dist, color='green', linestyle='--', linewidth=2,
                   label=f'Median: {median_dist:.1f}m')

        ax.set_title('Distance Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance (meters)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    # 4. Spatial heatmap (bottom-right)
    ax = axes[1, 1]
    spatial_heatmap = stats['_spatial_heatmap']

    # Apply log scale for better visualization
    heatmap_log = np.log1p(spatial_heatmap)

    im = ax.imshow(heatmap_log, cmap='hot', origin='lower',
                   extent=[-50, 50, -50, 50], aspect='equal')

    # Mark ego position (center)
    ax.plot(0, 0, 'b*', markersize=20, markeredgecolor='white', markeredgewidth=2)
    ax.text(0, -5, 'Ego', ha='center', color='white', fontsize=11,
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))

    ax.set_title('Spatial Distribution (Log Scale)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log(Count + 1)', fontsize=10)

    # Add title to entire figure
    fig.suptitle(f'Emergence Label Analysis - {stats["split"].upper()} Set',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze emergence labels')
    parser.add_argument('--train_labels', type=str,
                        default='data/emergence_labels/emergence_labels_train.pkl',
                        help='Path to training labels pickle file')
    parser.add_argument('--val_labels', type=str,
                        default='data/emergence_labels/emergence_labels_val.pkl',
                        help='Path to validation labels pickle file')
    parser.add_argument('--output_dir', type=str, default='data/emergence_labels',
                        help='Output directory for analysis results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("EMERGENCE LABEL ANALYSIS")
    print("="*80)
    print(f"Train labels: {args.train_labels}")
    print(f"Val labels:   {args.val_labels}")
    print(f"Output dir:   {output_dir}")
    print("="*80)

    # Load labels
    train_labels = load_labels(args.train_labels)
    val_labels = load_labels(args.val_labels)

    # Extract statistics
    print("\nExtracting statistics...")
    train_stats = extract_statistics(train_labels, 'train')
    val_stats = extract_statistics(val_labels, 'val')

    # Print statistics
    print_statistics(train_stats, val_stats)

    # Validate statistics
    print("Validating statistics...")
    print("\nTRAIN SET:")
    train_issues = validate_statistics(train_stats)
    if not train_issues:
        print("✅ All train statistics look reasonable!")
    else:
        print("Issues found:")
        for issue in train_issues:
            print(f"  {issue}")

    print("\nVAL SET:")
    val_issues = validate_statistics(val_stats)
    if not val_issues:
        print("✅ All val statistics look reasonable!")
    else:
        print("Issues found:")
        for issue in val_issues:
            print(f"  {issue}")

    # Save statistics to JSON
    # Remove raw data before saving
    train_stats_save = {k: v for k, v in train_stats.items() if not k.startswith('_')}
    val_stats_save = {k: v for k, v in val_stats.items() if not k.startswith('_')}

    stats_output = {
        'train': train_stats_save,
        'val': val_stats_save
    }

    stats_path = output_dir / 'analysis_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats_output, f, indent=2)
    print(f"\nSaved statistics to {stats_path}")

    # Create visualizations
    print("\nCreating visualizations...")
    viz_path = output_dir / 'distribution_plots.png'
    create_visualizations(train_stats, val_stats, viz_path)

    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output files:")
    print(f"  - {stats_path}")
    print(f"  - {viz_path}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
