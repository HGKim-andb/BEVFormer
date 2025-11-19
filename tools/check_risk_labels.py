#!/usr/bin/env python3
"""
Check risk labels statistics
"""
import pickle
import numpy as np
from pathlib import Path

def check_risk_labels(pkl_path):
    """Check risk label statistics"""
    print(f"Loading {pkl_path}...")

    with open(pkl_path, 'rb') as f:
        risk_labels = pickle.load(f)

    print(f"\nTotal scenes: {len(risk_labels)}")

    all_max_risks = []
    all_mean_risks = []
    all_nonzero_counts = []
    total_samples = 0

    for scene_token, scene_labels in risk_labels.items():
        total_samples += len(scene_labels)

        for label in scene_labels:
            risk_map = label['risk_map']
            metadata = label.get('metadata', {})

            # Calculate statistics
            max_risk = risk_map.max()
            mean_risk = risk_map.mean()
            nonzero_count = (risk_map > 0).sum()

            all_max_risks.append(max_risk)
            all_mean_risks.append(mean_risk)
            all_nonzero_counts.append(nonzero_count)

    # Overall statistics
    print(f"\n{'='*60}")
    print(f"RISK LABELS STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples: {total_samples}")
    print(f"\nMax Risk:")
    print(f"  Min:  {np.min(all_max_risks):.6f}")
    print(f"  Max:  {np.max(all_max_risks):.6f}")
    print(f"  Mean: {np.mean(all_max_risks):.6f}")
    print(f"  Std:  {np.std(all_max_risks):.6f}")

    print(f"\nMean Risk:")
    print(f"  Min:  {np.min(all_mean_risks):.6f}")
    print(f"  Max:  {np.max(all_mean_risks):.6f}")
    print(f"  Mean: {np.mean(all_mean_risks):.6f}")
    print(f"  Std:  {np.std(all_mean_risks):.6f}")

    print(f"\nNon-zero Cells:")
    print(f"  Min:  {np.min(all_nonzero_counts)}")
    print(f"  Max:  {np.max(all_nonzero_counts)}")
    print(f"  Mean: {np.mean(all_nonzero_counts):.1f}")
    print(f"  Total cells per map: {40000} (200x200)")

    # Check if all zeros
    num_all_zeros = sum(1 for mr in all_max_risks if mr == 0)
    print(f"\nSamples with all-zero risk maps: {num_all_zeros} / {total_samples} ({num_all_zeros/total_samples*100:.1f}%)")

    # Distribution
    print(f"\nRisk Distribution (max_risk per sample):")
    bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
    for i in range(len(bins)-1):
        count = sum(1 for mr in all_max_risks if bins[i] <= mr < bins[i+1])
        print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count} ({count/total_samples*100:.1f}%)")

    # High risk samples
    high_risk_samples = sum(1 for mr in all_max_risks if mr > 0.7)
    print(f"\nHigh-risk samples (max > 0.7): {high_risk_samples} ({high_risk_samples/total_samples*100:.1f}%)")

    print(f"\n{'='*60}")

    return {
        'total_samples': total_samples,
        'max_risks': all_max_risks,
        'mean_risks': all_mean_risks,
        'nonzero_counts': all_nonzero_counts,
    }

if __name__ == '__main__':
    import sys

    train_pkl = 'data/emergence_risk_v5_full/risk_labels_train.pkl'
    val_pkl = 'data/emergence_risk_v5_full/risk_labels_val.pkl'

    if Path(train_pkl).exists():
        print("\n" + "="*60)
        print("TRAIN SET")
        print("="*60)
        train_stats = check_risk_labels(train_pkl)
    else:
        print(f"Train labels not found: {train_pkl}")

    if Path(val_pkl).exists():
        print("\n" + "="*60)
        print("VALIDATION SET")
        print("="*60)
        val_stats = check_risk_labels(val_pkl)
    else:
        print(f"Val labels not found: {val_pkl}")
