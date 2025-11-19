#!/usr/bin/env python3
"""Compare risk calculation versions V1, V2, V3, V4"""

import pickle
import numpy as np
from pathlib import Path

# Load all versions
versions = {
    'V1 (Multiplicative)': 'data/emergence_risk_test/risk_labels_train.pkl',
    'V2 (Weighted Sum)': 'data/emergence_risk_v2/risk_labels_train.pkl',
    'V3 (Directional)': 'data/emergence_risk_v3/risk_labels_train.pkl',
    'V4 (Temporal)': 'data/emergence_risk_v4/risk_labels_train.pkl',
    'V5 (Continuous)': 'data/emergence_risk_v5_full/risk_labels_train.pkl',
}

print("=" * 100)
print("RISK CALCULATION VERSION COMPARISON (Scene-0061, 39 samples)")
print("=" * 100)

results = {}
for version_name, path in versions.items():
    pkl_path = Path(path)
    if not pkl_path.exists():
        print(f"⚠️  {version_name}: File not found - {path}")
        continue

    with open(pkl_path, 'rb') as f:
        labels_dict = pickle.load(f)

    # Get all samples
    all_labels = []
    for scene_token, scene_labels in labels_dict.items():
        all_labels.extend(scene_labels)

    if len(all_labels) == 0:
        print(f"⚠️  {version_name}: No labels found")
        continue

    # Compute statistics
    max_risks = [label['metadata']['max_risk'] for label in all_labels]
    mean_risks = [label['metadata']['mean_risk'] for label in all_labels]
    high_risk_cells = [label['metadata']['high_risk_cells'] for label in all_labels]

    results[version_name] = {
        'max_risk_avg': np.mean(max_risks),
        'max_risk_std': np.std(max_risks),
        'mean_risk_avg': np.mean(mean_risks),
        'mean_risk_std': np.std(mean_risks),
        'high_risk_cells_avg': np.mean(high_risk_cells),
        'high_risk_cells_std': np.std(high_risk_cells),
        'samples_above_0.7': sum(1 for r in max_risks if r > 0.7),
        'samples_above_0.5': sum(1 for r in max_risks if r > 0.5),
        'samples_above_0.3': sum(1 for r in max_risks if r > 0.3),
        'total_samples': len(all_labels),
    }

# Print comparison table
print("\n┌─────────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
print("│ Metric                  │ V1 (Mult)    │ V2 (WeightS) │ V3 (Direct)  │ V4 (Temporal)│ V5 (Contin)  │")
print("├─────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")

metrics = [
    ('Max risk (avg)', 'max_risk_avg', 'max_risk_std', '.3f'),
    ('Mean risk (avg)', 'mean_risk_avg', 'mean_risk_std', '.3f'),
    ('High-risk cells', 'high_risk_cells_avg', 'high_risk_cells_std', '.1f'),
]

for metric_name, avg_key, std_key, fmt in metrics:
    row = f"│ {metric_name:<23} │"
    for version_name in ['V1 (Multiplicative)', 'V2 (Weighted Sum)', 'V3 (Directional)', 'V4 (Temporal)', 'V5 (Continuous)']:
        if version_name in results:
            avg = results[version_name][avg_key]
            std = results[version_name][std_key]
            row += f" {avg:{fmt}} ± {std:{fmt[1:]}} │"
        else:
            row += "     N/A      │"
    print(row)

print("├─────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")

# Distribution
distribution = [
    ('Samples > 0.7', 'samples_above_0.7'),
    ('Samples > 0.5', 'samples_above_0.5'),
    ('Samples > 0.3', 'samples_above_0.3'),
]

for metric_name, key in distribution:
    row = f"│ {metric_name:<23} │"
    for version_name in ['V1 (Multiplicative)', 'V2 (Weighted Sum)', 'V3 (Directional)', 'V4 (Temporal)', 'V5 (Continuous)']:
        if version_name in results:
            count = results[version_name][key]
            total = results[version_name]['total_samples']
            pct = 100 * count / total if total > 0 else 0
            row += f" {count:2d} ({pct:5.1f}%)  │"
        else:
            row += "     N/A      │"
    print(row)

print("└─────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")

print("\n" + "=" * 100)
print("KEY CHANGES:")
print("=" * 100)
print("V1 → V2: Changed from multiplicative to weighted sum")
print("         Problem: V1 too conservative (0.054), one low factor destroyed score")
print("         Result: 15× increase in max_risk (0.054 → 0.808)")
print()
print("V2 → V3: Reduced component weights + added directional penalty")
print("         Problem: V2 too aggressive (85% samples > 0.7)")
print("         Changes:")
print("           - Occlusion: 0.4 → 0.3")
print("           - Urgency: 0.3 → 0.25")
print("           - Lateral: 0.15 → 0.12")
print("           - Context: 0.05 → 0.03")
print("           - Backward alignment penalty (alignment < 0 → score = 0)")
print("         Result: 16% decrease in max_risk (0.808 → 0.677), 21% samples > 0.7")
print()
print("V3 → V4: Added temporal trajectory awareness")
print("         Problem: Cells on past trajectory (behind ego) still have some risk")
print("         Changes:")
print("           - Added temporal_position_on_trajectory feature")
print("           - Applied exponential decay to proximity_score for past trajectory")
print("           - Behind cells: weight = exp(pos/5) × 0.2 (max 20%)")
print("           - Ahead cells: weight = 1.0 (full)")
print("         Result: Minimal change (temporal penalty overlaps with directional penalty)")
print("=" * 100)
