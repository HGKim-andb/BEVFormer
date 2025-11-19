#!/usr/bin/env python3
"""
Test impact of lateral distance (좌우 거리) on risk score

Shows how changing d_traj_max and d_far affects lateral tolerance
"""

import numpy as np

def compute_proximity_factor(dist_to_traj, d_close, d_far):
    """
    Compute Proximity factor (P)

    Args:
        dist_to_traj: perpendicular distance to trajectory (m)
        d_close: on-trajectory threshold (m)
        d_far: off-trajectory threshold (m)

    Returns:
        P: proximity factor [0, 1]
    """
    if dist_to_traj <= d_close:
        return 1.0
    elif dist_to_traj >= d_far:
        return 0.0
    else:
        # Linear interpolation
        return (d_far - dist_to_traj) / (d_far - d_close)


print("=" * 80)
print("좌우 거리(Lateral Distance) 영향 테스트")
print("=" * 80)
print()

# Test configurations
configs = [
    ("V5 Original (strict)", 15.0, 5.0, 15.0),
    ("V5 New (relaxed)", 20.0, 5.0, 20.0),
    ("Very Relaxed", 25.0, 5.0, 25.0),
]

# Test lateral distances
lateral_distances = [0, 2, 5, 8, 10, 12, 15, 18, 20]

print(f"{'Lateral Dist':<15}", end="")
for config_name, _, _, _ in configs:
    print(f"{config_name:>20}", end="")
print()
print("=" * 80)

for dist in lateral_distances:
    print(f"{dist:>4.0f}m          ", end="")

    for config_name, d_traj_max, d_close, d_far in configs:
        # Check if filtered out by trajectory corridor
        if dist > d_traj_max:
            proximity = 0.0
            status = "FILTERED"
        else:
            proximity = compute_proximity_factor(dist, d_close, d_far)
            status = f"P={proximity:.3f}"

        print(f"{status:>20}", end="")

    print()

print("=" * 80)
print()
print("KEY:")
print("  FILTERED = 경로에서 너무 멀어서 필터링됨 (is_on_trajectory=False)")
print("  P=X.XXX  = Proximity factor (1.0=경로상, 0.0=경로 밖)")
print()

# Example calculation with actual occlusion
print("=" * 80)
print("실제 위험도 계산 예시 (O=0.9, U=0.6 고정)")
print("=" * 80)
print()

O = 0.9  # Occlusion factor
U = 0.6  # Urgency factor

print(f"{'Lateral':<10} {'Config':<20} {'P':<10} {'Risk (O×U×P)':<15} {'Change':<10}")
print("-" * 70)

for dist in [5, 10, 15, 20]:
    baseline_risk = None

    for i, (config_name, d_traj_max, d_close, d_far) in enumerate(configs):
        if dist > d_traj_max:
            P = 0.0
        else:
            P = compute_proximity_factor(dist, d_close, d_far)

        risk = O * U * P

        if i == 0:
            baseline_risk = risk
            change = "-"
        else:
            if baseline_risk == 0:
                change = f"+{risk:.3f}" if risk > 0 else "-"
            else:
                pct_change = ((risk - baseline_risk) / baseline_risk) * 100
                change = f"{pct_change:+.1f}%"

        print(f"{dist:>3.0f}m      {config_name:<20} {P:<10.3f} {risk:<15.3f} {change:<10}")

    print()

print("=" * 80)
print("결론:")
print("=" * 80)
print("d_traj_max: 15m → 20m")
print("  - 15-20m 거리의 셀들이 이제 포함됨 (이전엔 FILTERED)")
print()
print("d_far: 15m → 20m")
print("  - 같은 거리에서 Proximity factor가 증가")
print("  - 예: 10m 떨어진 셀 → P: 0.500 → 0.667 (33% 증가)")
print("  - 예: 15m 떨어진 셀 → P: 0.000 → 0.333 (무한대 증가)")
print()
print("전체 효과:")
print("  좌우로 더 멀리 떨어진 셀들도 risk가 0이 아닌 값을 가지게 됨")
print("  → 좌우 거리에 대한 패널티가 감소 ✓")
print("=" * 80)
