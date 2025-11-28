#!/usr/bin/env python3
"""수정된 리스크 계산 테스트"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')

from tools.bev_risk_viz import RiskCalculationEngine, RiskConfig, RiskVisualizer

# 설정
config = RiskConfig(
    weight_trajectory=0.3,
    weight_occlusion=0.3,
    weight_temporal=0.2,
    weight_proximity=0.2,
    ego_velocity=10.0,
    bev_resolution=0.5
)

engine = RiskCalculationEngine(config)
print(f"✓ BEV 그리드: {engine.bev_width} × {engine.bev_height}")

# 차폐 시나리오: 앞쪽에 차량
H, W = engine.bev_height, engine.bev_width
occlusion = np.zeros((H, W), dtype=np.float32)

# 앞쪽 15m 지점에 차량 (10×10 셀)
center_y, center_x = H//2 + 30, W//2
occlusion[center_y-10:center_y+10, center_x-10:center_x+10] = 1.0

print(f"✓ 차폐 영역: ({center_y-10}:{center_y+10}, {center_x-10}:{center_x+10})")

# 리스크 계산
results = engine.calculate_risk_map(occlusion)

# 통계
print(f"\n=== 수정 후 통계 ===")
print(f"Risk Map - Max: {results['risk_map'].max():.3f}, Mean: {results['risk_map'].mean():.4f}")
print(f"T (masked) - Max: {results['T'].max():.3f}, Mean: {results['T'].mean():.4f}")
print(f"P (masked) - Max: {results['P'].max():.3f}, Mean: {results['P'].mean():.4f}")
print(f"T (raw) - Max: {results['T_raw'].max():.3f}, Mean: {results['T_raw'].mean():.4f}")
print(f"P (raw) - Max: {results['P_raw'].max():.3f}, Mean: {results['P_raw'].mean():.4f}")

# 시각화
visualizer = RiskVisualizer()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('수정된 리스크 계산: T와 P가 차폐 영역에만 적용', fontsize=16, fontweight='bold')

# Row 1: 원본 T, P (마스크 적용 전)
im1 = axes[0,0].imshow(results['T_raw'], cmap='YlOrRd', vmin=0, vmax=1)
axes[0,0].set_title('T (원본 - 마스크 적용 전)')
axes[0,0].plot(W//2, H//2, 'w*', markersize=15)
plt.colorbar(im1, ax=axes[0,0])

im2 = axes[0,1].imshow(results['P_raw'], cmap='plasma', vmin=0, vmax=1)
axes[0,1].set_title('P (원본 - 마스크 적용 전)')
axes[0,1].plot(W//2, H//2, 'w*', markersize=15)
plt.colorbar(im2, ax=axes[0,1])

im3 = axes[0,2].imshow(occlusion, cmap='Blues', vmin=0, vmax=1)
axes[0,2].set_title('차폐 마스크 (O)')
axes[0,2].plot(W//2, H//2, 'w*', markersize=15)
plt.colorbar(im3, ax=axes[0,2])

# Row 2: 마스크 적용 후 T, P, 최종 리스크
im4 = axes[1,0].imshow(results['T'], cmap='YlOrRd', vmin=0, vmax=1)
axes[1,0].set_title('T (마스크 적용 후) ✓')
axes[1,0].plot(W//2, H//2, 'w*', markersize=15)
plt.colorbar(im4, ax=axes[1,0])

im5 = axes[1,1].imshow(results['P'], cmap='plasma', vmin=0, vmax=1)
axes[1,1].set_title('P (마스크 적용 후) ✓')
axes[1,1].plot(W//2, H//2, 'w*', markersize=15)
plt.colorbar(im5, ax=axes[1,1])

im6 = axes[1,2].imshow(results['risk_map'], cmap='hot', vmin=0, vmax=1)
axes[1,2].set_title('최종 리스크 맵')
axes[1,2].plot(W//2, H//2, 'w*', markersize=15)
plt.colorbar(im6, ax=axes[1,2])

plt.tight_layout()
plt.savefig('fixed_risk_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✓ 저장: fixed_risk_comparison.png")

# 차폐 영역 내 vs 외부 비교
occlusion_region = occlusion > 0
T_in_occlusion = results['T'][occlusion_region]
T_outside = results['T'][~occlusion_region]

print(f"\n=== 차폐 영역 내/외부 비교 ===")
print(f"T 차폐 내부 - Max: {T_in_occlusion.max():.3f}, Mean: {T_in_occlusion.mean():.4f}")
print(f"T 차폐 외부 - Max: {T_outside.max():.3f}, Mean: {T_outside.mean():.4f}")
print(f"→ 외부는 모두 0이어야 함: {(T_outside == 0).all()}")

