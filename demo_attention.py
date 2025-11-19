#!/usr/bin/env python3
"""
Risk-Guided Attention 메커니즘 데모

이 스크립트는 attention이 어떻게 동작하는지 숫자로 보여줍니다.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print("="*80)
print("RISK-GUIDED ATTENTION 데모")
print("="*80)

# ============================================================================
# Step 1: 가상의 BEV Features 생성
# ============================================================================
print("\n[Step 1] BEV Features 생성")
print("-" * 80)

# 간단한 4×4 grid로 시연 (실제는 50×50)
# 8개 채널만 사용 (실제는 256개)
B, C, H, W = 1, 8, 4, 4

# Random features
np.random.seed(42)
bev_features = torch.randn(B, C, H, W)

print(f"BEV Features shape: {bev_features.shape}")
print(f"Sample feature vector at position (2, 3):")
print(f"  {bev_features[0, :, 2, 3].numpy().round(2)}")

# ============================================================================
# Step 2: Risk Map 생성 (간단한 패턴)
# ============================================================================
print("\n[Step 2] Risk Map 생성")
print("-" * 80)

# 교차로 시나리오: 오른쪽에 높은 risk
risk_map = torch.zeros(B, 1, H, W)
risk_map[0, 0, 0, 3] = 0.9  # 우측 상단: 높은 risk
risk_map[0, 0, 1, 3] = 0.8  # 우측 중간: 높은 risk
risk_map[0, 0, 2, 1] = 0.3  # 중앙: 중간 risk
# 나머지는 0.1 이하 (낮은 risk)

print("Risk Map (교차로 우측이 위험):")
print(risk_map[0, 0].numpy().round(2))
print("\n위치별 의미:")
print("  (0, 3) = 0.9 → 교차로 우측 (측면 차량)")
print("  (1, 3) = 0.8 → 교차로 우측")
print("  (2, 1) = 0.3 → 자차 전방")
print("  나머지 = 0.0 → 빈 공간")

# ============================================================================
# Step 3: Attention Weights 생성
# ============================================================================
print("\n[Step 3] Attention Weights 생성")
print("-" * 80)

# Spatial attention conv (실제 구현과 동일)
spatial_attention_conv = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 1, kernel_size=1),
)

# Forward pass
with torch.no_grad():
    attention_logits = spatial_attention_conv(risk_map)

# Sigmoid (0~1 범위)
temperature = 1.0
attention_weights = torch.sigmoid(attention_logits / temperature)

print(f"Attention Weights shape: {attention_weights.shape}")
print("Attention Weights (0~1):")
print(attention_weights[0, 0].numpy().round(3))

print("\n해석:")
high_attn = attention_weights[0, 0].max().item()
low_attn = attention_weights[0, 0].min().item()
print(f"  최고 attention: {high_attn:.3f} (위험 영역)")
print(f"  최저 attention: {low_attn:.3f} (안전 영역)")
print(f"  비율: {high_attn/low_attn:.1f}배 차이")

# ============================================================================
# Step 4: Attention 적용 (Element-wise Multiplication)
# ============================================================================
print("\n[Step 4] BEV Features에 Attention 적용")
print("-" * 80)

# Element-wise multiplication
attended_features = bev_features * attention_weights
# Broadcasting: [B, 8, 4, 4] × [B, 1, 4, 4] = [B, 8, 4, 4]

print(f"Attended Features shape: {attended_features.shape}")

# 특정 위치 비교
pos_high_risk = (0, 3)  # 높은 risk
pos_low_risk = (3, 0)   # 낮은 risk

print(f"\n위치별 비교:")
print(f"\n1. 높은 위험 영역 {pos_high_risk}:")
orig_high = bev_features[0, :, pos_high_risk[0], pos_high_risk[1]]
attn_high = attention_weights[0, 0, pos_high_risk[0], pos_high_risk[1]]
result_high = attended_features[0, :, pos_high_risk[0], pos_high_risk[1]]

print(f"   Original features: {orig_high.numpy().round(3)}")
print(f"   Attention weight:  {attn_high:.3f}")
print(f"   Attended features: {result_high.numpy().round(3)}")
print(f"   변화량: {((result_high/orig_high).mean().item()-1)*100:.1f}%")

print(f"\n2. 낮은 위험 영역 {pos_low_risk}:")
orig_low = bev_features[0, :, pos_low_risk[0], pos_low_risk[1]]
attn_low = attention_weights[0, 0, pos_low_risk[0], pos_low_risk[1]]
result_low = attended_features[0, :, pos_low_risk[0], pos_low_risk[1]]

print(f"   Original features: {orig_low.numpy().round(3)}")
print(f"   Attention weight:  {attn_low:.3f}")
print(f"   Attended features: {result_low.numpy().round(3)}")
print(f"   변화량: {((result_low/orig_low).mean().item()-1)*100:.1f}%")

# ============================================================================
# Step 5: Feature Magnitude 분석
# ============================================================================
print("\n[Step 5] Feature Magnitude 분석")
print("-" * 80)

# L2 norm (feature의 크기)
orig_magnitude = torch.norm(bev_features, dim=1, keepdim=True)
attended_magnitude = torch.norm(attended_features, dim=1, keepdim=True)

print("Original Feature Magnitudes:")
print(orig_magnitude[0, 0].numpy().round(3))

print("\nAttended Feature Magnitudes:")
print(attended_magnitude[0, 0].numpy().round(3))

print("\nMagnitude 변화 (attended / original):")
ratio = (attended_magnitude / (orig_magnitude + 1e-8))
print(ratio[0, 0].numpy().round(3))

# ============================================================================
# Step 6: 실제 효과 시뮬레이션
# ============================================================================
print("\n[Step 6] Detection에 미치는 영향 시뮬레이션")
print("-" * 80)

# 간단한 detection head (linear layer)
detection_head = nn.Linear(C, 1)  # 8채널 → 1 (objectness score)

with torch.no_grad():
    # Flatten features for linear layer
    orig_flat = bev_features.permute(0, 2, 3, 1).reshape(-1, C)
    attended_flat = attended_features.permute(0, 2, 3, 1).reshape(-1, C)

    # Detection scores
    orig_scores = torch.sigmoid(detection_head(orig_flat)).reshape(H, W)
    attended_scores = torch.sigmoid(detection_head(attended_flat)).reshape(H, W)

print("Original Detection Scores (without attention):")
print(orig_scores.numpy().round(3))

print("\nAttended Detection Scores (with attention):")
print(attended_scores.numpy().round(3))

print("\nScore 변화:")
score_diff = (attended_scores - orig_scores)
print(score_diff.numpy().round(3))

print("\n해석:")
print(f"  위험 영역 (0,3): {orig_scores[0,3]:.3f} → {attended_scores[0,3]:.3f} (변화: {score_diff[0,3]:+.3f})")
print(f"  안전 영역 (3,0): {orig_scores[3,0]:.3f} → {attended_scores[3,0]:.3f} (변화: {score_diff[3,0]:+.3f})")

# ============================================================================
# Step 7: Temperature 효과
# ============================================================================
print("\n[Step 7] Temperature 파라미터 효과")
print("-" * 80)

temperatures = [0.5, 1.0, 2.0]
print("Temperature가 attention weights에 미치는 영향:\n")

for temp in temperatures:
    attn_temp = torch.sigmoid(attention_logits / temp)
    max_attn = attn_temp[0, 0].max().item()
    min_attn = attn_temp[0, 0].min().item()

    print(f"Temperature = {temp}:")
    print(f"  Max attention: {max_attn:.3f}")
    print(f"  Min attention: {min_attn:.3f}")
    print(f"  Ratio (max/min): {max_attn/min_attn:.2f}x")
    print(f"  해석: ", end="")

    if temp < 1.0:
        print("더 극단적 (sharp) - 위험 영역만 강하게 강조")
    elif temp > 1.0:
        print("더 부드럽게 (soft) - 전체적으로 비슷한 weight")
    else:
        print("기본값 - 적당한 차별화")
    print()

# ============================================================================
# Summary
# ============================================================================
print("="*80)
print("요약")
print("="*80)
print("""
1. Risk Map이 위험 영역을 식별 (0~1 값)
2. Attention Conv가 risk를 attention weights로 변환
3. Element-wise multiplication으로 features 조정:
   - 높은 risk 영역: features 강화 (곱셈 factor > 0.5)
   - 낮은 risk 영역: features 억제 (곱셈 factor < 0.5)
4. Detection head가 강화된 features 사용
5. 결과: 위험 영역의 객체가 더 잘 탐지됨!

핵심 수식:
  attended_features = original_features × attention_weights

여기서 attention_weights는 risk map으로부터 학습된 conv로 생성됩니다.
""")

print("\n다음 단계:")
print("  - 실제 BEVFormer 모델에서 테스트: python test_risk_attention.py")
print("  - 학습 시작: 위 가이드 참고")
print("  - 시각화: tools/visualize_risk_simple.py 사용")
