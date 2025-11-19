# Risk-Guided Attention 구현 및 테스트 완료 요약

## ✅ 완료 사항

### 1. Risk-Guided Attention 메커니즘 완전 구현
- **Location**: `projects/mmdet3d_plugin/bevformer/dense_heads/risk_head.py`
- **Class**: `RiskGuidedAttentionHead`
- **기능**: Spatial attention을 사용하여 risk map 기반으로 BEV features 강화

### 2. Config 파일 생성
- **File**: `projects/configs/bevformer/bevformer_risk_tiny_attention.py`
- **설정**:
  - `use_risk_guidance=True` (활성화!)
  - `RiskGuidedAttentionHead` 사용
  - `attention_type='spatial'`
  - `attention_temp=1.0`

### 3. 테스트 완료
- ✅ **모델 빌드**: 성공
- ✅ **Forward pass**: 정상 동작
- ✅ **Attention 생성**: [0.369, 0.482] 범위
- ✅ **학습 시작**: 정상적으로 진행

### 4. 교육 자료 생성
- **ATTENTION_MECHANISM_EXPLAINED.md**: 완전한 이론 및 코드 설명
- **demo_attention.py**: 실행 가능한 데모 (4×4 예제)
- **RISK_ATTENTION_GUIDE.md**: 실험 및 비교 가이드

## 🔍 발견된 문제 및 해결

### 문제 1: Risk Loss가 0
**증상**:
```
loss_risk_mse: 0.0000, loss_risk_mae: 0.0000, loss_risk: 0.0000
```

**원인**:
- 20% subsampling → 1 scene, 40 samples만 생성됨
- NuScenes mini dataset train set: 8 scenes, 324 samples

**해결**:
```python
# Before:
risk_labels_path='data/emergence_risk_v5/risk_labels_train_20pct.pkl'  # 40 samples

# After:
risk_labels_path='data/emergence_risk_v5/risk_labels_train.pkl'  # 324 samples
```

### 문제 2: Dataset 크기
**NuScenes Mini Dataset**:
- Train: 8 scenes, 324 samples
- Val: 2 scenes, 80 samples
- **매우 작은 데이터셋!**

**권장 사항**:
1. **Short-term**: Mini dataset으로 개념 증명
2. **Long-term**: Full dataset (v1.0-trainval) 사용
   - Train: ~700 scenes, ~28,000 samples
   - Val: ~150 scenes, ~6,000 samples

## 📊 Attention 메커니즘 설명

### 핵심 원리
```python
# 1단계: Risk map 예측
risk_map = risk_head(bev_features)  # [B, 1, 200, 200]

# 2단계: Attention weights 생성
risk_small = downsample(risk_map, size=(50, 50))
attention = spatial_conv(risk_small)  # [B, 1, 50, 50]
attention = sigmoid(attention)  # 0~1 범위

# 3단계: BEV features 강화
attended_features = bev_features * attention  # Element-wise multiplication
# 높은 risk 영역: features × 0.9 (강화)
# 낮은 risk 영역: features × 0.2 (억제)
```

### 예시 (교차로 시나리오)
```
Risk Map:
  교차로 우측 (측면 차량): 0.9 → Attention: 0.85
  자차 전방 (빈 도로): 0.1  → Attention: 0.25

결과:
  교차로 features: × 0.85 (강화) → 객체 더 잘 탐지
  빈 도로 features: × 0.25 (억제) → False positive 감소
```

## 🚀 다음 단계

### 1. Full Dataset 사용 (우선!)
```bash
# 서버에서 full dataset risk labels 가져오기
# data/emergence_risk_v5_full/ 디렉토리에 복사

# Config 수정
risk_labels_path='data/emergence_risk_v5_full/risk_labels_train.pkl'
risk_labels_path='data/emergence_risk_v5_full/risk_labels_val.pkl'
```

### 2. Baseline 비교 실험
```bash
# Baseline (no attention)
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    projects/configs/bevformer/bevformer_risk_tiny.py \
    --work-dir work_dirs/bevformer_risk_baseline

# With attention
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    projects/configs/bevformer/bevformer_risk_tiny_attention.py \
    --work-dir work_dirs/bevformer_risk_attention
```

### 3. 파라미터 실험
```python
# Attention type
attention_type='spatial'   # Current
attention_type='channel'
attention_type='both'

# Temperature
attention_temp=0.5   # Sharp (더 극단적)
attention_temp=1.0   # Default
attention_temp=2.0   # Soft (더 부드럽게)
```

### 4. 성능 평가
```bash
# Detection metrics
python tools/test.py CONFIG CHECKPOINT --eval bbox

# Risk metrics (MSE, MAE)
# 학습 로그에서 loss_risk_mse, loss_risk_mae 확인
```

### 5. Visualization
```bash
# Attention weights 시각화
python tools/visualize_risk_predictions.py \
    --config CONFIG \
    --checkpoint CHECKPOINT \
    --output visualizations/attention_maps
```

## 📝 논문 작성 계획

### Method Section
1. **Background**: BEVFormer 구조 설명
2. **Risk Prediction Head**: CNN 기반 risk map 예측
3. **Risk-Guided Attention**:
   - Motivation: 위험 영역에 집중
   - Architecture: Spatial attention conv
   - Training: Multi-task learning (detection + risk)

### Experiments
1. **Ablation Study**:
   - Baseline (no risk)
   - + Risk prediction (no attention)
   - + Risk-guided attention (spatial)
   - + Risk-guided attention (channel)
   - + Risk-guided attention (both)

2. **Performance Metrics**:
   - Detection: mAP, NDS, mATE, mASE, etc.
   - Risk: MSE, MAE
   - Qualitative: Attention visualization

3. **Analysis**:
   - High-risk scenarios에서 성능 향상
   - Attention weights가 실제 위험 영역에 집중
   - Computational overhead 거의 없음 (~0.0003%)

## 🎓 이해도 체크

### 이해해야 할 핵심 개념:
1. ✅ Attention = features에 곱해지는 가중치 (0~1)
2. ✅ Element-wise multiplication으로 selective enhancement
3. ✅ Risk map → Attention conv → Attention weights
4. ✅ 학습 가능한 attention (conv weights가 최적화됨)
5. ✅ Multi-task learning으로 상호 보강

### 실행 가능한 데모:
```bash
# 간단한 예제로 이해
python demo_attention.py

# 실제 모델 테스트
python test_risk_attention.py
```

## 📚 참고 자료

### 생성된 파일:
1. **ATTENTION_MECHANISM_EXPLAINED.md** - 이론 + 코드 완전 설명
2. **demo_attention.py** - 4×4 간단한 예제
3. **RISK_ATTENTION_GUIDE.md** - 실험 가이드
4. **test_risk_attention.py** - 모델 테스트

### 핵심 코드:
- Risk Head: `projects/mmdet3d_plugin/bevformer/dense_heads/risk_head.py`
- Detector: `projects/mmdet3d_plugin/bevformer/detectors/bevformer_risk.py`
- Config: `projects/configs/bevformer/bevformer_risk_tiny_attention.py`

## 💡 주요 통찰

1. **Attention은 생각보다 간단**:
   - 핵심: `features × weights`
   - Weights는 0~1 범위
   - 학습 가능 (conv layers)

2. **Risk-guided의 의미**:
   - Risk map이 "어디를 봐야 할지" 알려줌
   - Attention weights로 변환
   - BEV features 선택적 강화

3. **Multi-task의 시너지**:
   - Risk 학습 → Better attention
   - Better attention → Better detection
   - Better detection → Better risk (간접적)

4. **Computational Efficiency**:
   - Attention overhead: ~320 params
   - BEVFormer: ~100M params
   - 거의 무시할 수준 (0.0003%)

## 🔧 문제 해결

### 현재 상태:
- ✅ 코드 구현 완료
- ✅ 테스트 통과
- ⚠️ Dataset 크기 문제 (mini: 324 samples)
- 🔄 Full dataset 필요

### 다음 액션:
1. Full dataset risk labels 생성 또는 가져오기
2. Config 업데이트
3. 학습 시작
4. Baseline 비교
5. 논문 작성

---

**상태**: 구현 완료, 테스트 완료, Full dataset 준비 중
**마지막 업데이트**: 2025-11-19
