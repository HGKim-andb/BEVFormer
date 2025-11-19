# Risk-Guided Attention 실험 가이드

## 개요

Risk-Guided Attention 메커니즘이 구현되고 테스트되었습니다. 이 가이드는 두 가지 모델을 비교하는 방법을 설명합니다.

## 모델 비교

### 1. Baseline: Risk Prediction Only
**Config**: `bevformer_risk_tiny.py`
- Risk head: `RiskPredictionHead`
- `use_risk_guidance=False`
- Risk map만 예측, detection에 영향 없음

### 2. Enhanced: Risk-Guided Attention
**Config**: `bevformer_risk_tiny_attention.py`
- Risk head: `RiskGuidedAttentionHead`
- `use_risk_guidance=True`
- Risk map이 spatial attention으로 BEV features 가중

## 구현 상세

### Risk-Guided Attention 동작 방식

```python
# 1. Risk map 예측
risk_map = risk_head(bev_features)  # [B, 1, 200, 200]

# 2. Risk map을 BEV resolution으로 downsample
risk_map_small = downsample(risk_map)  # [B, 1, 50, 50]

# 3. Attention weights 생성
attention_weights = spatial_attention_conv(risk_map_small)  # [B, 1, 50, 50]
attention_weights = sigmoid(attention_weights)  # [0, 1]

# 4. BEV features에 attention 적용
attended_features = bev_features * attention_weights  # Element-wise multiplication
```

### 주요 파라미터

```python
risk_head=dict(
    type='RiskGuidedAttentionHead',
    attention_type='spatial',  # Options: 'spatial', 'channel', 'both'
    attention_temp=1.0,        # Temperature for softmax (higher = softer)
)
```

## 실험 방법

### Step 1: Baseline 학습

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:$PYTHONPATH \
/home/hg-main/anaconda3/envs/vad1/bin/python tools/train.py \
    projects/configs/bevformer/bevformer_risk_tiny.py \
    --work-dir work_dirs/bevformer_risk_baseline

# Multi-GPU (3 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2 bash tools/dist_train.sh \
    projects/configs/bevformer/bevformer_risk_tiny.py 3
```

### Step 2: Risk-Guided Attention 학습

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:$PYTHONPATH \
/home/hg-main/anaconda3/envs/vad1/bin/python tools/train.py \
    projects/configs/bevformer/bevformer_risk_tiny_attention.py \
    --work-dir work_dirs/bevformer_risk_attention

# Multi-GPU (3 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2 bash tools/dist_train.sh \
    projects/configs/bevformer/bevformer_risk_tiny_attention.py 3
```

### Step 3: 결과 비교

#### 학습 로그 비교
```bash
# Baseline loss
grep "loss:" work_dirs/bevformer_risk_baseline/*.log | tail -20

# Attention loss
grep "loss:" work_dirs/bevformer_risk_attention/*.log | tail -20
```

#### Detection 성능 비교
```bash
# Evaluate baseline
PYTHONPATH=.:$PYTHONPATH python tools/test.py \
    projects/configs/bevformer/bevformer_risk_tiny.py \
    work_dirs/bevformer_risk_baseline/latest.pth \
    --eval bbox

# Evaluate attention model
PYTHONPATH=.:$PYTHONPATH python tools/test.py \
    projects/configs/bevformer/bevformer_risk_tiny_attention.py \
    work_dirs/bevformer_risk_attention/latest.pth \
    --eval bbox
```

#### Risk Prediction 비교
```bash
# Visualize baseline predictions
python tools/visualize_risk_simple.py \
    --checkpoint work_dirs/bevformer_risk_baseline/latest.pth \
    --split val \
    --num-samples 10 \
    --output visualizations/baseline

# Visualize attention predictions
python tools/visualize_risk_simple.py \
    --checkpoint work_dirs/bevformer_risk_attention/latest.pth \
    --split val \
    --num-samples 10 \
    --output visualizations/attention
```

## 검증 완료 사항

✅ **Config 생성 및 검증**
- `bevformer_risk_tiny_attention.py` 생성됨
- `RiskGuidedAttentionHead` 사용
- `use_risk_guidance=True` 설정됨

✅ **모델 빌드 테스트**
```
Risk head class: RiskGuidedAttentionHead
Has forward_with_attention: True
Attention type: spatial
```

✅ **Forward Pass 테스트**
```
✓ Risk map output: torch.Size([1, 1, 200, 200])
✓ Attention weights: torch.Size([1, 1, 50, 50])
✓ Attended features: torch.Size([1, 256, 50, 50])
✓ Attention range: [0.369, 0.482]
```

✅ **학습 시작 확인**
```
Epoch [1][50/28130]
loss_cls: 1.6653, loss_bbox: 1.8781
loss_risk_mse: 40.79, loss_risk_mae: 42.07, loss_risk: 61.83
loss: 166.01
```

## 기대 효과

### Detection 성능 향상
- 높은 risk 영역의 객체 탐지 정확도 향상
- False negative 감소 (특히 위험 시나리오에서)
- mAP, NDS 등 전반적인 metrics 개선

### Risk Prediction 정확도
- Risk map과 detection이 상호 보강
- End-to-end learning으로 더 정확한 risk 예측

### 논문 기여도
- **Baseline**: Multi-task learning (detection + risk prediction)
- **Contribution**: Risk-guided attention으로 성능 향상
- **Ablation study**: Attention 유무에 따른 성능 비교

## 추가 실험

### Attention Type 비교
```python
# Spatial attention (현재)
attention_type='spatial'

# Channel attention
attention_type='channel'

# Both
attention_type='both'
```

### Attention Temperature 튜닝
```python
attention_temp=0.5   # Sharper attention (더 극단적)
attention_temp=1.0   # Default
attention_temp=2.0   # Softer attention (더 부드러움)
```

### Risk Loss Weight 조정
```python
risk_loss_weight=50.0    # 낮은 가중치
risk_loss_weight=100.0   # Default
risk_loss_weight=200.0   # 높은 가중치
```

## 트러블슈팅

### DDP 에러 (multi-GPU)
```python
# Config에 추가 필요
find_unused_parameters = True
```

### OOM (Out of Memory)
```python
# Config 수정
samples_per_gpu=1  # 배치 크기 줄이기
fp16 = dict(loss_scale=512.)  # FP16 사용
```

### Attention weights가 너무 균일함
```python
# Temperature 낮추기
attention_temp=0.5
```

### Attention weights가 너무 극단적
```python
# Temperature 높이기
attention_temp=2.0
```

## 파일 위치

### Config Files
- Baseline: `projects/configs/bevformer/bevformer_risk_tiny.py`
- Attention: `projects/configs/bevformer/bevformer_risk_tiny_attention.py`

### Model Implementation
- Base head: `projects/mmdet3d_plugin/bevformer/dense_heads/risk_head.py`
  - `RiskPredictionHead` (baseline)
  - `RiskGuidedAttentionHead` (attention)
- Detector: `projects/mmdet3d_plugin/bevformer/detectors/bevformer_risk.py`

### Test Scripts
- Quick test: `test_risk_attention.py`
- Visualization: `tools/visualize_risk_simple.py`

## 다음 단계

1. **Full dataset 학습** (현재는 20% 데이터)
   - 서버에서 생성한 full dataset 가져오기
   - Config에서 경로 변경: `risk_labels_train_20pct.pkl` → `risk_labels_train.pkl`

2. **Multi-GPU 학습**
   - 3 GPUs로 학습 속도 향상
   - DDP 설정 확인

3. **논문 작성을 위한 결과 수집**
   - Detection metrics (mAP, NDS, etc.)
   - Risk prediction metrics (MSE, MAE)
   - Visualization samples
   - Ablation study 표

4. **Advanced experiments**
   - Channel attention vs Spatial attention 비교
   - Temperature parameter 실험
   - 다양한 risk threshold 실험
