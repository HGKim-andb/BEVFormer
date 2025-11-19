# Risk Prediction Validation Guide

Epoch 1이 끝난 후 risk prediction 성능을 확인하는 방법입니다.

## 1. Checkpoint 확인

학습이 완료되면 checkpoint가 저장됩니다:

```bash
ls -lh work_dirs/bevformer_risk_single2/epoch_*.pth
```

예상 출력:
```
-rw-r--r-- 1 user user 1.2G Nov 19 12:00 epoch_1.pth
```

## 2. Risk Prediction 시각화 및 평가

### 기본 사용법

```bash
python tools/visualize_risk_predictions.py \
    --config projects/configs/bevformer/bevformer_risk_tiny.py \
    --checkpoint work_dirs/bevformer_risk_single2/epoch_1.pth \
    --num-samples 20 \
    --output-dir visualizations/risk_epoch1
```

**출력**:
- `visualizations/risk_epoch1/sample_XXXX.png`: GT vs Pred 비교 이미지
- `visualizations/risk_epoch1/metrics.txt`: 정량적 지표

### High-risk 샘플만 확인

```bash
python tools/visualize_risk_predictions.py \
    --config projects/configs/bevformer/bevformer_risk_tiny.py \
    --checkpoint work_dirs/bevformer_risk_single2/epoch_1.pth \
    --num-samples 10 \
    --high-risk-only \
    --output-dir visualizations/risk_epoch1_highrisk
```

이렇게 하면 max_risk > 0.7인 샘플만 선택해서 평가합니다.

### Train set 확인

```bash
python tools/visualize_risk_predictions.py \
    --config projects/configs/bevformer/bevformer_risk_tiny.py \
    --checkpoint work_dirs/bevformer_risk_single2/epoch_1.pth \
    --split train \
    --num-samples 10 \
    --output-dir visualizations/risk_epoch1_train
```

## 3. 결과 확인

### 3.1 시각화 이미지

`visualizations/risk_epoch1/sample_XXXX.png` 파일들을 확인하세요.

각 이미지에는 3개의 subplot이 있습니다:
1. **Ground Truth Risk**: 실제 위험 맵
2. **Predicted Risk**: 모델이 예측한 위험 맵
3. **Absolute Difference**: 절대 오차

**좋은 예시**:
- Pred가 GT와 유사한 패턴
- High-risk 영역(빨간색)을 잘 찾음
- Difference가 전반적으로 어두움 (낮은 오차)

**나쁜 예시**:
- Pred가 전부 파란색 (모두 0 예측)
- GT에는 위험이 있지만 Pred는 없음
- Difference가 밝음 (높은 오차)

### 3.2 정량적 지표 (`metrics.txt`)

```txt
RISK PREDICTION METRICS
================================================================================

Checkpoint: work_dirs/bevformer_risk_single2/epoch_1.pth
Dataset: val
Num samples: 20

mse_mean: 0.001234        # 낮을수록 좋음 (< 0.01)
mae_mean: 0.023456        # 낮을수록 좋음 (< 0.05)

max_risk_gt_mean: 0.594   # GT의 평균 max risk
max_risk_pred_mean: 0.XXX # Pred의 평균 max risk (GT와 비슷해야 함)

mean_risk_gt_mean: 0.0099
mean_risk_pred_mean: 0.XXX

precision_mean: 0.XXX     # High-risk 영역 검출 정밀도 (> 0.5 좋음)
recall_mean: 0.XXX        # High-risk 영역 검출 재현율 (> 0.5 좋음)
f1_mean: 0.XXX            # F1 score (> 0.5 좋음)

zero_prediction_ratio_mean: 0.XXX  # 0 예측 비율 (< 0.5 좋음)
```

### 3.3 판단 기준

#### ✅ 학습이 잘 된 경우

1. **시각화**:
   - Pred가 GT와 유사한 hot-spot 패턴
   - High-risk 영역을 대체로 맞춤
   - Difference가 작음

2. **지표**:
   - `mae_mean < 0.05`
   - `precision_mean > 0.5` (high-risk 샘플에서)
   - `recall_mean > 0.3` (일부 놓쳐도 OK)
   - `zero_prediction_ratio_mean < 0.6`
   - `max_risk_pred_mean`이 `max_risk_gt_mean`과 유사

#### ⚠️ 학습이 잘 안 된 경우

**Case 1: 모든 것을 0으로 예측**
- `zero_prediction_ratio_mean > 0.9`
- `max_risk_pred_mean < 0.1`
- Pred 이미지가 전부 파란색

**해결책**:
- Risk loss weight를 더 높임 (100 → 500)
- High-risk 샘플만 학습 (`risk_threshold=0.3`)
- Focal loss 강도 증가

**Case 2: 위치는 맞지만 값이 너무 작음**
- Precision > 0.5, but max_risk_pred_mean << max_risk_gt_mean
- 패턴은 유사하지만 값의 scale이 다름

**해결책**:
- Risk loss weight 증가
- Sigmoid 제거하고 다른 activation 시도

**Case 3: 완전히 랜덤**
- Precision, Recall 모두 < 0.1
- Difference 이미지가 노란색/밝음

**해결책**:
- 더 오래 학습 (24 epoch)
- Learning rate 조정
- 모델 구조 검토

## 4. 추가 분석 도구

### 4.1 Risk Labels 통계 재확인

```bash
python tools/check_risk_labels.py
```

### 4.2 학습 로그 확인

```bash
# Loss 추이 확인
grep "loss_risk:" work_dirs/bevformer_risk_single2/*/log.txt | tail -100

# Tensorboard 실행
tensorboard --logdir=work_dirs/bevformer_risk_single2
```

브라우저에서 `http://localhost:6006` 접속하여:
- `loss_risk_mse`, `loss_risk_mae` 추이 확인
- Epoch별 변화 관찰

## 5. 다음 단계

### Epoch 1 결과가 좋은 경우
- Epoch 24까지 학습 계속
- 정기적으로 validation 확인 (Epoch 5, 10, 15, 20, 24)

### Epoch 1 결과가 나쁜 경우
- 위의 "해결책" 적용
- Config 수정 후 재학습

### 예시: Config 수정

```python
# projects/configs/bevformer/bevformer_risk_tiny.py

model = dict(
    type='BEVFormerRisk',
    risk_head=dict(
        type='RiskPredictionHead',
        # ...
    ),
    risk_loss_weight=500.0,  # 100 → 500으로 증가
)

data = dict(
    train=dict(
        type='NuScenesRiskDataset',
        risk_threshold=0.3,  # 0.0 → 0.3: high-risk samples만
        # ...
    ),
)
```

재학습:
```bash
# 기존 학습 중단
pkill -f bevformer_risk

# 새 config로 재시작
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    projects/configs/bevformer/bevformer_risk_tiny.py \
    --work-dir work_dirs/bevformer_risk_v2
```

## 6. 빠른 체크리스트

Epoch 1 완료 후:

- [ ] Checkpoint 생성 확인 (`epoch_1.pth`)
- [ ] Visualization 실행 (val set, 20 samples)
- [ ] Visualization 실행 (high-risk only, 10 samples)
- [ ] `metrics.txt` 확인
- [ ] 시각화 이미지 확인 (GT vs Pred)
- [ ] Zero prediction ratio 확인
- [ ] Precision/Recall 확인
- [ ] 필요시 config 수정
- [ ] Epoch 24까지 학습 계속 또는 재시작

## 7. 문제 해결

### Error: "CUDA out of memory"

Visualization 시 batch size는 1이지만, 메모리가 부족할 수 있습니다:

```bash
# CPU로 실행
python tools/visualize_risk_predictions.py \
    --config projects/configs/bevformer/bevformer_risk_tiny.py \
    --checkpoint work_dirs/bevformer_risk_single2/epoch_1.pth \
    --num-samples 5 \  # 샘플 수 줄임
    --output-dir visualizations/risk_epoch1
```

### Error: "No risk_map in result"

모델이 risk_head를 가지고 있는지 확인:

```python
import torch
checkpoint = torch.load('work_dirs/bevformer_risk_single2/epoch_1.pth')
print('risk_head' in checkpoint['state_dict'])  # True여야 함
```

### 시각화 이미지가 안 보임

SSH로 접속한 경우, 이미지를 로컬로 복사:

```bash
scp -r server:/path/to/visualizations/risk_epoch1 ./
```

또는 Jupyter notebook에서 확인:

```python
from IPython.display import Image
Image('visualizations/risk_epoch1/sample_0000.png')
```
