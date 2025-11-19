# Quick Validation Guide - Epoch 1 완료 후

## 1. Epoch 1 완료 확인

학습이 계속 진행 중입니다. Epoch 1이 완료되면 checkpoint가 자동으로 저장됩니다.

### 확인 방법

```bash
# Checkpoint 파일 확인
ls -lh work_dirs/bevformer_risk_single2/*.pth

# Epoch 1 완료되면 이렇게 표시됨:
# -rw-r--r-- 1 user user 1.2G Nov 19 XX:XX epoch_1.pth
```

또는 학습 로그 확인:

```bash
# 최근 로그 확인 (새 터미널에서)
tail -f work_dirs/bevformer_risk_single2/*/scalars.json

# Epoch 2가 시작되면 Epoch 1 완료된 것
```

## 2. Visualization 실행 (Epoch 1 완료 후)

### 간단한 테스트 (3 샘플)

```bash
cd /home/hg-main/data2/BEVFormer

/home/hg-main/anaconda3/envs/vad1/bin/python tools/visualize_risk_predictions.py \
    --config projects/configs/bevformer/bevformer_risk_tiny.py \
    --checkpoint work_dirs/bevformer_risk_single2/epoch_1.pth \
    --num-samples 3 \
    --output-dir visualizations/risk_epoch1_quick
```

**예상 시간**: 약 1-2분

### 전체 Validation (20 샘플)

```bash
/home/hg-main/anaconda3/envs/vad1/bin/python tools/visualize_risk_predictions.py \
    --config projects/configs/bevformer/bevformer_risk_tiny.py \
    --checkpoint work_dirs/bevformer_risk_single2/epoch_1.pth \
    --num-samples 20 \
    --output-dir visualizations/risk_epoch1
```

**예상 시간**: 약 5-10분

### High-risk 샘플만 확인

```bash
/home/hg-main/anaconda3/envs/vad1/bin/python tools/visualize_risk_predictions.py \
    --config projects/configs/bevformer/bevformer_risk_tiny.py \
    --checkpoint work_dirs/bevformer_risk_single2/epoch_1.pth \
    --num-samples 10 \
    --high-risk-only \
    --output-dir visualizations/risk_epoch1_highrisk
```

## 3. 결과 확인

### 3.1 시각화 이미지 보기

로컬 컴퓨터로 복사 (SSH 사용하는 경우):

```bash
# 로컬 터미널에서
scp -r your-server:/home/hg-main/data2/BEVFormer/visualizations/risk_epoch1 ./
```

또는 서버에서 직접 확인:

```bash
# 이미지 리스트
ls visualizations/risk_epoch1/*.png

# 첫 번째 이미지 정보
file visualizations/risk_epoch1/sample_0000.png
```

### 3.2 Metrics 확인

```bash
cat visualizations/risk_epoch1/metrics.txt
```

**중요한 지표**:

```
zero_prediction_ratio_mean: X.XXX  # < 0.6이면 OK
precision_mean: X.XXX              # > 0.3이면 OK (high-risk 검출)
recall_mean: X.XXX                 # > 0.2이면 OK
mae_mean: X.XXX                    # < 0.05면 Good
max_risk_pred_mean: X.XXX          # GT와 비슷하면 OK
```

## 4. 판단 기준

### ✅ 좋은 경우

- `zero_prediction_ratio < 0.6`: 모델이 risk를 예측하고 있음
- `precision > 0.3`: High-risk 영역을 어느 정도 찾음
- `mae < 0.05`: 전반적으로 오차가 작음
- 시각화에서 GT와 Pred의 패턴이 유사

→ **계속 학습!** Epoch 24까지 진행

### ⚠️ 나쁜 경우

**Case 1: 모두 0 예측**
- `zero_prediction_ratio > 0.9`
- `max_risk_pred_mean < 0.1`
- Pred 이미지가 전부 파란색

→ **Config 수정 필요**:
```python
# projects/configs/bevformer/bevformer_risk_tiny.py
model = dict(
    risk_loss_weight=500.0,  # 100 → 500
)
```

**Case 2: 랜덤 예측**
- `precision < 0.1`, `recall < 0.1`
- Pred 이미지가 GT와 완전히 다름

→ **더 학습하거나 구조 검토**

## 5. Quick Commands 정리

```bash
# 1. Epoch 1 완료 확인
ls work_dirs/bevformer_risk_single2/epoch_1.pth

# 2. 빠른 테스트 (3 samples)
/home/hg-main/anaconda3/envs/vad1/bin/python tools/visualize_risk_predictions.py \
    --config projects/configs/bevformer/bevformer_risk_tiny.py \
    --checkpoint work_dirs/bevformer_risk_single2/epoch_1.pth \
    --num-samples 3 \
    --output-dir visualizations/test

# 3. Metrics 확인
cat visualizations/test/metrics.txt

# 4. 이미지 개수 확인
ls visualizations/test/*.png | wc -l

# 5. Risk labels 통계 (참고용)
/home/hg-main/anaconda3/envs/vad1/bin/python tools/check_risk_labels.py
```

## 6. Troubleshooting

### Error: "No such file: epoch_1.pth"

Epoch 1이 아직 완료되지 않았습니다. 학습 진행 상황 확인:

```bash
# 학습 프로세스 확인
ps aux | grep train.py

# 최근 로그 확인
tail -20 work_dirs/bevformer_risk_single2/*/log.json
```

### Error: "CUDA out of memory"

```bash
# 샘플 수를 줄여서 실행
/home/hg-main/anaconda3/envs/vad1/bin/python tools/visualize_risk_predictions.py \
    --config projects/configs/bevformer/bevformer_risk_tiny.py \
    --checkpoint work_dirs/bevformer_risk_single2/epoch_1.pth \
    --num-samples 1 \
    --output-dir visualizations/test
```

### 시각화 이미지를 볼 수 없음

Jupyter notebook 사용:

```python
from IPython.display import Image, display
import glob

images = glob.glob('visualizations/risk_epoch1/*.png')
for img_path in images[:5]:  # 처음 5개만
    print(f"\n{img_path}")
    display(Image(img_path))
```

## 7. 예상 Timeline

현재 학습 진행 중이므로:

- **Epoch 1 완료**: 약 5-6시간 후
- **Validation 실행**: 5-10분
- **결과 분석**: 10-20분
- **필요시 Config 수정 및 재학습**: 결정

**Total**: Epoch 1 완료 후 약 30분 내 validation 완료 가능
