# Simple Risk Visualization Guide

현재 상황: Full prediction visualization은 복잡한 데이터 로딩 문제가 있습니다.
대신 **간단한 GT visualization**로 risk labels를 확인할 수 있습니다.

## Quick Start - GT Visualization만

### 1. Ground Truth Risk Maps 시각화

```bash
cd /home/hg-main/data2/BEVFormer

# Train set에서 5개 샘플 시각화
/home/hg-main/anaconda3/envs/vad1/bin/python tools/visualize_risk_simple.py \
    work_dirs/bevformer_risk_single/epoch_2.pth \
    --num-samples 5 \
    --output-dir visualizations/gt_train

# Val set에서 5개 샘플 시각화
/home/hg-main/anaconda3/envs/vad1/bin/python tools/visualize_risk_simple.py \
    work_dirs/bevformer_risk_single/epoch_2.pth \
    --num-samples 5 \
    --split val \
    --output-dir visualizations/gt_val
```

### 2. 결과 확인

```bash
# 생성된 이미지 확인
ls visualizations/gt_train/*.png

# 통계 확인
cat visualizations/gt_train/summary.txt
```

### 3. 예상 출력

```
GROUND TRUTH RISK LABELS SUMMARY
============================================================
Dataset: train
Num samples visualized: 5

Max Risk: min=0.0000, max=0.9870, mean=0.4039
Mean Risk: min=0.000000, max=0.020232, mean=0.005185
Non-zero: min=0.00%, max=3.16%, mean=1.32%
```

**해석**:
- `Max Risk mean=0.4039`: 샘플들의 최대 위험도 평균이 0.4 (중간 수준)
- `Mean Risk mean=0.005185`: Risk map이 매우 sparse (대부분 0)
- `Non-zero mean=1.32%`: 전체 셀의 약 1.3%만 위험 있음

### 4. 이미지 로컬로 복사 (SSH 사용시)

```bash
# 로컬 컴퓨터의 터미널에서
scp -r your-server:/home/hg-main/data2/BEVFormer/visualizations/gt_train ./
```

---

## 현재 문제점

Full prediction visualization (`visualize_risk_predictions.py`)은 다음 문제가 있습니다:

1. **Val set mismatch**: Validation risk labels가 실제 validation dataset과 매칭되지 않음
   - Risk labels: mini dataset 2 scenes (80 samples)
   - Val dataset: full dataset 6019 samples
   - 매칭 없음!

2. **Data loading complexity**: Multi-view temporal image loading이 test mode에서 다르게 작동

### 임시 해결책

**Option 1**: GT만 시각화 (위의 simple script 사용) ✅
- 장점: 빠르고 간단, 학습 데이터 확인 가능
- 단점: 모델 prediction 확인 불가

**Option 2**: Training 코드 내에서 validation시 자동 저장
- Epoch마다 validation 시 risk map 저장
- 학습 로그에서 risk metrics 확인

**Option 3**: 전체 validation dataset에 대해 risk labels 재생성
- 시간 소요: 약 1-2시간
- 하지만 mini dataset에서는 불필요

---

## 학습 진행 확인 방법

### 1. Loss 추이 확인

```bash
# 최근 로그 확인
grep "loss_risk:" work_dirs/bevformer_risk_single2/*/log.txt | tail -20

# Risk loss 값 확인
# Epoch 1 초반: loss_risk ~10-60 (큼)
# Epoch 1 후반: loss_risk ~0.001-0.01 (작음)
```

### 2. Checkpoint 확인

```bash
# Checkpoint 파일 확인
ls -lh work_dirs/bevformer_risk_single2/*.pth

# Epoch 1 완료되면:
# epoch_1.pth (~1.2GB)
```

### 3. Tensorboard 확인

```bash
# Tensorboard 실행
tensorboard --logdir=work_dirs/bevformer_risk_single2

# 브라우저에서 localhost:6006 접속
# Scalars → loss_risk_mse, loss_risk_mae 그래프 확인
```

---

## 다음 단계

### Epoch 1 완료 후

1. **GT와 학습된 label 비교**:
   - GT visualization으로 패턴 확인
   - 학습 로그에서 risk loss 감소 확인
   - Loss가 0.001 이하로 수렴했으면 OK

2. **판단 기준**:
   - ✅ **좋음**: `loss_risk` < 0.01, 안정적으로 감소
   - ⚠️ **나쁨**: `loss_risk` > 0.1, 또는 발산

3. **조치**:
   - 좋음 → Epoch 24까지 계속 학습
   - 나쁨 → Risk loss weight 조정 (100 → 500)

---

## Summary

**현재 사용 가능한 도구**:

1. ✅ **GT Visualization**: `tools/visualize_risk_simple.py`
   - Risk labels 패턴 확인
   - 통계 확인

2. ✅ **Loss Monitoring**: Training logs
   - `loss_risk_mse`, `loss_risk_mae` 확인
   - 수렴 여부 판단

3. ✅ **Tensorboard**: 그래프로 확인

**아직 작동 안 함**:

- ❌ Full prediction visualization (데이터 로딩 문제)
  - 복잡한 temporal multi-view 데이터 처리 필요
  - 나중에 수정 가능하지만 당장은 불필요

**권장 워크플로우**:

1. GT visualization으로 데이터 확인 ✅
2. 학습 진행하면서 loss 모니터링 ✅
3. Epoch 1 완료 후 loss 값으로 판단 ✅
4. 필요시 config 수정 후 재학습

간단하고 효과적입니다!
