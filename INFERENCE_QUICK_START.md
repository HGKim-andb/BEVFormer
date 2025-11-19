# Risk Prediction Inference - Quick Start

내일 제출용 빠른 가이드입니다.

## 1. 학습 시작 (백그라운드)

```bash
cd /home/hg-main/data2/BEVFormer
conda activate vad1

# 학습 시작
PYTHONPATH=.:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 nohup python tools/train.py \
    projects/configs/bevformer/bevformer_risk_tiny.py \
    --work-dir work_dirs/bevformer_risk_final \
    > train_final.log 2>&1 &

# 학습 진행 확인
tail -f train_final.log
```

## 2. 인퍼런스 실행 (간단 버전)

학습 중이거나 완료 후 바로 실행 가능:

```bash
# Train set에서 20개 샘플 inference
python tools/inference_risk.py \
    --config projects/configs/bevformer/bevformer_risk_tiny.py \
    --checkpoint work_dirs/bevformer_risk_single/epoch_2.pth \
    --risk-labels data/emergence_risk_v5_full/risk_labels_train.pkl \
    --samples 20 \
    --output visualizations/inference_train

# Val set에서 10개 샘플 inference
python tools/inference_risk.py \
    --config projects/configs/bevformer/bevformer_risk_tiny.py \
    --checkpoint work_dirs/bevformer_risk_single/epoch_2.pth \
    --risk-labels data/emergence_risk_v5_full/risk_labels_val.pkl \
    --samples 10 \
    --output visualizations/inference_val
```

**예상 시간**: 10-30초

## 3. 결과 확인

```bash
# 생성된 이미지 확인
ls visualizations/inference_train/*.png

# Summary 확인
cat visualizations/inference_train/inference_summary.txt
```

## 4. 제출용 결과물

### 필요한 파일들

1. **모델 Checkpoint**:
   ```bash
   work_dirs/bevformer_risk_final/epoch_XX.pth
   ```

2. **학습 로그**:
   ```bash
   train_final.log
   work_dirs/bevformer_risk_final/*/log.txt
   ```

3. **Inference 결과**:
   ```bash
   visualizations/inference_train/
   ├── risk_sample_0000.png  (시각화 이미지들)
   ├── risk_sample_0001.png
   ├── ...
   └── inference_summary.txt  (통계 요약)
   ```

4. **Config 파일**:
   ```bash
   projects/configs/bevformer/bevformer_risk_tiny.py
   ```

5. **코드**:
   ```bash
   projects/mmdet3d_plugin/bevformer/dense_heads/risk_head.py
   projects/mmdet3d_plugin/bevformer/detectors/bevformer_risk.py
   projects/mmdet3d_plugin/datasets/nuscenes_risk_dataset.py
   ```

### 결과 압축

```bash
# 제출용 파일 압축
mkdir -p submission
cp -r visualizations/inference_train submission/
cp train_final.log submission/
cp work_dirs/bevformer_risk_final/epoch_*.pth submission/ 2>/dev/null || true
cp projects/configs/bevformer/bevformer_risk_tiny.py submission/

# 압축
tar -czf risk_bevformer_submission.tar.gz submission/
```

## 5. 학습 모니터링

### Loss 확인

```bash
# 최근 loss 확인
tail -50 train_final.log | grep "loss_risk"

# 특정 값 추출
grep "loss_risk:" train_final.log | tail -20
```

### 학습 중단 (필요시)

```bash
# 학습 프로세스 확인
ps aux | grep train.py

# 중단
pkill -f train.py
```

## 6. 시간별 체크리스트

### 지금 바로 (5분)

- [ ] 학습 시작 (백그라운드)
- [ ] Inference 스크립트 테스트 실행

### 학습 중 (30분마다)

- [ ] Loss 확인 (`tail -20 train_final.log | grep loss_risk`)
- [ ] `loss_risk < 0.01`인지 확인

### Epoch 1 완료 후 (약 5시간 후)

- [ ] Inference 실행 (20 samples)
- [ ] 결과 이미지 확인
- [ ] Summary 통계 확인

### 제출 전 (마지막)

- [ ] 최종 checkpoint 확인
- [ ] Inference 결과 생성
- [ ] 제출용 파일 압축
- [ ] README 작성

## 7. 빠른 문제 해결

### "ModuleNotFoundError: No module named 'projects'"

```bash
export PYTHONPATH=.:$PYTHONPATH
```

### Checkpoint 파일이 없음

```bash
# 기존 checkpoint 사용
ls work_dirs/bevformer_risk_single/*.pth
```

### 학습이 너무 느림

Single GPU는 느립니다 (약 5일). Multi-GPU 사용:

```bash
bash tools/dist_train.sh \
    projects/configs/bevformer/bevformer_risk_tiny.py 8 \
    --work-dir work_dirs/bevformer_risk_final_8gpu
```

## 8. 최소 제출 요구사항

시간이 부족하면 다음만 준비:

1. ✅ **Config 파일**: `bevformer_risk_tiny.py`
2. ✅ **모델 코드**: `risk_head.py`, `bevformer_risk.py`
3. ✅ **Inference 결과**: 10-20개 시각화 이미지
4. ✅ **Summary**: 통계 파일
5. ✅ **학습 로그**: Loss 추이

이것만 있어도 제출 가능합니다!

## 9. 현재 상황 빠른 확인

```bash
# 1. 학습 진행 중인지
ps aux | grep train.py

# 2. 최근 loss 값
tail -20 train_final.log | grep "loss_risk:"

# 3. Checkpoint 파일
ls -lh work_dirs/*/epoch_*.pth | tail -5

# 4. Inference 가능한지 테스트
python tools/inference_risk.py \
    --config projects/configs/bevformer/bevformer_risk_tiny.py \
    --checkpoint work_dirs/bevformer_risk_single/epoch_2.pth \
    --samples 3 \
    --output visualizations/test_inference
```

모두 정상이면 제출 준비 완료!
