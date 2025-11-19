# Risk-Guided BEVFormer Training Guide

## Config 파일 수정 완료

`img_norm_cfg` 에러가 수정되었습니다. 이제 학습을 시작할 수 있습니다.

## 학습 명령어

### 1. 멀티 GPU 학습 (권장)

```bash
# 8 GPUs
./tools/dist_train.sh \
    projects/configs/bevformer/bevformer_risk_tiny.py \
    8 \
    --work-dir work_dirs/bevformer_risk_tiny

# 4 GPUs
./tools/dist_train.sh \
    projects/configs/bevformer/bevformer_risk_tiny.py \
    4 \
    --work-dir work_dirs/bevformer_risk_tiny
```

### 2. 단일 GPU 학습

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    projects/configs/bevformer/bevformer_risk_tiny.py \
    --work-dir work_dirs/bevformer_risk_tiny_single
```

### 3. 빠른 테스트 (2 epochs, 학습이 잘 시작되는지 확인)

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    projects/configs/bevformer/bevformer_risk_tiny_trainonly.py \
    --work-dir work_dirs/bevformer_risk_test

# Multi GPU (2 GPUs)
./tools/dist_train.sh \
    projects/configs/bevformer/bevformer_risk_tiny_trainonly.py \
    2 \
    --work-dir work_dirs/bevformer_risk_test
```

## 주의사항

### Validation Risk Labels 생성

현재 `data/emergence_risk_v5_full/risk_labels_train.pkl`만 존재합니다.
Validation을 실행하려면 validation labels도 필요합니다:

```bash
# 이미 있는 스크립트로 생성 (시간이 걸림)
# nuScenes mini dataset 기준으로 train과 val을 모두 생성
python tools/create_risk_labels.py \
    --dataroot data/nuscenes \
    --version v1.0-mini \
    --output_dir data/emergence_risk_v5
```

**또는** 간단하게 train labels를 val로 복사 (테스트용):
```bash
cp data/emergence_risk_v5_full/risk_labels_train.pkl \
   data/emergence_risk_v5_full/risk_labels_val.pkl
```

## 학습 진행 확인

### TensorBoard
```bash
tensorboard --logdir work_dirs/bevformer_risk_tiny
```

### Log 파일
```bash
tail -f work_dirs/bevformer_risk_tiny/$(ls -t work_dirs/bevformer_risk_tiny/*.log | head -1)
```

## 예상 로그 출력

정상적으로 시작되면 다음과 같은 로그가 출력됩니다:

```
2025-xx-xx xx:xx:xx,xxx - INFO - Environment info:
...
2025-xx-xx xx:xx:xx,xxx - INFO - Config:
...
2025-xx-xx xx:xx:xx,xxx - INFO - Model type: BEVFormerRisk
...
2025-xx-xx xx:xx:xx,xxx - INFO - Start running, work_dir: work_dirs/bevformer_risk_tiny
...
Epoch [1][50/xxxx]  lr: x.xxxxx, loss: x.xxxx, loss_risk: x.xxxx, ...
```

## 주요 Loss 항목

- `loss`: 전체 loss
- `loss_cls`: Classification loss (detection)
- `loss_bbox`: Bounding box loss (detection)
- `loss_risk`: Risk prediction loss (새로 추가)
- `loss_risk_mse`: Risk MSE loss
- `loss_risk_mae`: Risk MAE loss

## Troubleshooting

### 1. "risk_labels_val.pkl not found"
→ Validation labels 생성 또는 train labels 복사

### 2. "CUDA out of memory"
→ Config에서 `samples_per_gpu=1`을 더 줄이거나, GPU 개수 늘리기

### 3. "NuScenesRiskDataset not in registry"
→ `import projects.mmdet3d_plugin` 확인
→ 이미 수정되어 있어야 함

### 4. "img_norm_cfg not defined"
→ 이미 수정되었음 (config 파일에 추가됨)

## GPU 메모리 요구사항

- **Single GPU**: 최소 11GB (RTX 2080 Ti 이상)
- **Recommended**: 24GB (RTX 3090, A5000 등)
- **Multi-GPU**: 각 GPU당 11GB 이상

## 학습 시간 (nuScenes trainval 기준)

- **1 epoch**: ~2-3시간 (8 GPUs)
- **24 epochs**: ~48-72시간 (8 GPUs)

## Next Steps

1. ✅ Config 수정 완료
2. ⏳ Validation labels 생성 (선택)
3. ⏳ 빠른 테스트 (2 epochs)
4. ⏳ 전체 학습 (24 epochs)
5. ⏳ 평가 및 시각화

---

**수정사항 (2025-11-18)**:
- `img_norm_cfg` 정의 추가
- Single/Multi GPU 모두 지원
- Training-only config 추가 (validation 없이 테스트)
