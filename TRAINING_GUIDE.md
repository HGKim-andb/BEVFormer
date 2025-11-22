# Risk-Guided Attention Training Guide

## Quick Start (Multi-GPU Server)

### 1. NCCL Error 해결

멀티 GPU 학습 시 NCCL 에러가 발생하면 다음 환경 변수를 설정:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

### 2. 학습 시작 (4 GPU 예시)

```bash
# 환경 변수 설정 + 학습 실행
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
bash tools/dist_train.sh \
    projects/configs/bevformer/bevformer_risk_tiny_attention.py \
    4 \
    --work-dir ./work_dirs/bevformer_risk_tiny_attention
```

### 3. 데이터셋 설정

**현재 설정: 20% 데이터셋** (빠른 학습용)
- Train: `risk_labels_train_20pct.pkl` (약 65 samples)
- Val: `risk_labels_val_20pct.pkl` (약 16 samples)

**전체 데이터셋으로 변경하려면:**
Config 파일에서 다음 부분 수정:
```python
risk_labels_path='data/emergence_risk_v5/risk_labels_train.pkl',  # 324 samples
risk_labels_path='data/emergence_risk_v5/risk_labels_val.pkl',    # 80 samples
```

## 학습 설정

- **Epochs**: 6
- **Batch size**: 1 per GPU
- **Learning rate**: 2e-4
- **Attention temp**: 3.0 (sharper attention)
- **Risk loss weight**: 100.0

## 예상 소요 시간

- **20% 데이터**: ~2-3시간 (4 GPU 기준)
- **전체 데이터**: ~1일 (4 GPU 기준)

## 학습 진행 확인

```bash
# 로그 실시간 확인
tail -f work_dirs/bevformer_risk_tiny_attention/*.log

# Tensorboard
tensorboard --logdir work_dirs/bevformer_risk_tiny_attention
```

## 체크포인트

- `epoch_1.pth`, `epoch_2.pth`, ..., `epoch_6.pth`
- 저장 위치: `work_dirs/bevformer_risk_tiny_attention/`

## Inference

학습 완료 후:
```bash
python inference_risk_attention.py \
    --config projects/configs/bevformer/bevformer_risk_tiny_attention.py \
    --checkpoint work_dirs/bevformer_risk_tiny_attention/epoch_6.pth \
    --sample-idx 0 \
    --output-dir inference_outputs
```

## Troubleshooting

### NCCL Error 발생 시
```bash
# 환경 변수 추가
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO  # 디버깅용
```

### CUDA Out of Memory
- GPU 메모리가 부족하면 GPU 개수 늘리기
- 또는 gradient accumulation 사용
