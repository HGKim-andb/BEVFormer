# Risk-Guided Attention 1주일 실험 계획

## 목표
기존 BEVFormer 모델과 Risk-Guided Attention 모델의 정량적/정성적 비교

## 현재 상황 분석

### 문제점
- **Risk loss가 너무 작음**: `loss_risk = 0.0001` (detection loss의 0.001%)
- **모델이 risk를 학습하지 못함**:
  - Precision/Recall/F1 = 0.0
  - Max risk correlation = -0.57 (음수!)
- **원인**: `risk_loss_weight=10.0`이 너무 작음

### 해결 전략
Risk loss weight를 증가시켜 detection loss와 균형 맞추기

---

## 📅 Day-by-Day 계획

### **Day 1 (오늘): Risk Loss Weight 실험**

#### 실험 1: weight=100 (현재 설정)
```bash
cd ~/Project/BEVFormer2/BEVFormer
git pull

# 3 epoch 학습 (~2시간)
CUDA_VISIBLE_DEVICES=0,1,2 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
bash tools/dist_train.sh \
    projects/configs/bevformer/bevformer_risk_tiny_attention.py \
    3 \
    --work-dir ./work_dirs/risk_w100_3ep
```

#### 성공 기준
- ✅ `loss_risk` > 0.005 (현재의 50배 이상)
- ✅ `grad_norm` < 30 (안정적)
- ✅ `risk_precision` > 0.0 (뭐라도 예측)
- ✅ `risk_max_corr` > -0.3 (개선)

#### 확인 사항
Training log에서:
```
loss_risk_mse: 0.000X
loss_risk_mae: 0.00X
loss_risk: 0.00X  <- 이 값이 0.005 이상이어야 함
grad_norm: XX.XX  <- 30 이하여야 함
```

Evaluation에서:
```
risk_precision: X.XX  <- 0보다 커야 함
risk_recall: X.XX
risk_f1: X.XX
risk_max_corr: X.XX  <- -0.3보다 커야 함
```

---

### **Day 2: 결과 분석 및 조정**

#### Case A: weight=100이 성공적
→ Day 3으로 바로 진행

#### Case B: grad_norm 폭발 (>50)
```python
# Config 수정
risk_loss_weight=50.0
```
재실험 (3 epoch, 2시간)

#### Case C: 여전히 risk loss가 작음 (<0.005)
```python
# Config 수정
risk_loss_weight=500.0
```
재실험 (3 epoch, 2시간)

#### 최종 선택
- Best weight 결정 (50, 100, 또는 500)
- Config 파일에 최종 반영

---

### **Day 3-5: 최적 설정으로 긴 학습 (3일)**

#### Config 최종 설정
```python
# projects/configs/bevformer/bevformer_risk_tiny_attention.py

# Risk configuration
risk_loss_weight=[실험에서 선택된 값]  # 50, 100, or 500

# Training epochs
total_epochs = 12
max_epochs = 12

# Dataset - 전체 데이터 사용
risk_labels_path='data/emergence_risk_v5/risk_labels_train.pkl'  # 324 samples (20% → 100%)
```

#### 학습 실행
```bash
# 새 work_dir로 12 epoch 학습
CUDA_VISIBLE_DEVICES=0,1,2 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
bash tools/dist_train.sh \
    projects/configs/bevformer/bevformer_risk_tiny_attention.py \
    3 \
    --work-dir ./work_dirs/risk_final_12ep
```

#### 예상 소요 시간
- 20% 데이터 (65 samples): ~500 iter/epoch × 12 = 6000 iter
- 전체 데이터 (324 samples): ~2350 iter/epoch × 12 = 28200 iter
- **예상 시간**: ~3일 (3 GPU 기준)

#### 모니터링
```bash
# 실시간 로그 확인
tail -f work_dirs/risk_final_12ep/*.log

# 주요 확인 사항
# - loss_risk가 점진적으로 감소하는지
# - risk_precision/recall이 증가하는지
# - grad_norm이 안정적인지
```

---

### **Day 6: 모델 평가**

#### 1. 새 모델 평가
```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
bash tools/dist_test.sh \
    projects/configs/bevformer/bevformer_risk_tiny_attention.py \
    work_dirs/risk_final_12ep/epoch_12.pth \
    3 \
    --eval bbox
```

**예상 결과:**
```
Detection Metrics:
  mAP: X.XXX
  NDS: X.XXX
  mATE, mASE, mAOE, mAVE, mAAE: X.XXX

Risk Metrics:
  risk_mse: X.XXXX
  risk_rmse: X.XXXX
  risk_mae: X.XXXX
  risk_precision: X.XXXX  <- 목표: > 0.3
  risk_recall: X.XXXX     <- 목표: > 0.3
  risk_f1: X.XXXX         <- 목표: > 0.3
  risk_max_corr: X.XXXX   <- 목표: > 0.5
```

#### 2. 기존 모델 평가 (비교용)
```bash
# 기존 BEVFormer 체크포인트로 평가
CUDA_VISIBLE_DEVICES=0,1,2 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
bash tools/dist_test.sh \
    [기존 config 경로] \
    [기존 checkpoint 경로] \
    3 \
    --eval bbox
```

#### 3. 결과 비교 테이블 작성
| Metric | Baseline | Risk-Guided | Improvement |
|--------|----------|-------------|-------------|
| mAP | X.XXX | X.XXX | +X.X% |
| NDS | X.XXX | X.XXX | +X.X% |
| Risk MSE | - | X.XXX | - |
| Risk Precision | - | X.XXX | - |
| Risk Recall | - | X.XXX | - |
| Risk F1 | - | X.XXX | - |

---

### **Day 7: 분석 및 시각화**

#### 1. Inference 및 Visualization
```bash
# Risk map visualization
python inference_risk_attention.py \
    --config projects/configs/bevformer/bevformer_risk_tiny_attention.py \
    --checkpoint work_dirs/risk_final_12ep/epoch_12.pth \
    --sample-idx 0 \
    --output-dir visualizations/risk_final
```

#### 2. 정성적 분석
- [ ] Risk map quality 확인
- [ ] Attention map visualization
- [ ] High-risk 시나리오에서 성능 비교
- [ ] False positive/negative 분석

#### 3. 최종 보고서 작성
**포함 내용:**
1. 실험 설정 및 동기
2. Weight tuning 과정 및 결과
3. 정량적 성능 비교
4. 정성적 분석 (시각화)
5. 결론 및 향후 개선 방향

---

## 체크리스트

### Day 1
- [x] weight=100 실험 완료 - loss_risk=0.0004 (너무 작음)
- [x] weight=1000 실험 완료 - grad_norm=inf (폭발!)
- [ ] weight=500 실험 진행중
- [ ] Training log 분석
- [ ] 성공 기준 달성 여부 확인
- [ ] 다음 단계 결정

### Day 2
- [ ] 추가 실험 (필요시)
- [ ] Best weight 선택
- [ ] Config 최종 수정
- [ ] Git commit & push

### Day 3-5
- [ ] 긴 학습 시작
- [ ] 매일 로그 확인
- [ ] Loss 감소 추이 모니터링

### Day 6
- [ ] 새 모델 평가
- [ ] 기존 모델 평가
- [ ] 비교 테이블 작성

### Day 7
- [ ] Visualization 생성
- [ ] 정성적 분석
- [ ] 최종 보고서 작성

---

## 예상 타임라인

| Day | Task | Duration | Output |
|-----|------|----------|--------|
| 1 | Weight=100 실험 | 2시간 | Training log, metrics |
| 2 | 추가 실험 & 조정 | 2-4시간 | Best weight, final config |
| 3-5 | 긴 학습 (12 epoch) | 3일 | Trained model checkpoint |
| 6 | 평가 | 4시간 | Metrics comparison table |
| 7 | 분석 & 시각화 | 1일 | Final report & visualizations |

**Total**: 7일

---

## 중요 파일 위치

### Config
- Main: `projects/configs/bevformer/bevformer_risk_tiny_attention.py`
- Base: `projects/configs/bevformer/bevformer_tiny.py`

### Checkpoints
- Experiments: `work_dirs/risk_w[100|50|500]_3ep/`
- Final: `work_dirs/risk_final_12ep/`

### Logs
- Training: `work_dirs/*/[timestamp].log`
- Tensorboard: `work_dirs/*/[timestamp]/`

### Code
- Dataset: `projects/mmdet3d_plugin/datasets/nuscenes_risk_dataset.py`
- Model: `projects/mmdet3d_plugin/bevformer/detectors/bevformer_risk.py`
- Risk Head: `projects/mmdet3d_plugin/bevformer/dense_heads/risk_head.py`

---

## 트러블슈팅

### Gradient Explosion
```python
# Config 수정
risk_loss_weight = [현재값 / 2]
```

### Risk Loss 여전히 작음
```python
# Config 수정
risk_loss_weight = [현재값 × 5]
```

### CUDA OOM
```python
# Config 수정
samples_per_gpu = 2  # 4 → 2
```

### NCCL Error
```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
```

---

## 연락처 & 문서

- Training Guide: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- Python Path Fix: [FIX_PYTHON_PATH.md](FIX_PYTHON_PATH.md)
- Git Repo: https://github.com/HGKim-andb/BEVFormer

---

## 업데이트 로그

- 2025-11-29: 초기 계획 작성
- Risk loss weight: 10 → 100 (Day 1 실험)
