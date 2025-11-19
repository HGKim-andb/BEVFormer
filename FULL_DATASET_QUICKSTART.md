# 🚀 Full Dataset V5 Generation - Quick Start

## 최소 3단계로 완료

### ⚡ Step 1: 첫 배치 테스트 (2-3시간)

```bash
bash tools/generate_full_dataset_batches.sh
```

선택 화면에서:
```
Choice [1/2/3]: 2         # Single batch 선택
Enter batch number (1-10): 1  # 배치 1번 실행
```

**결과 확인**:
```bash
ls -lh data/emergence_risk_v5_full_batch_1/
# risk_labels_train.pkl (~1.5GB) 생성 확인
```

---

### ⚡ Step 2: 전체 배치 실행 (백그라운드, 20시간)

첫 배치가 성공하면 전체 실행:

```bash
# 백그라운드 실행 (터미널 닫아도 계속 실행됨)
nohup bash -c '
for batch in {1..10}; do
    echo "=== Batch $batch started at $(date) ==="
    bash tools/generate_full_dataset_batches.sh <<< "2
$batch"
    echo "=== Batch $batch completed at $(date) ==="
done
' > full_dataset.log 2>&1 &

# 프로세스 ID 확인
echo $!  # 이 번호를 기억해두세요 (종료 시 필요)
```

**진행상황 확인**:
```bash
# 실시간 로그 확인
tail -f full_dataset.log

# 완료된 배치 개수 확인
ls -d data/emergence_risk_v5_full_batch_* | wc -l

# Ctrl+C로 빠져나오기 (프로세스는 계속 실행됨)
```

---

### ⚡ Step 3: 배치 병합 (5분)

모든 배치 완료 후:

```bash
python tools/merge_risk_batches.py \
    --input_dirs data/emergence_risk_v5_full_batch_* \
    --output_dir data/emergence_risk_v5_full
```

**최종 결과**:
```
data/emergence_risk_v5_full/
├── risk_labels_train.pkl  (~15GB, 850 scenes, 34,149 samples)
├── risk_labels_val.pkl    (if val split exists)
└── risk_config.json
```

---

## 📊 예상 타임라인

| 단계 | 시간 | 진행 확인 |
|------|------|-----------|
| 배치 1 (테스트) | 2-3시간 | `ls data/emergence_risk_v5_full_batch_1/` |
| 배치 2-10 (자동) | 18-22시간 | `tail -f full_dataset.log` |
| 병합 | 5분 | `ls -lh data/emergence_risk_v5_full/` |
| **총합** | **~24시간** | |

---

## 🎯 한줄 명령어 (고급)

전체를 한번에 실행하려면:

```bash
# 전체 데이터셋 한번에 (배치 없이)
python tools/create_risk_labels.py \
    --dataroot /home/hg-main/data2/datasets/nuscenes/data/nuscenes \
    --version v1.0-trainval \
    --output_dir data/emergence_risk_v5_full \
    --parallel
```

⚠️ **주의**: 중간에 실패하면 처음부터 다시 시작해야 함!

---

## 🛠️ 문제 발생 시

### 프로세스 중단하기
```bash
# 프로세스 ID로 종료 (Step 2에서 저장한 번호)
kill <PID>

# 또는 전체 python 프로세스 확인 후 종료
ps aux | grep create_risk_labels
kill <해당 PID>
```

### 디스크 공간 부족
```bash
# 현재 사용량 확인
df -h data/

# 필요 공간: 최소 20GB
```

### 메모리 부족
```bash
# create_risk_labels.py 편집
# Line ~90: step = 4  (2 → 4로 변경, 메모리 절약)
```

---

## ✅ 완료 후 시각화

```bash
# 고위험 샘플 50개 시각화
python tools/visualize_risk_samples.py \
    --labels data/emergence_risk_v5_full/risk_labels_train.pkl \
    --dataroot /home/hg-main/data2/datasets/nuscenes/data/nuscenes \
    --version v1.0-trainval \
    --num_samples 50 --min_risk 0.7 \
    --output_dir visualizations/full_v5_top50
```

---

## 📚 더 자세한 정보

- **상세 가이드**: [Full_Dataset_Generation.md](docs/Full_Dataset_Generation.md)
- **V5 알고리즘**: [Risk_Calculation_v5.md](docs/Risk_Calculation_v5.md)
- **문제 해결**: [Full_Dataset_Generation.md#troubleshooting](docs/Full_Dataset_Generation.md#troubleshooting)

---

**작성일**: 2025-11-18
**소요 시간**: 배치 방식 ~24시간 (백그라운드 자동 실행)
