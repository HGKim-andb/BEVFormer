# Full Dataset Transfer Guide

풀 데이터셋 risk labels가 다른 서버(sp@katech)에서 생성되었으므로, 이 서버(hg-main@hgmain-Z490)로 파일을 전송해야 합니다.

## 1. 소스 서버 (sp@katech)에서 확인

```bash
# Risk labels 파일 확인
ls -lh ~/Project/BEVFormer/data/emergence_risk_v5_full/

# 예상 출력:
# risk_labels_train.pkl  (~XXX MB)
# risk_labels_val.pkl    (~XXX MB)
# risk_config.json
```

## 2. 파일 전송 방법

### Option A: scp로 직접 전송 (권장)

소스 서버에서 실행:

```bash
# 전체 디렉토리 전송
scp -r ~/Project/BEVFormer/data/emergence_risk_v5_full/ \
    hg-main@hgmain-Z490:/home/hg-main/data2/BEVFormer/data/

# 또는 개별 파일 전송
scp ~/Project/BEVFormer/data/emergence_risk_v5_full/*.pkl \
    ~/Project/BEVFormer/data/emergence_risk_v5_full/*.json \
    hg-main@hgmain-Z490:/home/hg-main/data2/BEVFormer/data/emergence_risk_v5_full/
```

### Option B: rsync로 전송 (더 안전)

```bash
rsync -avz --progress \
    ~/Project/BEVFormer/data/emergence_risk_v5_full/ \
    hg-main@hgmain-Z490:/home/hg-main/data2/BEVFormer/data/emergence_risk_v5_full/
```

### Option C: 중간 저장소 사용

소스 서버에서:
```bash
# 압축
tar -czf risk_labels_full.tar.gz -C ~/Project/BEVFormer/data emergence_risk_v5_full/

# 공유 스토리지나 NAS로 복사
cp risk_labels_full.tar.gz /path/to/shared/storage/
```

타겟 서버에서:
```bash
# 압축 해제
cd /home/hg-main/data2/BEVFormer/data
tar -xzf /path/to/shared/storage/risk_labels_full.tar.gz
```

## 3. 전송 확인

타겟 서버(현재 서버)에서 확인:

```bash
cd /home/hg-main/data2/BEVFormer

# 파일 존재 확인
ls -lh data/emergence_risk_v5_full/
# 예상: risk_labels_train.pkl, risk_labels_val.pkl, risk_config.json

# 통계 확인
PYTHONPATH=.:$PYTHONPATH /home/hg-main/anaconda3/envs/vad1/bin/python tools/check_risk_labels.py
```

## 4. 예상 결과

전송 완료 후 다음과 같이 표시되어야 합니다:

```
============================================================
TRAIN SET
============================================================
Risk labels file: data/emergence_risk_v5_full/risk_labels_train.pkl
Total scenes: ~700 scenes
Total samples: ~28,000 samples

Risk Map Statistics:
  Max risk: min=0.000, max=1.000, mean=0.XXX
  Mean risk: min=0.000000, max=0.XXXXXX, mean=0.XXXXXX
  Non-zero cells: min=0, max=XXXX, mean=XXX

============================================================
VAL SET
============================================================
Risk labels file: data/emergence_risk_v5_full/risk_labels_val.pkl
Total scenes: ~150 scenes
Total samples: ~6,000 samples

Risk Map Statistics:
  Max risk: min=0.000, max=1.000, mean=0.XXX
  Mean risk: min=0.000000, max=0.XXXXXX, mean=0.XXXXXX
  Non-zero cells: min=0, max=XXXX, mean=XXX
```

## 5. 다음 단계

파일 전송이 완료되면:

1. **설정 확인**:
   ```bash
   grep -r "emergence_risk_v5_full" projects/configs/bevformer/bevformer_risk_tiny.py
   ```

2. **학습 시작**:
   ```bash
   PYTHONPATH=.:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 nohup python tools/train.py \
       projects/configs/bevformer/bevformer_risk_tiny.py \
       --work-dir work_dirs/bevformer_risk_full \
       > train_full.log 2>&1 &
   ```

3. **진행 확인**:
   ```bash
   tail -f train_full.log
   ```

## Troubleshooting

### "Permission denied" 오류

```bash
# 타겟 서버에서 디렉토리 권한 확인
ls -ld data/emergence_risk_v5_full/

# 권한 수정 (필요시)
chmod 755 data/emergence_risk_v5_full/
```

### "No such file or directory"

디렉토리가 없으면 생성:
```bash
mkdir -p /home/hg-main/data2/BEVFormer/data/emergence_risk_v5_full
```

### 파일 크기 확인

전송 전후 파일 크기 비교:
```bash
# 소스 서버
du -sh ~/Project/BEVFormer/data/emergence_risk_v5_full/

# 타겟 서버
du -sh /home/hg-main/data2/BEVFormer/data/emergence_risk_v5_full/
```

---

**참고**: Full dataset risk labels는 mini dataset보다 훨씬 크므로 (약 XX배), 전송 시간이 걸릴 수 있습니다 (약 X-XX분).
