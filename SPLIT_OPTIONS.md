# Train/Val Split 옵션 비교

## Option 1: 간단한 8:2 Split (현재 사용 중)

**위치**: `data/emergence_risk_v5_full/`

```
Train: 8 scenes, 324 samples (80%)
Val:   2 scenes, 80 samples (20%)
```

**장점**:
- ✅ 더 많은 학습 데이터 (324 vs 162)
- ✅ 학습에 유리
- ✅ Mini dataset은 어차피 테스트용

**단점**:
- ❌ nuScenes 공식 split과 다름
- ❌ Val set이 작음 (80 samples)

---

## Option 2: 공식 Split 기반 (더 정확)

**위치**: `data/emergence_risk_v5_official/`

```
Train: 4 scenes, 162 samples (50%)
Val:   4 scenes, 162 samples (50%)
```

**장점**:
- ✅ nuScenes 공식 split 준수
- ✅ Full dataset으로 확장 시 일관성 유지
- ✅ 균형잡힌 split

**단점**:
- ❌ 학습 데이터가 적음 (162 vs 324)
- ❌ Mini dataset에서는 과도하게 작음

---

## 권장사항

### Mini Dataset (현재)
→ **Option 1 (8:2 split)** 사용 권장
- Mini dataset은 빠른 테스트/개발용
- 더 많은 학습 데이터가 유리
- 현재 설정 그대로 사용

### Full Dataset (나중에)
→ **Option 2 (공식 split)** 사용 필수
- 공식 split 기반으로 risk labels 생성
- 논문 실험 시 재현 가능성 보장

---

## 사용 방법

### Option 1 사용 (현재 - 변경 불필요)
```bash
# Config는 이미 올바른 경로 사용
./tools/dist_train.sh \
    projects/configs/bevformer/bevformer_risk_tiny.py \
    8 \
    --work-dir work_dirs/bevformer_risk_tiny
```

### Option 2로 변경하려면
1. Config 수정:
```python
# projects/configs/bevformer/bevformer_risk_tiny.py
data = dict(
    train=dict(
        risk_labels_path='data/emergence_risk_v5_official/risk_labels_train.pkl',
    ),
    val=dict(
        risk_labels_path='data/emergence_risk_v5_official/risk_labels_val.pkl',
    ),
)
```

2. 학습 시작:
```bash
./tools/dist_train.sh \
    projects/configs/bevformer/bevformer_risk_tiny.py \
    8 \
    --work-dir work_dirs/bevformer_risk_tiny_official
```

---

## 결론

**현재 mini dataset으로 개발/테스트하는 동안**: Option 1 (8:2) 사용 ✅

**Full dataset으로 본격 학습할 때**: Option 2 (공식 split) 사용 ✅

