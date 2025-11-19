# Mini Dataset vs Full Dataset 구조 차이

## 문제점

**같은 파일명을 사용합니다!**

### Full Dataset 생성 시
```bash
python tools/create_data.py nuscenes --version v1.0
```
생성되는 파일:
- `data/nuscenes/nuscenes_infos_temporal_train.pkl` (trainval 데이터)
- `data/nuscenes/nuscenes_infos_temporal_test.pkl` (test 데이터)

### Mini Dataset 생성 시
```bash
python tools/create_data.py nuscenes --version v1.0-mini
```
생성되는 파일:
- `data/nuscenes/nuscenes_infos_temporal_train.pkl` (mini_train 데이터) ⚠️ **같은 이름!**
- `data/nuscenes/nuscenes_infos_temporal_val.pkl` (mini_val 데이터)

**결과**: 같은 디렉토리에 두 개를 만들면 **덮어쓰기**됩니다!

---

## 해결 방법

### 방법 1: 다른 디렉토리 사용 (권장)

#### Mini Dataset
```bash
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes_mini \
    --extra-tag nuscenes \
    --version v1.0-mini \
    --canbus ./data
```

#### Full Dataset
```bash
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0 \
    --canbus ./data
```

**설정 파일 수정 필요**:
```python
# mini용 설정 파일 생성 또는 수정
data_root = 'data/nuscenes_mini/'  # mini dataset 경로
```

---

### 방법 2: 다른 Prefix 사용

#### Mini Dataset
```bash
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes_mini \  # 다른 prefix
    --version v1.0-mini \
    --canbus ./data
```

생성되는 파일:
- `data/nuscenes/nuscenes_mini_infos_temporal_train.pkl`
- `data/nuscenes/nuscenes_mini_infos_temporal_val.pkl`

**설정 파일 수정 필요**:
```python
data = dict(
    train=dict(
        ann_file=data_root + 'nuscenes_mini_infos_temporal_train.pkl',  # prefix 변경
        ...
    ),
    val=dict(
        ann_file=data_root + 'nuscenes_mini_infos_temporal_val.pkl',
        ...
    ),
)
```

---

### 방법 3: 설정 파일에서 경로 직접 지정

같은 디렉토리에 두 개를 만들되, 설정 파일에서 다른 파일명 사용:

#### Mini Dataset 생성 (다른 prefix)
```bash
python tools/create_data.py nuscenes \
    --extra-tag nuscenes_mini \
    --version v1.0-mini \
    ...
```

#### 설정 파일 수정
```python
# projects/configs/bevformer/bevformer_tiny_mini.py (새 파일 생성)
_base_ = './bevformer_tiny.py'

data = dict(
    train=dict(
        ann_file='data/nuscenes/nuscenes_mini_infos_temporal_train.pkl',
        ...
    ),
    val=dict(
        ann_file='data/nuscenes/nuscenes_mini_infos_temporal_val.pkl',
        ...
    ),
)
```

---

## 권장 방법

**방법 1 (다른 디렉토리)**을 권장합니다:

### 장점
- 설정 파일 수정 최소화
- 데이터 관리가 명확함
- Full dataset과 충돌 없음

### 단점
- 디스크 공간이 약간 더 필요 (원본 이미지는 공유 가능)

### 구조 예시
```
data/
├── nuscenes/              # Full dataset
│   ├── maps/
│   ├── samples/
│   ├── sweeps/
│   ├── nuscenes_infos_temporal_train.pkl
│   └── nuscenes_infos_temporal_test.pkl
│
└── nuscenes_mini/          # Mini dataset (별도 디렉토리)
    ├── nuscenes_infos_temporal_train.pkl
    └── nuscenes_infos_temporal_val.pkl
```

**참고**: Mini dataset은 원본 이미지 파일(`samples/`, `sweeps/`)을 공유할 수 있습니다. 
`--root-path`는 원본 데이터 경로를 가리키고, `--out-dir`은 info 파일만 저장하는 경로입니다.

---

## 요약

| 방법 | 파일명 충돌 | 설정 파일 수정 | 권장도 |
|------|------------|---------------|--------|
| 방법 1: 다른 디렉토리 | ❌ 없음 | ✅ 최소 (data_root만) | ⭐⭐⭐ |
| 방법 2: 다른 prefix | ❌ 없음 | ⚠️ 중간 (ann_file 경로) | ⭐⭐ |
| 방법 3: 같은 디렉토리 | ⚠️ 있음 | ⚠️ 필요 | ⭐ |

**결론**: **같은 구조를 만들 필요 없습니다**. 다른 디렉토리나 다른 prefix를 사용하여 구분하세요!

