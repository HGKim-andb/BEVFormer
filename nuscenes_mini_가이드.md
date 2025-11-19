# nuScenes Mini Dataset 학습 가이드

## 1. 데이터 준비

### 1.1 데이터 다운로드
nuScenes mini dataset을 다운로드합니다. (약 3.8GB)
- 공식 사이트: https://www.nuscenes.org/download
- 또는 nuscenes-devkit을 사용하여 다운로드

### 1.2 데이터 변환
mini dataset을 위한 info 파일을 생성합니다:

```bash
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0-mini \
    --canbus ./data
```

**중요**: `--version v1.0-mini`로 지정해야 합니다.

이 명령어는 다음 파일들을 생성합니다:
- `data/nuscenes/nuscenes_infos_temporal_train.pkl` (mini train)
- `data/nuscenes/nuscenes_infos_temporal_val.pkl` (mini val)

## 2. 설정 파일 확인/수정

기존 설정 파일(`bevformer_tiny.py`, `bevformer_small.py` 등)을 그대로 사용할 수 있습니다. 
파일 경로가 동일하기 때문입니다.

### 2.1 설정 파일 위치
- `projects/configs/bevformer/bevformer_tiny.py` (메모리 절약용)
- `projects/configs/bevformer/bevformer_small.py`
- `projects/configs/bevformer/bevformer_base.py`

### 2.2 설정 파일 확인
설정 파일에서 다음 부분을 확인하세요:

```python
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',  # ✅ 이 파일이 생성됨
        ...
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',  # ✅ 이 파일이 생성됨
        ...
    ),
)
```

## 3. 학습 실행

### 3.1 단일 GPU 학습
```bash
python tools/train.py \
    projects/configs/bevformer/bevformer_tiny.py \
    --work-dir ./work_dirs/bevformer_tiny_mini
```

### 3.2 다중 GPU 학습
```bash
bash tools/dist_train.sh \
    projects/configs/bevformer/bevformer_tiny.py \
    4 \  # GPU 개수
    --work-dir ./work_dirs/bevformer_tiny_mini
```

### 3.3 FP16 학습 (메모리 절약)
```bash
python tools/fp16/train.py \
    projects/configs/bevformer_fp16/bevformer_tiny_fp16.py \
    --work-dir ./work_dirs/bevformer_tiny_fp16_mini
```

## 4. 학습 파라미터 조정 (선택사항)

Mini dataset은 데이터가 적으므로 (약 323개 train, 81개 val), 다음을 고려할 수 있습니다:

### 4.1 Epoch 수 조정
설정 파일에서 `max_epochs`를 줄일 수 있습니다:
```python
runner = dict(type='EpochBasedRunner', max_epochs=12)  # 기본 24에서 줄임
```

### 4.2 Learning Rate 조정
작은 데이터셋이므로 learning rate를 조금 낮출 수 있습니다:
```python
optimizer = dict(type='AdamW', lr=1e-4)  # 기본 2e-4에서 줄임
```

### 4.3 Validation 빈도 증가
더 자주 validation을 수행:
```python
evaluation = dict(interval=2)  # 기본값보다 자주
```

## 5. 검증/테스트

### 5.1 검증 실행
```bash
python tools/test.py \
    projects/configs/bevformer/bevformer_tiny.py \
    ./work_dirs/bevformer_tiny_mini/latest.pth \
    --eval bbox
```

### 5.2 결과 확인
Mini dataset의 경우 전체 dataset과 비교하여 성능이 낮을 수 있습니다 (데이터가 적기 때문).

## 6. 주의사항

1. **데이터 경로**: `data_root = 'data/nuscenes/'`가 올바른지 확인
2. **CAN bus 데이터**: `--canbus` 경로에 CAN bus 데이터가 있는지 확인
3. **메모리**: Mini dataset도 메모리를 많이 사용하므로, `bevformer_tiny` 또는 `bevformer_tiny_fp16` 사용 권장
4. **학습 시간**: Mini dataset은 빠르게 학습되지만, 성능은 제한적입니다

## 7. 트러블슈팅

### 문제: "FileNotFoundError: nuscenes_infos_temporal_train.pkl"
**해결**: 데이터 변환 단계를 다시 실행하세요.

### 문제: "ModuleNotFoundError: nuscenes"
**해결**: nuscenes-devkit 설치
```bash
pip install nuscenes-devkit
```

### 문제: 메모리 부족
**해결**: 
- `bevformer_tiny_fp16.py` 사용
- `samples_per_gpu=1` 확인
- `bev_h`, `bev_w` 크기 줄이기

## 8. 예상 결과

Mini dataset으로 학습한 모델의 성능:
- **NDS**: ~30-40% (전체 dataset: ~50%+)
- **mAP**: ~20-30% (전체 dataset: ~40%+)

이는 정상적인 현상이며, 전체 dataset으로 학습해야 더 높은 성능을 얻을 수 있습니다.

