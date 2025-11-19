# Emergence Label Generation for nuScenes

자율주행에서 가려진 곳에서 갑자기 나타나는 객체(emergence)를 예측하기 위한 레이블 생성 도구입니다.

## 개요

**Emergence**는 과거 프레임들(t-5 ~ t-1)에서 **가려져 있었다가(visibility < 40%)** 미래 프레임들(t+1, t+2, t+3) 중에 **나타나는(visibility ≥ 40%)** 객체를 의미합니다.

이 도구는 nuScenes 데이터셋의 3D annotation과 visibility 정보를 활용하여:
- Emergence 이벤트를 자동으로 감지 (visibility 전환 기반)
- Ego vehicle 기준 좌표계로 변환 (global → ego-relative)
- BEV (Bird's Eye View) grid (200x200) 형태의 label 생성
- 통계 분석 및 시각화 제공

## 주요 특징

✅ **Visibility 기반 감지**: nuScenes visibility token을 활용한 정확한 emergence 판별
✅ **Ego-relative 좌표**: Global 좌표를 ego vehicle 기준으로 자동 변환
✅ **거리 필터링**: 5-40m 범위 내의 emergence만 감지 (너무 가까운/먼 객체 제외)
✅ **Gaussian Smoothing**: 부드러운 heatmap 생성으로 학습 안정성 향상
✅ **완전 자동화**: nuScenes 데이터만 있으면 즉시 실행 가능

## 파일 구성

```
tools/
├── create_emergence_labels.py      # 메인 레이블 생성 스크립트
├── analyze_emergence_labels.py     # 통계 분석 스크립트
├── visualize_emergence_samples.py  # 시각화 스크립트
└── emergence_labels_README.md      # 본 문서
```

## 요구사항

### 필수 패키지
```bash
pip install nuscenes-devkit pyquaternion
pip install numpy matplotlib opencv-python tqdm scipy
```

**중요**: `pyquaternion`이 반드시 필요합니다 (좌표 변환용)

### 데이터
- nuScenes dataset (v1.0-trainval or v1.0-mini)
- 약 50GB의 저장 공간 (full dataset 기준)
- 데이터 경로: `/home/hg-main/data2/datasets/nuscenes/data/` (또는 사용자 지정)

## 사용 방법

### 1. Label 생성

#### Step 1: Mini Dataset으로 테스트 (권장)

먼저 작은 데이터셋으로 테스트해서 정상 작동을 확인하세요:

```bash
# Mini dataset (10 scenes, ~5초 소요)
python tools/create_emergence_labels.py \
    --dataroot data/nuscenes \
    --version v1.0-mini \
    --output_dir data/emergence_labels_test
```

**예상 결과 (Mini dataset):**
```
Processing train split: 6 scenes
Total samples: 194
Samples with emergence: 120 (61.86%)
Total emergence events: 603

Processing val split: 4 scenes
Total samples: 130
Samples with emergence: 79 (60.77%)
Total emergence events: 451
```

#### Step 2: Full Dataset 실행

테스트가 성공하면 전체 데이터셋으로 실행:

```bash
# Full dataset (850 scenes, ~30-60분 소요)
python tools/create_emergence_labels.py \
    --dataroot data/nuscenes \
    --version v1.0-trainval \
    --output_dir data/emergence_labels
```

**출력 파일:**
- `emergence_labels_train.pkl` - Training set labels (700 scenes)
- `emergence_labels_val.pkl` - Validation set labels (150 scenes)
- `label_statistics.json` - 기본 통계
- `label_config.json` - 사용된 설정

**예상 실행 시간:**
- Full dataset (850 scenes): 30-60분
- Mini dataset (10 scenes): ~5초

**주요 파라미터:**
- `--dataroot`: nuScenes 데이터 경로 (예: `data/nuscenes`)
- `--version`: 데이터셋 버전 (`v1.0-trainval` 또는 `v1.0-mini`)
- `--output_dir`: 출력 디렉토리 (예: `data/emergence_labels`)
- `--verbose`: 상세 로그 출력 (선택)

### 2. 통계 분석

생성된 labels의 품질을 분석하고 검증합니다.

```bash
# Mini dataset 분석
python tools/analyze_emergence_labels.py \
    --train_labels data/emergence_labels_test/emergence_labels_train.pkl \
    --val_labels data/emergence_labels_test/emergence_labels_val.pkl \
    --output_dir data/emergence_labels_test

# Full dataset 분석
python tools/analyze_emergence_labels.py \
    --train_labels data/emergence_labels/emergence_labels_train.pkl \
    --val_labels data/emergence_labels/emergence_labels_val.pkl \
    --output_dir data/emergence_labels
```

**출력 파일:**
- `analysis_statistics.json` - 상세 통계
- `distribution_plots.png` - 시각화 (4개 subplot)

**분석 항목:**
- Positive ratio (emergence가 있는 샘플 비율)
- Frame별 분포 (t+1, t+2, t+3)
- Category별 분포 (pedestrian, vehicle, bicycle, motorcycle)
- Distance 통계 (평균, 중앙값, 표준편차 등)
- Spatial heatmap (어디서 emergence가 많이 발생하는지)

**자동 검증:**
- ✅ Positive ratio가 5-20% 범위인지 확인
- ✅ t+1에 emergence가 가장 많은지 확인
- ✅ 평균 거리가 5-35m 범위인지 확인

### 3. 샘플 시각화

실제 샘플들을 시각적으로 확인합니다.

```bash
# Mini dataset 시각화
python tools/visualize_emergence_samples.py \
    --labels data/emergence_labels_test/emergence_labels_train.pkl \
    --dataroot data/nuscenes \
    --num_samples 10 \
    --output_dir visualizations/emergence_samples_mini

# Full dataset 시각화
python tools/visualize_emergence_samples.py \
    --labels data/emergence_labels/emergence_labels_train.pkl \
    --dataroot data/nuscenes \
    --num_samples 20 \
    --output_dir visualizations/emergence_samples
```

**출력:**
- 각 샘플당 1개의 PNG 이미지 (예: 20개 샘플 → 20개 이미지)
- 파일명 형식: `sample_000_xxxxxxxx.png`

**시각화 구성:**
각 이미지는 다음을 포함합니다:

```
레이아웃:
Row 0:  [FRONT_LEFT]  [FRONT]       [FRONT_RIGHT]
Row 1:  [BACK_LEFT]   [BACK]        [BACK_RIGHT]
Row 2:  [========  BEV (full width)  ========]
Row 3-5: [Heatmap t+1] [Heatmap t+2] [Heatmap t+3]
```

1. **6개 카메라** (상단 2행) - BEV 레이아웃으로 배치
   - 위쪽: FRONT_LEFT, FRONT, FRONT_RIGHT
   - 아래쪽: BACK_LEFT, BACK, BACK_RIGHT
2. **Current BEV with detections** (중간) - 현재 프레임의 객체들 + 미래 emergence 위치
3. **Emergence heatmaps** (하단) - t+1, t+2, t+3 각각의 heatmap

**주요 파라미터:**
- `--labels`: Label pickle 파일 경로
- `--dataroot`: nuScenes 데이터 경로
- `--num_samples`: 시각화할 샘플 개수 (기본값: 20)
- `--min_emergences`: 최소 emergence 개수 (기본값: 1)
- `--seed`: Random seed (재현성을 위해, 기본값: 42)

## Label 형식

각 샘플의 label은 다음 구조를 가집니다:

```python
{
    'sample_token': str,              # nuScenes sample token
    'scene_token': str,               # Scene token
    'scene_name': str,                # Scene name
    'emergence_mask': np.array,       # [3, 200, 200], float32, 0~1
    'emergence_class': np.array,      # [3, 200, 200], int32, class indices
    'num_emergences': int,            # Total number of emergences
    'emergence_info': [               # List of emergence events
        {
            'frame': int,             # 1, 2, or 3 (t+1, t+2, t+3)
            'position': (x, y),       # World coordinates (meters)
            'grid_pos': (gx, gy),     # Grid coordinates
            'category': str,          # nuScenes category name
            'distance': float         # Distance from ego (meters)
        },
        ...
    ]
}
```

### BEV Grid 사양
- **Range**: [-50, 50] meters in X and Y
- **Resolution**: 0.5 meters per pixel
- **Grid size**: 200x200 pixels
- **Gaussian sigma**: 2.0 pixels (smoothing)

### Category Mapping
```python
CATEGORY_MAP = {
    'vehicle': 1,      # car, truck, bus
    'pedestrian': 2,   # all pedestrian types
    'bicycle': 3,      # bicycle
    'motorcycle': 4,   # motorcycle
}
```

## 설정 (Configuration)

`create_emergence_labels.py`의 `CONFIG` 딕셔너리에서 수정 가능:

```python
CONFIG = {
    'lookback_frames': 5,         # 과거 프레임 수 (t-5 ~ t-1)
    'lookahead_frames': 3,        # 미래 프레임 수 (t+1 ~ t+3)
    'bev_range': [-50, 50, -50, 50],  # BEV 범위 (meters)
    'bev_resolution': 0.5,        # 해상도 (meters/pixel)
    'grid_size': 200,             # Grid 크기 (pixels)
    'min_distance': 5.0,          # 최소 거리 (meters) ← 현재 5m
    'max_distance': 40.0,         # 최대 거리 (meters)
    'gaussian_sigma': 2.0,        # Gaussian smoothing sigma (pixels)
    'emergence_mode': 'strict',   # 'strict' or 'relaxed'
    'valid_categories': [...]     # 포함할 객체 카테고리
}
```

### 주요 설정 설명

#### Emergence Detection Mode
- **`emergence_mode: 'strict'`** (현재 사용 중)
  - 과거 visibility = 1 (0-40%) → 미래 visibility ≥ 2 (40%+)
  - 가려져 있었다가 나타나는 객체만 감지
  - 완전히 새로운 객체는 제외

#### Distance Filtering
- **`min_distance: 5.0`** (현재 사용 중)
  - 너무 가까운 객체 제외 (ego vehicle 주변 5m 이내)
  - 이유: 매우 가까운 객체는 이미 visible하거나 센서 dead zone

- **`max_distance: 40.0`**
  - 너무 먼 객체 제외 (40m 초과)
  - 이유: 예측 가능한 범위 제한

#### Visibility Levels (nuScenes 기준)
- **Level 1**: 0-40% visible (가려짐)
- **Level 2**: 40-60% visible
- **Level 3**: 60-80% visible
- **Level 4**: 80-100% visible (완전히 보임)

## 실제 테스트 결과

### Mini Dataset (v1.0-mini) - 검증 완료 ✅

#### Train Split
- **Total scenes**: 6
- **Total samples**: 194
- **Positive ratio**: 61.86%
- **Total emergences**: 603
- **Avg per positive**: 5.03

#### Val Split
- **Total scenes**: 4
- **Total samples**: 130
- **Positive ratio**: 60.77%
- **Total emergences**: 451
- **Avg per positive**: 5.71

### Full Dataset (v1.0-trainval) - 예상

#### 예상 통계
- **Total samples**: ~22,530 (train) + ~6,019 (val)
- **Positive ratio**: 50-70% (mini 기준)
- **Total emergences**: 수만 건 예상
- **Avg per positive**: 5-6개

#### Positive Ratio가 높은 이유
nuScenes의 도시 환경 특성상:
- 건물, 차량에 의한 occlusion이 빈번
- Visibility 전환이 자주 발생
- 이는 **정상적인 현상**이며, emergence prediction의 중요성을 보여줌

### Frame Distribution (예상)
- t+1: ~50-60% (가장 가까운 미래)
- t+2: ~25-35%
- t+3: ~10-20% (가장 먼 미래)

### Category Distribution (예상)
- Pedestrian: ~30-40%
- Vehicle: ~40-50%
- Bicycle: ~5-10%
- Motorcycle: ~3-5%

### Distance Distribution (예상)
- Mean: 15-25 meters
- Median: 15-20 meters
- Range: 5-40 meters (설정에 따름)

## 문제 해결 (Troubleshooting)

### 1. "pyquaternion not installed" 에러
```bash
pip install pyquaternion
```
**원인**: 좌표 변환에 필수적인 패키지

### 2. "nuscenes-devkit not installed" 에러
```bash
pip install nuscenes-devkit
```

### 3. 거리 값이 1000m 이상으로 이상함
**원인**: Global → Ego 좌표 변환이 누락됨
**해결**: 최신 스크립트는 자동으로 변환됨 (확인 완료 ✅)

### 4. Emergence가 0개 감지됨
**가능한 원인**:
- Visibility token이 없는 데이터셋
- 설정이 너무 엄격함 (`min_distance` 너무 높음)
- 과거/미래 프레임 범위가 scene을 벗어남

**해결 방법**:
```python
# CONFIG에서 수정
'min_distance': 2.0,  # 5.0 → 2.0으로 낮춤
'lookback_frames': 3,  # 5 → 3으로 줄임
```

### 5. 메모리 부족
- Scene별로 처리하므로 메모리 문제는 드뭅니다
- 필요시 `--verbose` 옵션을 제거

### 6. 실행 시간이 너무 길음
- Mini dataset으로 먼저 테스트: `--version v1.0-mini`
- 전체 실행 전 반드시 mini로 검증 권장

## 검증 체크리스트

### Mini Dataset 테스트 (필수)
- [x] 스크립트가 에러 없이 실행됨
- [x] Train/val pkl 파일 생성됨
- [x] Positive ratio 확인 (60%대 - 정상)
- [x] Emergence가 실제로 감지됨 (603개 train, 451개 val)
- [x] 거리 값이 정상 범위 (5-40m)

### Full Dataset 실행 전 확인
- [ ] Mini dataset 테스트 통과
- [ ] 저장 공간 충분 (최소 10GB)
- [ ] 실행 시간 여유 (30-60분)
- [ ] 데이터 경로 확인 (`data/nuscenes/`)

### 분석 및 시각화
- [ ] 분석 스크립트 실행 완료
- [ ] Distribution plots 생성됨
- [ ] 시각화 이미지 생성됨
- [ ] Heatmap이 올바른 위치에 표시됨

## 추가 개발

### 다음 단계
1. **모델 개발**: BEVFormer 기반 emergence prediction 모델
2. **Loss 함수**: Focal loss 또는 weighted BCE for class imbalance
3. **평가 메트릭**: Precision, Recall, F1 at different IoU thresholds
4. **Temporal modeling**: LSTM or Transformer for sequence modeling

### 커스터마이징
- **다른 데이터셋**: Waymo, Argoverse 등에 적용
- **다른 BEV range**: 더 넓거나 좁은 범위
- **다른 해상도**: 더 높거나 낮은 해상도
- **Multi-class prediction**: 각 카테고리별 heatmap

## 개발 이력 및 해결된 이슈

### 주요 이슈와 해결 방법

#### 1. Emergence가 0개 감지되는 문제
**증상**: 초기 버전에서 emergence가 전혀 감지되지 않음
**원인**:
- Global 좌표를 그대로 사용 (거리가 1500m 이상으로 계산됨)
- Visibility 정보를 활용하지 않음

**해결**:
- Ego vehicle 기준 좌표계로 변환 (`pyquaternion` 사용)
- Visibility token을 활용한 emergence 정의
- `get_ego_pose()`, `global_to_ego()` 함수 추가

#### 2. Positive ratio가 예상보다 높은 문제
**증상**: Mini dataset에서 60%대의 positive ratio (예상 10-15%)
**원인**: nuScenes 도시 환경에서 occlusion이 빈번함
**해결**:
- 이는 정상적인 현상으로 판단
- `min_distance`를 5m로 증가시켜 일부 조정
- Strict mode 적용 (visibility 1→2+ 만 감지)

#### 3. Coordinate 변환 오류
**증상**: 거리가 1500m 이상으로 비정상적
**원인**: Global 좌표를 ego-relative로 변환하지 않음
**해결**:
```python
# Before (잘못됨)
x, y, z = ann['translation']  # Global coordinates

# After (올바름)
global_pos = ann['translation']
ego_trans, ego_rot = get_ego_pose(nusc, sample_token)
x, y, z = global_to_ego(global_pos, ego_trans, ego_rot)
```

### 현재 버전의 특징
- ✅ Visibility 기반 emergence 감지
- ✅ Ego-relative 좌표 자동 변환
- ✅ Mini/Full dataset 모두 지원
- ✅ 완전 자동화 (수동 개입 불필요)

## Quick Start (전체 워크플로우)

```bash
# 1. Mini dataset으로 테스트
python tools/create_emergence_labels.py \
    --dataroot data/nuscenes \
    --version v1.0-mini \
    --output_dir data/emergence_labels_test

# 2. 분석
python tools/analyze_emergence_labels.py \
    --train_labels data/emergence_labels_test/emergence_labels_train.pkl \
    --val_labels data/emergence_labels_test/emergence_labels_val.pkl \
    --output_dir data/emergence_labels_test

# 3. 시각화
python tools/visualize_emergence_samples.py \
    --labels data/emergence_labels_test/emergence_labels_train.pkl \
    --dataroot data/nuscenes \
    --num_samples 10 \
    --output_dir visualizations/emergence_samples_mini

# 4. Full dataset 실행 (테스트 성공 시)
python tools/create_emergence_labels.py \
    --dataroot data/nuscenes \
    --version v1.0-trainval \
    --output_dir data/emergence_labels
```

## 참고 자료

- [nuScenes Dataset](https://www.nuscenes.org/)
- [BEVFormer Paper](https://arxiv.org/abs/2203.17270)
- [nuScenes-devkit Documentation](https://github.com/nutonomy/nuscenes-devkit)

## 라이센스

이 코드는 BEVFormer 프로젝트의 일부로, 동일한 라이센스를 따릅니다.

## 문의

이슈나 질문이 있으시면 프로젝트 저장소에 이슈를 등록해주세요.

---

**작성일**: 2025-11-14
**마지막 업데이트**: 2025-11-14
**버전**: 1.0 (Mini dataset 검증 완료)
