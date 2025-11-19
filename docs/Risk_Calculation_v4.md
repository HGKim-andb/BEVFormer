# Risk Calculation V4 (Temporal Trajectory Awareness)

## 업데이트 날짜: 2025-11-17

---

## 변경 요약

**V3 (Directional) → V4 (Temporal Trajectory)** 으로 Risk Score 계산 방법 개선

### 핵심 변경사항
- **계산 방식**: 시간적 경로 인식 추가 (Temporal Trajectory Awareness)
- **목적**: 이미 지나온 경로(뒤쪽)와 앞으로 갈 경로(앞쪽) 구분
- **성능**: V3와 거의 동일 (방향 페널티와 중복)

---

## V3 vs V4 비교

### Scene-0061 테스트 결과 (39 samples)

| 지표 | V3 (Directional) | V4 (Temporal) | 차이 |
|------|-----------------|---------------|------|
| Max risk (평균) | 0.677 ± 0.081 | **0.677 ± 0.081** | 0% |
| Mean risk (평균) | 0.191 ± 0.030 | **0.191 ± 0.030** | 0% |
| High-risk cells | 111.5 ± 327.7 | **110.5 ± 325.7** | -1% |
| Samples > 0.7 | 8 (20.5%) | **8 (20.5%)** | 0% |

### 왜 V3와 동일한가?

V4의 temporal weighting은 **뒤쪽 경로**의 proximity score를 줄이는데, V3의 directional penalty가 이미 **뒤쪽 셀**의 alignment score를 0으로 만들었기 때문에 중복 효과입니다.

**하지만** V4는 더 세밀한 제어를 제공합니다:
- V3: 뒤쪽(alignment < 0) 전체를 0으로 처리
- V4: 경로상에서 **얼마나 뒤**에 있는지에 따라 exponential decay 적용

---

## V4 알고리즘 설명

### 새로운 기능: Temporal Position

```python
def compute_temporal_position_on_trajectory(cell_pos, ego_state):
    """
    셀의 종방향 위치를 계산

    Returns:
        temporal_pos: < 0 이면 뒤(과거), > 0 이면 앞(미래)
    """
    ego_pos = ego_state['position']
    ego_heading = ego_state['heading']

    to_cell = cell_pos - ego_pos
    heading_vec = [cos(heading), sin(heading)]

    # 진행방향으로 투영 (longitudinal component)
    return dot(to_cell, heading_vec)
```

### Temporal Weighting 적용

기존 proximity score에 temporal weight 적용:

```python
# 기본 proximity 계산 (거리 기반)
if dist_to_traj < 2.0:
    base_proximity = 0.05
elif dist_to_traj < 5.0:
    base_proximity = 0.05 * (5.0 - dist) / 3.0
elif dist_to_traj < 10.0:
    base_proximity = 0.03 * (10.0 - dist) / 5.0
else:
    base_proximity = 0.0

# Temporal weighting 적용
temporal_pos = features['temporal_position_on_trajectory']

if temporal_pos < 0:  # 뒤쪽 (이미 지나온 경로)
    decay = exp(temporal_pos / 5.0)
    temporal_weight = decay * 0.2  # 최대 20%만 적용
else:  # 앞쪽 (앞으로 갈 경로)
    temporal_weight = 1.0  # 100% 적용

proximity_score = base_proximity * temporal_weight
```

### Temporal Weight 예시

| 위치 | temporal_pos | decay | weight | 의미 |
|------|--------------|-------|--------|------|
| 20m 뒤 | -20m | exp(-4) = 0.018 | **0.004** | 거의 0 |
| 10m 뒤 | -10m | exp(-2) = 0.135 | **0.027** | 3% |
| 5m 뒤 | -5m | exp(-1) = 0.368 | **0.074** | 7% |
| Ego | 0m | exp(0) = 1.0 | **1.0** | 100% |
| 5m 앞 | +5m | - | **1.0** | 100% |
| 10m 앞 | +10m | - | **1.0** | 100% |

---

## 실제 계산 예시

### 예시 1: 경로상 5m 뒤 (Behind, on trajectory)

**V3 (Directional)**:
```
Alignment: -0.95 (뒤쪽)
  → alignment_score = 0.0  (뒤쪽이라 0)

Base proximity: 0.05 (경로상 2m 이내)
  → proximity_score = 0.05  (그대로)

Total trajectory: 0.0 + 0.05 = 0.05
```

**V4 (Temporal)**:
```
Alignment: -0.95 (뒤쪽)
  → alignment_score = 0.0  (V3와 동일)

Base proximity: 0.05 (경로상 2m 이내)
Temporal position: -5m (뒤쪽)
  → temporal_weight = exp(-5/5) × 0.2 = 0.074
  → proximity_score = 0.05 × 0.074 = 0.004  (93% 감소!)

Total trajectory: 0.0 + 0.004 = 0.004
```

**결과**: V4가 더 낮은 위험도 (0.05 → 0.004)

---

### 예시 2: 경로상 5m 앞 (Ahead, on trajectory)

**V3 (Directional)**:
```
Alignment: 0.95 (앞쪽)
  → alignment_score = 0.15 × 0.95 = 0.14

Base proximity: 0.05 (경로상 2m 이내)
  → proximity_score = 0.05

Total trajectory: 0.14 + 0.05 = 0.19
```

**V4 (Temporal)**:
```
Alignment: 0.95 (앞쪽)
  → alignment_score = 0.15 × 0.95 = 0.14  (V3와 동일)

Base proximity: 0.05 (경로상 2m 이내)
Temporal position: +5m (앞쪽)
  → temporal_weight = 1.0
  → proximity_score = 0.05 × 1.0 = 0.05  (그대로)

Total trajectory: 0.14 + 0.05 = 0.19
```

**결과**: V4와 V3 동일 (앞쪽은 변화 없음)

---

## 코드 위치

### 파일: `tools/risk_utils.py`

**새 함수: `compute_temporal_position_on_trajectory()`** (Line 654-683)
```python
def compute_temporal_position_on_trajectory(cell_pos: np.ndarray,
                                           ego_state: Dict) -> float:
    """
    Compute cell's longitudinal position relative to ego vehicle

    Returns:
        temporal_pos: < 0 for behind (past), > 0 for ahead (future)
    """
    ego_pos = ego_state['position']
    ego_heading = ego_state['heading']

    to_cell = cell_pos - ego_pos
    heading_vec = np.array([np.cos(ego_heading), np.sin(ego_heading)])

    # Project onto heading (longitudinal component)
    temporal_pos = np.dot(to_cell, heading_vec)

    return temporal_pos
```

**수정: `compute_cell_features()`** (Line 806-808)
```python
# Temporal position on trajectory (behind vs ahead of ego)
temporal_pos = compute_temporal_position_on_trajectory(cell_pos, ego_state)
features['temporal_position_on_trajectory'] = temporal_pos
```

**수정: `compute_risk_score()`** (Line 904-917)
```python
# Apply temporal weighting (이미 지나간 경로는 위험도 낮춤)
temporal_pos = features.get('temporal_position_on_trajectory', 0.0)

if temporal_pos < 0:  # Behind ego (past trajectory)
    decay = np.exp(temporal_pos / 5.0)
    temporal_weight = decay * 0.2
else:  # Ahead of ego (future trajectory)
    temporal_weight = 1.0

proximity_score = base_proximity * temporal_weight
```

---

## Config 설정

**파일: `tools/risk_utils.py`** (Line 19-21)

```python
CONFIG = {
    'version': 'v4_temporal_trajectory',
    # ... rest of config ...
}
```

---

## 사용 방법

### 1. 레이블 생성

```bash
# Mini dataset 테스트
python tools/create_risk_labels.py \
    --dataroot /path/to/nuscenes \
    --version v1.0-mini \
    --output_dir data/emergence_risk_v4 \
    --scenes scene-0061

# Full dataset
python tools/create_risk_labels.py \
    --dataroot /path/to/nuscenes \
    --version v1.0-trainval \
    --output_dir data/emergence_risk_v4
```

### 2. Temporal 기능 디버깅

```bash
# 뒤쪽/앞쪽 셀의 temporal weighting 확인
python tools/debug_temporal.py
```

출력 예시:
```
Position            Temporal    Base Prox     Weight   Final Prox       Risk
--------------------------------------------------------------------------------
Behind 20m            -20.0m       0.0000      0.00x       0.0000     0.1250
Behind 10m            -10.0m       0.0000      0.03x       0.0000     0.1950
Behind 5m              -5.0m       0.0196      0.07x       0.0014     0.2964
At ego                  0.0m       0.0500      1.00x       0.0500     0.4950
Ahead 5m                5.0m       0.0500      1.00x       0.0500     0.4950
Ahead 10m              10.0m       0.0500      1.00x       0.0500     0.3950
```

### 3. 버전 비교

```bash
# V1/V2/V3/V4 모두 비교
python tools/compare_versions.py
```

---

## 전체 버전 히스토리

### V1: Multiplicative (곱셈)
- **문제**: 너무 낮음 (max_risk = 0.054)
- **원인**: 하나의 낮은 factor가 전체 점수를 파괴
- **예시**: 0.3 × 0.4 × 0.013 = 0.0016

### V2: Weighted Sum (가중합)
- **개선**: 곱셈 → 가중합으로 변경
- **문제**: 너무 높음 (max_risk = 0.808, 85% > 0.7)
- **결과**: 15배 증가

### V3: Directional Penalty (방향 페널티)
- **개선**: 가중치 감소 + 뒤쪽 방향 페널티
- **변경**:
  - Occlusion: 0.4 → 0.3
  - Urgency: 0.3 → 0.25
  - Backward alignment: 0 (뒤쪽 = 0)
- **결과**: 적절한 수준 (max_risk = 0.677, 21% > 0.7)

### V4: Temporal Trajectory (시간적 경로)
- **개선**: 경로상 시간적 위치 인식
- **변경**:
  - `temporal_position_on_trajectory` 추가
  - 뒤쪽 경로: exponential decay (exp(pos/5) × 0.2)
  - 앞쪽 경로: 100% weight
- **결과**: V3와 거의 동일 (방향 페널티와 중복)

---

## 향후 개선 사항

### 1. 속도 기반 Temporal Weighting

현재는 고정 decay rate (5.0m)를 사용하지만, 속도에 따라 조정 가능:

```python
# 빠른 속도 → 더 긴 temporal horizon
if ego_velocity > 15:  # > 54 km/h
    decay_distance = 8.0  # 더 멀리까지 고려
elif ego_velocity > 10:  # > 36 km/h
    decay_distance = 5.0  # 현재 기본값
else:  # < 36 km/h
    decay_distance = 3.0  # 더 가까이만 고려

decay = np.exp(temporal_pos / decay_distance)
```

### 2. 차선 변경 시 Temporal 조정

차선 변경 중일 때는 옆쪽 경로도 "미래 경로"로 간주:

```python
if is_lane_changing:
    # 옆쪽 셀도 temporal weight 적용
    lateral_offset = compute_lateral_offset(cell_pos, ego_state)
    if abs(lateral_offset) < lane_width:
        # 차선 변경 방향의 셀은 미래 경로로 처리
        temporal_weight = 1.0
```

### 3. 궤적 예측 통합

현재는 단순히 ego heading만 사용하지만, 실제 궤적 예측 사용 가능:

```python
if ego_state['trajectory']:  # 실제 궤적이 있으면
    # 경로상의 각 점에서 temporal position 계산
    temporal_positions = [
        compute_distance_along_trajectory(cell_pos, trajectory)
        for trajectory in ego_state['trajectory']
    ]
    temporal_pos = min(temporal_positions)
```

---

## 참고 자료

- **V1 코드**: `tools/risk_utils.py.backup`
- **V2 문서**: `docs/Risk_Calculation_v2.md`
- **V3 문서**: (V2.md에 포함)
- **V4 코드**: `tools/risk_utils.py` (current)
- **Temporal 디버그**: `tools/debug_temporal.py`
- **버전 비교**: `tools/compare_versions.py`

---

## 문의

- 이슈 발견 시 프로젝트 저장소에 이슈 등록
- V3로 복원 필요 시: git checkout으로 이전 커밋 복원

---

**작성일**: 2025-11-17
**버전**: v4.0 (Temporal Trajectory Awareness)
