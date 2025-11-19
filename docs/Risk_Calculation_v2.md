# Risk Calculation V2 (가중합 방식)

## 업데이트 날짜: 2025-11-17

---

## 변경 요약

**V1 (곱셈 방식) → V2 (가중합 방식)** 으로 Risk Score 계산 방법 변경

### 핵심 변경사항
- **계산 방식**: 곱셈 → 가중합
- **성능 개선**: 위험도 평균 15~234배 증가
- **High-risk 샘플**: 0% → 84.6%

---

## V1 vs V2 비교

### Scene-0061 테스트 결과 (39 samples)

| 지표 | V1 (곱셈) | V2 (가중합) | 개선율 |
|------|----------|------------|--------|
| Max risk (평균) | 0.054 ± 0.039 | **0.808 ± 0.096** | **×15** |
| Mean risk (평균) | 0.001 ± 0.001 | **0.234 ± 0.040** | **×234** |
| High-risk cells | 0.0 ± 0.0 | **1782.4 ± 2006.1** | **∞** |
| Samples > 0.7 | 0 (0.0%) | **33 (84.6%)** | - |
| Samples > 0.5 | 0 (0.0%) | **39 (100.0%)** | - |
| Samples > 0.3 | 0 (0.0%) | **39 (100.0%)** | - |

### 개별 셀 비교 예시

**Cell (30, 3)** - 차량 뒤 가려진 영역:
- V1: 0.0196 (매우 낮음)
- V2: **0.7295** (높음)
- 개선: **37배**

---

## V2 알고리즘 설명

### 기본 원리

**V1의 문제점:**
```python
# 곱셈 방식: 하나라도 낮으면 전체가 낮아짐
risk = 0.3 × 0.4 × 0.9 × 0.995 × 0.013 × 1.0 × 1.0
     = 0.0014 (너무 낮음!)
```

**V2의 해결책:**
```python
# 가중합 방식: 각 요소가 독립적으로 기여
risk = occlusion(0.4) + urgency(0.3) + trajectory(0.2) + lateral(0.15) + context(0.05)
     = 0.40 + 0.20 + 0.10 + 0.08 + 0.02
     = 0.80 (적절함!)
```

---

## Risk 구성 요소

총 5개 구성요소, 각각 독립적으로 점수 기여:

### 1. Occlusion (가림) - 40%

**최대 점수: 0.4**

```python
if occluded:
    score = 0.4 × occlusion_strength × type_diversity
else:
    score = 0.0
```

**예시:**
- 트럭 뒤 (strength=1.0, diversity=1.0): **0.40**
- 승용차 뒤 (strength=0.6, diversity=0.8): **0.19**
- 가려지지 않음: **0.00**

---

### 2. Urgency (긴급도) - 30%

**최대 점수: 0.3**

거리 기반 긴급도:

| 거리 | 긴급도 | 점수 |
|------|--------|------|
| < 8m | Very close | **0.30** |
| 8-15m | Medium | **0.20** |
| 15-30m | Far | **0.10** |
| > 30m | Very far | **0.05** |

**예시:**
- 5m 앞: **0.30** (매우 위험)
- 12m 앞: **0.20** (중간)
- 25m 앞: **0.10** (낮음)

---

### 3. Trajectory (경로) - 20%

**최대 점수: 0.2**

두 가지 하위 요소:

**3a. Direction Alignment (방향 일치)** - 최대 0.1
```python
alignment_score = 0.1 × cos(angle_to_ego)
```

**3b. Path Proximity (경로 근접도)** - 최대 0.1
```python
if dist_to_path < 2m:  score = 0.10
elif dist_to_path < 5m:  score = 0.07 (linear)
elif dist_to_path < 10m: score = 0.03 (linear)
else: score = 0.00
```

**예시:**
- 정면 2m 이내: **0.20** (0.1 + 0.1)
- 정면 5m: **0.17** (0.1 + 0.07)
- 정면 10m: **0.13** (0.1 + 0.03)
- 옆 10m: **0.00** (0.0 + 0.0)

---

### 4. Lateral (측면) - 15%

**최대 점수: 0.25 (collision boost 포함)**

차선 기반 위험도:

| 위치 | 기본 점수 | Collision Boost | 최대 |
|------|-----------|-----------------|------|
| Same lane (< 1.5m) | 0.15 | - | **0.15** |
| Adjacent lane (1.5-5m) | 0.10 | +0.10 | **0.20** |
| Far lanes (> 5m) | 0.03 | - | **0.03** |

**예시:**
- 같은 차선: **0.15**
- 옆 차선 + 충돌 경로: **0.20**
- 옆 차선: **0.10**
- 먼 차선: **0.03**

---

### 5. Context (상황) - 5%

**최대 점수: 0.05**

두 가지 요소:

**5a. Shadow Depth (가림 깊이)**
```python
if depth < 3m:  +0.02
elif depth < 8m: +0.01
```

**5b. Ego Velocity (차량 속도)**
```python
if velocity > 15 m/s (54 km/h): +0.03
elif velocity > 10 m/s (36 km/h): +0.02
elif velocity > 5 m/s (18 km/h): +0.01
```

**예시:**
- 바로 뒤 + 고속: **0.05** (0.02 + 0.03)
- 중간 뒤 + 중속: **0.03** (0.01 + 0.02)
- 먼 뒤 + 저속: **0.01** (0.00 + 0.01)

---

## 실제 계산 예시

### 예시 1: 트럭 뒤 12m, 정면 2m 이내

```
Occlusion:  0.40  (트럭, strength=1.0, diversity=1.0)
Urgency:    0.20  (12m = medium)
Trajectory: 0.20  (정면 + 경로 위)
Lateral:    0.15  (같은 차선)
Context:    0.03  (중간속도)
──────────────────
Total:      0.98  → clipped to 0.98
```

**최종 위험도: 0.98 (매우 높음)**

---

### 예시 2: 승용차 뒤 25m, 옆 차선

```
Occlusion:  0.24  (승용차, strength=0.6, diversity=1.0)
Urgency:    0.10  (25m = far)
Trajectory: 0.13  (정면 + 10m 떨어짐)
Lateral:    0.10  (옆 차선)
Context:    0.02  (저속)
──────────────────
Total:      0.59
```

**최종 위험도: 0.59 (중간)**

---

### 예시 3: 먼 장애물 40m, 옆 방향

```
Occlusion:  0.10  (작은 장애물)
Urgency:    0.05  (40m = very far)
Trajectory: 0.05  (비스듬한 방향 + 먼 경로)
Lateral:    0.03  (먼 차선)
Context:    0.01  (저속)
──────────────────
Total:      0.24
```

**최종 위험도: 0.24 (낮음)**

---

## 코드 위치

### 파일: `tools/risk_utils.py`

**함수: `compute_risk_score(features)`** (Line 789-902)

```python
def compute_risk_score(features: Dict) -> float:
    """
    Compute final risk score using weighted sum of independent components

    Component breakdown:
        - Occlusion:  40% (0.0-0.4)
        - Urgency:    30% (0.0-0.3)
        - Trajectory: 20% (0.0-0.2)
        - Lateral:    15% (0.0-0.25 with boost)
        - Context:    5%  (0.0-0.05)
        Total max:    100% (1.0, up to 1.1 with boosts)
    """
    # Component 1: Occlusion
    occlusion_score = 0.4 * strength * diversity if occluded else 0.0

    # Component 2: Urgency
    urgency_score = {
        < 8m: 0.30,
        8-15m: 0.20,
        15-30m: 0.10,
        > 30m: 0.05
    }

    # Component 3: Trajectory
    alignment_score = 0.1 * cos(angle)
    proximity_score = 0.1 if < 2m else (decay)
    trajectory_score = alignment_score + proximity_score

    # Component 4: Lateral
    lateral_score = {
        same_lane: 0.15,
        adjacent: 0.10 (+ 0.10 if collision),
        far: 0.03
    }

    # Component 5: Context
    context_score = shadow_depth_factor + velocity_factor

    # Final sum
    total = occlusion + urgency + trajectory + lateral + context
    return clip(total, 0.0, 1.0)
```

---

## Config 설정

**파일: `tools/risk_utils.py`** (Line 19-54)

```python
CONFIG = {
    # Version
    'version': 'v2_weighted_sum',

    # Risk component weights
    'risk_weights': {
        'occlusion': 0.4,   # 40%
        'urgency': 0.3,     # 30%
        'trajectory': 0.2,  # 20%
        'lateral': 0.15,    # 15%
        'context': 0.05,    # 5%
    },

    # ... other configs ...
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
    --output_dir data/emergence_risk_v2 \
    --scenes scene-0061

# Full dataset
python tools/create_risk_labels.py \
    --dataroot /path/to/nuscenes \
    --version v1.0-trainval \
    --output_dir data/emergence_risk_v2
```

### 2. 분석

```bash
python tools/analyze_risk_labels.py \
    --labels data/emergence_risk_v2/risk_labels_train.pkl
```

### 3. 시각화

```bash
# Scene별 시간순 시각화
python tools/visualize_risk_samples.py \
    --labels data/emergence_risk_v2/risk_labels_train.pkl \
    --dataroot /path/to/nuscenes \
    --version v1.0-mini \
    --scenes scene-0061 \
    --by_scene \
    --num_samples 10 \
    --min_risk 0.5 \
    --output_dir visualizations/risk_v2
```

### 4. 디버그

```bash
# 특정 셀의 위험도 계산 과정 확인
python tools/debug_risk.py
```

---

## 백업 및 복원

### V1 백업

```bash
# V1 코드 백업됨
cp tools/risk_utils.py.backup tools/risk_utils.py  # V1로 복원 시
```

### V1/V2 비교 테스트

```bash
# V1 결과
data/emergence_risk_test/risk_labels_train.pkl

# V2 결과
data/emergence_risk_v2/risk_labels_train.pkl
```

---

## 향후 개선 사항

### 1. 가중치 조정

현재 고정값:
```python
occlusion: 0.4  # 조정 가능: 0.3-0.5
urgency: 0.3    # 조정 가능: 0.25-0.35
trajectory: 0.2 # 조정 가능: 0.15-0.25
```

### 2. Urgency 세분화

현재 4단계 → 6단계로 확장:
```python
< 5m: 0.35 (추가)
5-8m: 0.30
8-12m: 0.25 (추가)
12-15m: 0.20
```

### 3. Dynamic Weighting

상황별 가중치 동적 조정:
```python
if high_speed:
    urgency_weight *= 1.2
    trajectory_weight *= 1.1
```

---

## 참고 자료

- **원본 사양**: `tools/new_labeling.md`
- **V1 코드**: `tools/risk_utils.py.backup`
- **V2 코드**: `tools/risk_utils.py`
- **테스트 결과**: `data/emergence_risk_v2/`
- **시각화**: `visualizations/risk_v2_by_scene/`

---

## 문의

- 이슈 발견 시 프로젝트 저장소에 이슈 등록
- V1으로 복원 필요 시: `cp tools/risk_utils.py.backup tools/risk_utils.py`

---

**작성일**: 2025-11-17
**버전**: v2.0 (Weighted Sum Method)
