# BEV-RiskViz Quick Start Guide

이 가이드는 BEV Risk Map Generator를 빠르게 시작할 수 있도록 도와드립니다.

## 1분 시작 가이드

### Step 1: 설치 확인

```bash
cd /home/hg-main/data2/BEVFormer

# 예제 스크립트 실행
PYTHONPATH=.:$PYTHONPATH python tools/bev_risk_viz/example_usage.py
```

성공하면 5개의 시각화 파일이 생성됩니다:
- ✅ `example_1_simple.png` - 기본 리스크 맵
- ✅ `example_2_breakdown.png` - 리스크 팩터 분석
- ✅ `example_3_comparison.png` - 파라미터 비교
- ✅ `example_4_*` - 다양한 포맷 (PNG, NPY, CSV, PDF)
- ✅ `example_5_velocity_impact.png` - 속도 영향 분석

---

## 주요 사용 방법

### 방법 1: 데모 시나리오 실행

```bash
# 간단한 데모
PYTHONPATH=.:$PYTHONPATH python tools/bev_risk_viz/cli.py \
    --mode demo \
    --demo-scenario "Multi-Vehicle Intersection" \
    --export png,pdf
```

**사용 가능한 시나리오**:
1. `Simple Occlusion` - 단순 차폐 상황
2. `Multi-Vehicle Intersection` - 다중 차량 교차로
3. `Parking Lot Exit` - 주차장 출구
4. `Highway Merge` - 고속도로 합류
5. `Pedestrian Crossing` - 보행자 횡단

---

### 방법 2: Python API 사용

```python
import numpy as np
from tools.bev_risk_viz import RiskCalculationEngine, RiskConfig
from tools.bev_risk_viz import RiskVisualizer, RiskDataExporter

# 1. 설정 생성
config = RiskConfig(
    weight_trajectory=0.3,      # α - 궤적 정렬
    weight_occlusion=0.3,       # β - 차폐 심각도
    weight_temporal=0.2,        # γ - 시간 긴급도
    weight_proximity=0.2,       # δ - 근접도
    bev_x_range=(-50, 50),      # BEV X 범위 (미터)
    bev_y_range=(-50, 50),      # BEV Y 범위 (미터)
    bev_resolution=0.5,         # 그리드 해상도 (미터)
    ego_velocity=10.0,          # 자차 속도 (m/s)
    ego_heading=0.0             # 자차 방향 (라디안)
)

# 2. 엔진 초기화
engine = RiskCalculationEngine(config)

# 3. 차폐 마스크 생성 (예시: 앞쪽에 차량)
H, W = engine.bev_height, engine.bev_width
occlusion_mask = np.zeros((H, W), dtype=np.float32)
center = (H//2, W//2)
occlusion_mask[center[0]+20:center[0]+40, center[1]-10:center[1]+10] = 1.0

# 4. 리스크 계산
risk_results = engine.calculate_risk_map(occlusion_mask)

# 5. 시각화
visualizer = RiskVisualizer()
visualizer.plot_risk_heatmap(
    risk_results['risk_map'],
    title='내 리스크 맵'
)

# 6. 내보내기
exporter = RiskDataExporter(output_dir='my_exports')
exporter.export_png(risk_results['risk_map'], 'my_risk_map')
exporter.export_pdf_report(risk_results, 'my_report')
```

---

### 방법 3: 대화형 GUI

```bash
# Streamlit 앱 실행
streamlit run tools/bev_risk_viz/gui_app.py
```

웹 브라우저에서 http://localhost:8501 접속

**GUI 기능**:
- 🎛️ 실시간 파라미터 조절 (슬라이더)
- 📊 리스크 팩터 분석 (θ, O, T, P)
- 📈 통계 정보 및 분포
- 💾 다양한 형식으로 내보내기
- 🎬 nuScenes 데이터 탐색

---

## 파라미터 조정 가이드

### 리스크 팩터 가중치 (α, β, γ, δ)

```yaml
risk_weights:
  trajectory_alignment: 0.3  # α - 궤적과의 정렬 (직진 경로)
  occlusion_severity: 0.3    # β - 차폐 영역의 심각도
  temporal_urgency: 0.2      # γ - 시간적 긴급도 (충돌까지의 시간)
  proximity: 0.2             # δ - 자차와의 거리
```

**추천 설정**:

| 시나리오 | α (궤적) | β (차폐) | γ (시간) | δ (근접) |
|---------|---------|---------|---------|---------|
| 고속 주행 | 0.4 | 0.2 | 0.3 | 0.1 |
| 시내 주행 | 0.3 | 0.3 | 0.2 | 0.2 |
| 주차장 | 0.2 | 0.4 | 0.1 | 0.3 |
| 교차로 | 0.3 | 0.4 | 0.2 | 0.1 |

### BEV 그리드 설정

```yaml
bev_grid:
  x_range: {min: -50.0, max: 50.0}  # 좌우 범위
  y_range: {min: -50.0, max: 50.0}  # 전후 범위
  resolution: 0.5                    # 셀 크기 (미터)
```

**해상도 선택**:
- `0.1m` - 매우 높은 정밀도 (느림, 메모리 많이 사용)
- `0.5m` - **권장** (좋은 균형)
- `1.0m` - 빠른 처리 (낮은 정밀도)

---

## 실전 예제

### 예제 1: 차폐 기반 위험 분석

```python
from tools.bev_risk_viz import *
import numpy as np

# 설정
config = RiskConfig(
    weight_occlusion=0.5,  # 차폐에 높은 가중치
    weight_trajectory=0.3,
    bev_resolution=0.5
)
engine = RiskCalculationEngine(config)

# 복잡한 차폐 시나리오 (주차된 차량들)
H, W = engine.bev_height, engine.bev_width
occlusion = np.zeros((H, W))

# 왼쪽에 주차된 차량들
for i in range(5):
    y = H//2 + 20 + i*20
    occlusion[y:y+15, W//2-25:W//2-15] = 1.0

# 오른쪽에 주차된 차량들
for i in range(5):
    y = H//2 + 20 + i*20
    occlusion[y:y+15, W//2+15:W//2+25] = 1.0

# 리스크 계산
results = engine.calculate_risk_map(occlusion)

# 시각화
visualizer = RiskVisualizer()
fig = visualizer.plot_factor_breakdown(results)

# 저장
exporter = RiskDataExporter()
exporter.export_pdf_report(results, 'parking_lot_risk_analysis')
```

### 예제 2: nuScenes 데이터 처리

```python
from tools.bev_risk_viz import NuScenesLoader, RiskCalculationEngine

# nuScenes 로더 초기화
loader = NuScenesLoader(
    data_root='data/nuscenes',
    version='v1.0-mini'
)

# 씬 선택
scenes = loader.get_scene_list()
scene = scenes[0]
frames = loader.get_scene_frames(scene['token'], max_frames=10)

# 첫 프레임 처리
frame_data = loader.load_frame(frames[0])

# 객체로부터 차폐 마스크 생성
occlusion = loader.create_occlusion_mask_from_objects(
    frame_data.annotations,
    bev_range=(-50, 50, -50, 50),
    bev_resolution=0.5
)

# 실제 자차 속도 사용
ego_speed, ego_heading = loader.get_ego_velocity(frames[0])

# 리스크 계산
engine = RiskCalculationEngine()
results = engine.calculate_risk_map(
    occlusion,
    ego_velocity=ego_speed,
    ego_heading=ego_heading
)

print(f"Scene: {scene['name']}")
print(f"Max Risk: {results['risk_map'].max():.3f}")
```

### 예제 3: 배치 처리

```bash
# CLI로 전체 씬 처리
PYTHONPATH=.:$PYTHONPATH python tools/bev_risk_viz/cli.py \
    --mode nuscenes \
    --scene scene-0001 \
    --batch \
    --output-dir batch_results \
    --export png,npy
```

---

## 출력 파일 형식

### PNG 이미지
- **용도**: 논문, 발표, 보고서
- **해상도**: 150 DPI (설정 가능)
- **컬러맵**: 녹색(안전) → 노란색(중간) → 빨간색(위험)

### NumPy 배열
```python
# .npz 파일 로드
data = np.load('risk_map.npz')
risk_map = data['risk_map']      # 최종 리스크 맵
theta = data['theta']            # 궤적 정렬 팩터
O = data['O']                    # 차폐 심각도 팩터
T = data['T']                    # 시간 긴급도 팩터
P = data['P']                    # 근접도 팩터
```

### CSV 데이터
```csv
X,Y,Risk
-50.00,-50.00,0.123456
-49.50,-50.00,0.234567
-49.00,-50.00,0.345678
...
```

### PDF 보고서
- Page 1: 메타데이터 및 설정
- Page 2: 메인 리스크 히트맵
- Page 3: 팩터 분석 (6개 서브플롯)
- Page 4: 통계 분석 및 분포

---

## 설정 파일 사용

### config.yaml 편집

```bash
# 기본 설정 파일 복사
cp tools/bev_risk_viz/config.yaml my_config.yaml

# 편집
nano my_config.yaml

# 사용
PYTHONPATH=.:$PYTHONPATH python tools/bev_risk_viz/cli.py \
    --config my_config.yaml \
    --mode demo
```

---

## 자주 묻는 질문 (FAQ)

### Q1: 리스크 값의 의미는?
**A**: 리스크 값은 0~1 범위로 정규화됩니다:
- `0.0 - 0.3`: 낮은 위험 (녹색)
- `0.3 - 0.7`: 중간 위험 (노란색)
- `0.7 - 1.0`: 높은 위험 (빨간색)

### Q2: 그리드 해상도를 어떻게 선택하나?
**A**:
- 빠른 테스트: `1.0m`
- 일반 사용: `0.5m` (권장)
- 높은 정밀도: `0.2m` 이하

### Q3: 메모리 에러가 발생하면?
**A**: `config.yaml`에서 해상도를 높이세요:
```yaml
bev_grid:
  resolution: 1.0  # 0.5에서 1.0으로
```

### Q4: 어떤 파라미터가 가장 중요한가?
**A**: 시나리오에 따라 다릅니다:
- 고속도로: `weight_trajectory`, `weight_temporal`
- 주차장: `weight_occlusion`, `weight_proximity`
- 교차로: `weight_occlusion`, `weight_temporal`

### Q5: nuScenes 없이 사용 가능한가?
**A**: 네! 데모 모드나 커스텀 시나리오로 사용 가능합니다.

---

## 다음 단계

1. ✅ **예제 실행**: `example_usage.py`로 기본 기능 확인
2. 📖 **문서 읽기**: `README.md`에서 전체 API 확인
3. 🎮 **GUI 시도**: Streamlit 앱으로 인터랙티브 탐색
4. 🔧 **파라미터 조정**: `config.yaml` 커스터마이징
5. 📊 **데이터 분석**: nuScenes 데이터로 실전 분석

---

## 추가 리소스

- **전체 문서**: [README.md](README.md)
- **설치 가이드**: [INSTALL.md](INSTALL.md)
- **프로젝트 요약**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- **설정 파일**: [config.yaml](config.yaml)

## 도움말

```bash
# CLI 도움말
python tools/bev_risk_viz/cli.py --help

# 설정 검증
python tools/bev_risk_viz/config_loader.py

# 예제 스크립트
python tools/bev_risk_viz/example_usage.py
```

---

**즐거운 리스크 분석 되세요! 🚗💨**
