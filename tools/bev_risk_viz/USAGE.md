# BEV-RiskViz 사용 가이드

## 🚀 빠른 시작 (3가지 방법)

### 방법 1: 간편 스크립트 (권장)

```bash
# 데모 실행
./run_bev_riskviz.sh demo

# 특정 시나리오 선택
./run_bev_riskviz.sh demo "Parking Lot Exit"

# 예제 스크립트 실행 (5가지 예제)
./run_bev_riskviz.sh example

# GUI 실행
./run_bev_riskviz.sh gui

# nuScenes 씬 처리
./run_bev_riskviz.sh nuscenes scene-0001
```

### 방법 2: 직접 Python 실행

```bash
# 예제 스크립트
PYTHONPATH=.:$PYTHONPATH python tools/bev_risk_viz/example_usage.py

# CLI 데모
PYTHONPATH=.:$PYTHONPATH python tools/bev_risk_viz/cli.py --mode demo --export png,pdf

# GUI 실행
streamlit run tools/bev_risk_viz/gui_app.py
```

### 방법 3: Python API

```python
import sys
sys.path.insert(0, '.')

from tools.bev_risk_viz import RiskCalculationEngine, RiskConfig
import numpy as np

# 설정
config = RiskConfig(
    weight_trajectory=0.3,
    weight_occlusion=0.3,
    weight_temporal=0.2,
    weight_proximity=0.2,
    ego_velocity=10.0
)

# 엔진 생성
engine = RiskCalculationEngine(config)

# 차폐 마스크 생성
H, W = engine.bev_height, engine.bev_width
occlusion = np.zeros((H, W))
occlusion[90:110, 110:130] = 1.0

# 리스크 계산
results = engine.calculate_risk_map(occlusion)

print(f"Max Risk: {results['risk_map'].max():.3f}")
```

---

## 📋 명령어 상세

### 1. 데모 모드

```bash
# 기본 데모
./run_bev_riskviz.sh demo

# 시나리오 선택
./run_bev_riskviz.sh demo "Simple Occlusion"
./run_bev_riskviz.sh demo "Multi-Vehicle Intersection"
./run_bev_riskviz.sh demo "Parking Lot Exit"
./run_bev_riskviz.sh demo "Highway Merge"
./run_bev_riskviz.sh demo "Pedestrian Crossing"
```

**생성 파일**:
- `exports/[시나리오]_risk_map.png` - 리스크 히트맵
- `exports/[시나리오]_report.pdf` - 분석 리포트

### 2. 예제 스크립트

```bash
./run_bev_riskviz.sh example
```

**생성 파일** (8개):
- `example_1_simple.png` - 기본 리스크 맵
- `example_2_breakdown.png` - 팩터 분석 (6-panel)
- `example_3_comparison.png` - 파라미터 비교
- `example_4_risk_map.png` - PNG 내보내기
- `example_4_risk_data.npz` - NumPy 데이터
- `example_4_risk_map.csv` - CSV 데이터
- `example_4_risk_report.pdf` - PDF 리포트
- `example_5_velocity_impact.png` - 속도 영향 분석

### 3. 대화형 GUI

```bash
./run_bev_riskviz.sh gui
```

**기능**:
- ✅ 실시간 파라미터 조절 (슬라이더)
- ✅ 리스크 팩터 분석 (θ, O, T, P)
- ✅ 통계 정보 및 분포
- ✅ 다양한 형식 내보내기
- ✅ nuScenes 데이터 탐색

**접속**: http://localhost:8501

### 4. nuScenes 모드

```bash
# 단일 프레임
./run_bev_riskviz.sh nuscenes scene-0001

# 또는 CLI로 배치 처리
./run_bev_riskviz.sh cli \
    --mode nuscenes \
    --scene scene-0001 \
    --batch \
    --export png,npy
```

### 5. CLI 고급 사용

```bash
# 모든 옵션 보기
./run_bev_riskviz.sh cli --help

# 커스텀 설정 파일 사용
./run_bev_riskviz.sh cli \
    --config my_config.yaml \
    --mode demo \
    --export png,pdf,npy,csv

# 출력 디렉토리 지정
./run_bev_riskviz.sh cli \
    --mode demo \
    --output-dir my_outputs \
    --export png,pdf
```

---

## 🎨 시각화 옵션

### PNG 이미지
- 고해상도 (150 DPI 기본)
- 커스텀 컬러맵 (녹색→노랑→빨강)
- 논문/발표 자료용

### PDF 리포트
- 4페이지 종합 분석
  - Page 1: 메타데이터 및 설정
  - Page 2: 메인 리스크 히트맵
  - Page 3: 팩터 분석 (θ, O, T, P)
  - Page 4: 통계 분석

### NumPy 배열
```python
# .npz 파일 로드
import numpy as np
data = np.load('risk_map.npz')

risk_map = data['risk_map']  # 최종 리스크
theta = data['theta']        # 궤적 정렬
O = data['O']                # 차폐 심각도
T = data['T']                # 시간 긴급도
P = data['P']                # 근접도
```

### CSV 데이터
```csv
X,Y,Risk
-50.00,-50.00,0.123456
-49.50,-50.00,0.234567
...
```

---

## ⚙️ 파라미터 조정

### 설정 파일 사용

```bash
# 1. 설정 파일 복사
cp tools/bev_risk_viz/config.yaml my_config.yaml

# 2. 편집
nano my_config.yaml

# 3. 사용
./run_bev_riskviz.sh cli --config my_config.yaml --mode demo
```

### 주요 파라미터

#### 리스크 가중치
```yaml
risk_weights:
  trajectory_alignment: 0.3  # α
  occlusion_severity: 0.3    # β
  temporal_urgency: 0.2      # γ
  proximity: 0.2             # δ
```

#### BEV 그리드
```yaml
bev_grid:
  x_range: {min: -50.0, max: 50.0}
  y_range: {min: -50.0, max: 50.0}
  resolution: 0.5  # 0.1~1.0 권장
```

#### 자차 상태
```yaml
ego_vehicle:
  velocity: 10.0  # m/s
  heading: 0.0    # radians
```

---

## 🔧 문제 해결

### 1. 모듈 임포트 에러

```bash
# 해결: PYTHONPATH 설정
export PYTHONPATH=.:$PYTHONPATH
python tools/bev_risk_viz/cli.py --help
```

또는 스크립트 사용:
```bash
./run_bev_riskviz.sh demo  # 자동으로 PYTHONPATH 설정
```

### 2. Streamlit 없음

```bash
pip install streamlit
```

### 3. nuScenes 없음

```bash
pip install nuscenes-devkit
```

### 4. 메모리 에러

설정 파일에서 해상도 증가:
```yaml
bev_grid:
  resolution: 1.0  # 0.5 → 1.0
```

---

## 📊 출력 파일 위치

기본 출력 디렉토리: `exports/`

```
exports/
├── Multi-Vehicle_Intersection_risk_map.png
├── Multi-Vehicle_Intersection_report.pdf
└── ...

example_1_simple.png
example_2_breakdown.png
...
```

커스텀 디렉토리 지정:
```bash
./run_bev_riskviz.sh cli --mode demo --output-dir my_results
```

---

## 💡 실전 활용 예시

### 예시 1: 빠른 데모 테스트

```bash
# 한 줄로 데모 실행 및 결과 확인
./run_bev_riskviz.sh demo && open exports/Multi-Vehicle_Intersection_risk_map.png
```

### 예시 2: 파라미터 비교

```python
from tools.bev_risk_viz import RiskCalculationEngine, RiskConfig

# 3가지 설정 비교
configs = [
    RiskConfig(weight_trajectory=0.5, weight_occlusion=0.3),
    RiskConfig(weight_trajectory=0.3, weight_occlusion=0.5),
    RiskConfig(weight_trajectory=0.25, weight_occlusion=0.25),
]

for i, cfg in enumerate(configs):
    engine = RiskCalculationEngine(cfg)
    results = engine.calculate_risk_map(occlusion)
    print(f"Config {i}: Max={results['risk_map'].max():.3f}")
```

### 예시 3: 배치 처리

```bash
# 여러 씬을 순차 처리
for scene in scene-0001 scene-0002 scene-0003; do
    ./run_bev_riskviz.sh nuscenes $scene
done
```

---

## 📚 추가 문서

- **전체 문서**: [README.md](README.md)
- **빠른 시작**: [QUICK_START.md](QUICK_START.md) (한글)
- **설치 가이드**: [INSTALL.md](INSTALL.md)
- **프로젝트 요약**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

---

**도움이 필요하면**:
```bash
./run_bev_riskviz.sh help
./run_bev_riskviz.sh cli --help
```
