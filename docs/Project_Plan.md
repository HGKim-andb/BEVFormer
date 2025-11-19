## 요청사항

nuScenes 데이터셋에서 emergence labels를 생성하는 Python 스크립트를 작성해주세요.

### 입력
- nuScenes dataset (v1.0-trainval)
- Dataroot: `/path/to/nuscenes`

### 출력
- `emergence_labels_train.pkl`: Training set labels
- `emergence_labels_val.pkl`: Validation set labels
- `label_statistics.json`: 통계 정보
- `label_config.json`: 설정 저장

### 핵심 로직

#### Step 1: Scene 순회
각 scene의 연속된 sample들을 순서대로 처리

#### Step 2: 각 sample (t)에 대해
1. **과거 객체 수집** (t-5 ~ t-1):
   - 각 프레임의 모든 annotation 가져오기
   - instance_token으로 객체 식별
   - Set에 저장: `past_object_ids`

2. **미래에서 emergence 찾기** (t+1, t+2, t+3):
   - 각 future frame의 annotation 확인
   - 조건 체크:
     * `instance_token NOT IN past_object_ids`
     * `2m < distance < 40m`
     * `category in VALID_CATEGORIES`
   - 조건 만족하면 emergence로 기록

3. **BEV grid로 변환**:
   - World coordinates (x, y) → BEV grid (grid_x, grid_y)
   - BEV 범위: [-50, 50, -50, 50] meters
   - Resolution: 0.5m per pixel → 200x200 grid
   - Gaussian smoothing (sigma=2.0)

#### Step 3: 저장
각 sample마다:
```python
{
    'sample_token': str,
    'scene_token': str,
    'scene_name': str,
    'emergence_mask': np.array [3, 200, 200],  # float32, 0~1
    'emergence_class': np.array [3, 200, 200], # int32, class indices
    'num_emergences': int,
    'emergence_info': [
        {
            'frame': int (1, 2, or 3),
            'position': (x, y),
            'grid_pos': (gx, gy),
            'category': str,
            'distance': float
        },
        ...
    ]
}
```

### 설정 (Config)
```python
CONFIG = {
    'lookback_frames': 5,
    'lookahead_frames': 3,
    'bev_range': [-50, 50, -50, 50],
    'bev_resolution': 0.5,
    'min_distance': 2.0,
    'max_distance': 40.0,
    'gaussian_sigma': 2.0,
    'valid_categories': [
        'vehicle.car',
        'vehicle.truck',
        'vehicle.bus',
        'vehicle.bicycle',
        'vehicle.motorcycle',
        'human.pedestrian.adult',
        'human.pedestrian.child',
    ]
}
```

### Category Mapping
```python
CATEGORY_MAP = {
    'vehicle': 1,
    'pedestrian': 2,
    'bicycle': 3,
    'motorcycle': 4,
}

def simplify_category(full_category_name):
    if 'vehicle' in full_category_name:
        if 'bicycle' in full_category_name:
            return 'bicycle'
        elif 'motorcycle' in full_category_name:
            return 'motorcycle'
        else:
            return 'vehicle'
    elif 'pedestrian' in full_category_name:
        return 'pedestrian'
    return 'other'
```

### 중요한 함수들

#### 1. World → BEV Grid 변환
```python
def world_to_grid(x, y, bev_range, resolution):
    """
    World coordinates to BEV grid indices
    
    Args:
        x, y: World coordinates (meters)
        bev_range: [x_min, x_max, y_min, y_max]
        resolution: meters per pixel
    
    Returns:
        grid_x, grid_y: Grid indices
    """
    x_min, x_max, y_min, y_max = bev_range
    grid_x = int((x - x_min) / resolution)
    grid_y = int((y - y_min) / resolution)
    return grid_x, grid_y
```

#### 2. Gaussian Smoothing
```python
def add_gaussian_to_grid(grid, center, sigma):
    """
    Add Gaussian blob at center position
    
    Args:
        grid: np.array [H, W]
        center: (y, x) in grid coordinates
        sigma: Gaussian width
    """
    y, x = center
    kernel_size = int(sigma * 3)
    
    # Create meshgrid
    y_range = np.arange(max(0, y-kernel_size), 
                       min(grid.shape[0], y+kernel_size+1))
    x_range = np.arange(max(0, x-kernel_size),
                       min(grid.shape[1], x+kernel_size+1))
    
    if len(y_range) == 0 or len(x_range) == 0:
        return
    
    yy, xx = np.meshgrid(y_range, x_range, indexing='ij')
    
    # Gaussian
    gaussian = np.exp(-((yy-y)**2 + (xx-x)**2) / (2*sigma**2))
    
    # Add (max operation for overlaps)
    grid[y_range[0]:y_range[-1]+1, 
         x_range[0]:x_range[-1]+1] = np.maximum(
        grid[y_range[0]:y_range[-1]+1, x_range[0]:x_range[-1]+1],
        gaussian
    )
```

#### 3. Scene의 모든 Sample 가져오기
```python
def get_scene_samples(nusc, scene_token):
    """
    Get all samples in a scene in order
    
    Returns:
        List of sample tokens
    """
    scene = nusc.get('scene', scene_token)
    samples = []
    
    sample_token = scene['first_sample_token']
    while sample_token:
        samples.append(sample_token)
        sample = nusc.get('sample', sample_token)
        sample_token = sample['next']
    
    return samples
```

### 파일 구조
```
tools/
├── create_emergence_labels.py          # Main script
├── emergence_utils.py                  # Utility functions
└── configs/
    └── emergence_config.py             # Config

출력:
data/
├── emergence_labels_train.pkl
├── emergence_labels_val.pkl
├── label_statistics.json
└── label_config.json
```

### 실행 방법
```bash
python tools/create_emergence_labels.py \
    --dataroot /path/to/nuscenes \
    --version v1.0-trainval \
    --output_dir data/emergence_labels
```

### 필요한 라이브러리
```python
import numpy as np
import pickle
import json
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from scipy.ndimage import gaussian_filter
import argparse
from pathlib import Path
```

### 진행상황 표시
```python
# Scene 단위 진행
for scene in tqdm(nusc.scene, desc="Processing scenes"):
    ...

# 중간중간 로그
print(f"Scene {scene['name']}: Found {num_emergences} emergences")
```

### 에러 처리
```python
try:
    # Process sample
    ...
except Exception as e:
    print(f"Error processing sample {sample_token}: {e}")
    continue  # Skip this sample
```

### 최종 통계 출력
```python
"""
================================
Emergence Label Generation Complete
================================
Total scenes: 850
Total samples: 28,130
Samples with emergence: 3,245 (11.5%)
Total emergence events: 5,678
Avg emergences per positive sample: 1.75

Category distribution:
  - Pedestrian: 2,456 (43.3%)
  - Vehicle: 2,134 (37.6%)
  - Bicycle: 892 (15.7%)
  - Motorcycle: 196 (3.4%)

Per-frame distribution:
  - t+1: 3,421 (60.2%)
  - t+2: 1,678 (29.6%)
  - t+3: 579 (10.2%)

Saved to: data/emergence_labels/
"""
```

### 중요 노트
1. **Memory 관리**: Scene별로 처리하고 중간 결과는 메모리에서 제거
2. **Validation**: Train/val split은 nuScenes 공식 split 사용
3. **Reproducibility**: Random seed 고정 불필요 (deterministic)
4. **속도**: 예상 실행 시간 30-60분 (850 scenes)

### 검증 사항
스크립트 완성 후 다음을 확인:
- [ ] Positive ratio가 5-15% 사이인가?
- [ ] t+1에 emergence가 가장 많은가?
- [ ] Distance distribution이 10-30m 범위인가?
- [ ] Category distribution이 reasonable한가?