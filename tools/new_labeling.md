📋 프로젝트 개요
자율주행에서 가려진 영역(occluded regions)의 emergence risk를 예측하기 위한 BEV risk map label을 생성합니다.

🎯 목표
nuScenes 데이터셋의 각 sample에 대해 **BEV Risk Map [200×200]**을 생성합니다.
각 pixel의 값(0.0~1.0)은 "그 위치에서 갑자기 객체가 나타날 위험도"를 나타냅니다.

💡 핵심 개념
Emergence Risk란?
큰 객체에 의해 가려진 영역에서
Ego 차량이 접근할 때
반응 시간이 부족한 상황에서
갑자기 객체가 나타날 위험도
Risk가 높은 경우:

큰 객체(버스, 트럭)에 의해 가려짐 (Occlusion)
Ego가 그 방향으로 진행 중 (Trajectory)
반응 시간이 부족한 거리 (8-20m, TTC 1-2초)
옆 차선에서 교차 (Lateral collision course)


📐 Risk Map 생성 알고리즘
Input (한 sample):
python{
    'ego_state': {
        'position': (x, y),          # ego vehicle 위치
        'velocity': v,               # 속도 (m/s)
        'heading': θ,                # 방향 (radians)
    },
    'detected_objects': [
        {
            'position': (x, y, z),
            'size': (width, length, height),
            'class': 'vehicle.car',
            'velocity': (vx, vy),    # optional
        },
        ...
    ]
}
```

### **Process (각 BEV cell마다):**
```
For cell at (grid_x, grid_y):
    
    1. World 좌표로 변환
       cell_world = grid_to_world(grid_x, grid_y)
    
    2. Feature 계산
       features = compute_features(cell_world, ego_state, objects)
    
    3. Risk score 계산
       risk = compute_risk_score(features)
    
    4. 저장
       risk_map[grid_y, grid_x] = risk
Output:
python{
    'sample_token': str,
    'risk_map': np.array([200, 200], dtype=float32),  # 0.0~1.0
    'ego_state': {...},
    'metadata': {
        'max_risk': float,
        'high_risk_cells': int,  # risk > 0.7
    }
}

🔧 구현 상세
1. Configuration
pythonCONFIG = {
    # BEV Grid
    'bev_range': [-50, 50, -50, 50],  # [x_min, x_max, y_min, y_max] meters
    'bev_resolution': 0.5,             # meters per pixel
    'bev_h': 200,
    'bev_w': 200,
    
    # Distance thresholds (TTC based)
    'far_distance': 30.0,      # > 30m: low urgency
    'medium_distance': 15.0,   # 15-30m: medium urgency
    'close_distance': 8.0,     # 8-15m: high urgency
    # < 8m: critical or safe (depends on lateral offset)
    
    # Lateral (좌우) thresholds
    'same_lane_threshold': 1.5,      # < 1.5m: same lane
    'adjacent_lane_threshold': 5.0,  # 1.5-5m: adjacent lanes
    'lane_width': 3.5,                # standard lane width
    
    # Occlusion
    'min_occluder_area': 1.0,  # m², minimum size to occlude
    
    # Collision course
    'collision_distance_threshold': 2.0,  # meters
    'prediction_horizon': 3.0,            # seconds
    
    # Risk weights
    'base_occlusion_weight': 0.3,
}

2. 핵심 함수들
2.1 Ego State 추출
pythondef get_ego_state(nusc, sample):
    """
    Ego vehicle의 상태 추출
    
    Returns:
        {
            'position': (x, y),
            'velocity': float,  # m/s
            'heading': float,   # radians
            'trajectory': np.array  # predicted path
        }
    """
    구현 내용:
    
    1. Ego pose 가져오기
       - nusc.get('ego_pose', ...)
       - translation[:2] → position
       - rotation → heading
    
    2. Velocity 계산
       - Current와 previous sample 비교
       - delta_position / delta_time
    
    3. Trajectory 예측 (간단하게)
       - 현재 heading 방향으로 직선
       - length = velocity × 3.0 (3초)
       - future_pos = current_pos + velocity * heading_vector * t
2.2 Detected Objects 추출
pythondef get_detected_objects(nusc, sample):
    """
    Sample의 모든 detected objects
    
    Returns:
        List[{
            'position': (x, y, z),
            'size': (w, l, h),
            'class': str,
            'velocity': (vx, vy) or None,
        }]
    """
    구현 내용:
    
    1. sample['anns'] 순회
    2. 각 annotation:
       - translation → position
       - size → (w, l, h)
       - category_name → class
       - nusc.box_velocity() → velocity (optional)
    
    3. Ego frame으로 변환
       - World → Ego coordinate
2.3 Occlusion 체크 (Ray Casting)
pythondef find_occluding_object(cell_pos, ego_state, objects):
    """
    Cell이 어떤 객체에 의해 가려지는지 확인
    
    Args:
        cell_pos: (x, y) in world coordinates
        ego_state: ego vehicle state
        objects: list of detected objects
    
    Returns:
        occluding_object or None
    """
    구현 내용:
    
    1. Ego → Cell 방향 ray 생성
       ray_direction = (cell_pos - ego_pos) / distance
    
    2. 각 object에 대해:
       - Ray와 object bounding box의 교차 확인
       - 간단히: object가 ray 위에 있고
                cell보다 ego에 가까운가?
    
    3. 가장 가까운 occluding object 반환
    
    간단한 방법:
    - Ego-Cell 선분과 object bbox의 2D 교차 체크
    - Object center가 선분 근처 + ego보다 가까움
2.4 Lateral Distance 계산
pythondef compute_lateral_distance(ego_state, obj_position):
    """
    Ego 진행 방향에 수직인 거리
    
    Returns:
        lateral_distance: float (absolute)
        lateral_side: -1 (left) or 1 (right)
    """
    구현 내용:
    
    1. Ego heading 벡터
       heading_vec = [cos(heading), sin(heading)]
    
    2. Ego → Object 벡터
       ego_to_obj = obj_position - ego_position
    
    3. Perpendicular vector (수직)
       perp_vec = [-sin(heading), cos(heading)]
    
    4. Projection (내적)
       lateral_dist = dot(ego_to_obj, perp_vec)
    
    5. Return abs(lateral_dist), sign(lateral_dist)
2.5 Collision Course 체크
pythondef check_collision_course(ego_state, obj_state, horizon=3.0):
    """
    향후 3초 동안 경로가 교차하는지
    
    Returns:
        is_collision: bool
        min_distance: float
    """
    구현 내용:
    
    1. Ego 예상 경로
       ego_future = ego_pos + ego_vel * heading_vec * horizon
       ego_path = (ego_pos, ego_future)
    
    2. Object 예상 경로
       if obj_velocity is not None:
           obj_future = obj_pos + obj_vel * horizon
       else:
           obj_future = obj_pos  # stationary
       obj_path = (obj_pos, obj_future)
    
    3. 두 선분의 최소 거리 계산
       min_dist = minimum_distance_between_segments(ego_path, obj_path)
    
    4. Return min_dist < threshold (2.0m)
2.6 Feature 계산
pythondef compute_cell_features(cell_pos, ego_state, objects):
    """
    한 cell에 대한 모든 features
    
    Returns:
        {
            # Occlusion
            'is_occluded': bool,
            'occluder': object or None,
            'occlusion_strength': float,  # 0~1
            'type_diversity': float,      # 0~1
            
            # Longitudinal (앞뒤)
            'ego_distance': float,
            'time_to_collision': float,
            'longitudinal_urgency': float,  # 0~1
            'lateral_offset': float,  # ego가 occluder 얼마나 통과
            
            # Lateral (좌우)
            'lateral_distance': float,
            'lateral_risk': float,  # 0~1
            'same_lane': bool,
            'is_collision_course': bool,
            
            # Trajectory
            'ego_alignment': float,  # cos(angle), 0~1
            'distance_to_trajectory': float,
            'trajectory_factor': float,  # 0~1
            
            # Others
            'shadow_depth': float,
            'velocity_boost': float,  # 0.7~1.3
        }
    """
    구현 내용:
    
    1. Occlusion features
       - find_occluding_object()
       - if occluder:
           - area = occluder['size'][0] * occluder['size'][1]
           - occlusion_strength = compute_occlusion_strength(occluder)
           - type_diversity = compute_type_diversity(occluder)
    
    2. Longitudinal features
       - ego_distance = distance(ego_pos, cell_pos)
       - ttc = ego_distance / ego_velocity
       - urgency = compute_longitudinal_urgency(ttc, ego_distance)
       - if close (<8m):
           - lateral_offset = compute_lateral_offset(ego, occluder)
    
    3. Lateral features
       - lateral_dist = compute_lateral_distance(ego, occluder)
       - lateral_risk = compute_lateral_risk(lateral_dist, ...)
       - same_lane = (lateral_dist < 1.5)
       - is_collision_course = check_collision_course(...)
    
    4. Trajectory features
       - ego_to_cell = cell - ego_pos
       - cell_direction = atan2(ego_to_cell)
       - alignment = cos(cell_direction - ego_heading)
       - dist_to_traj = distance(cell, ego_trajectory)
       - trajectory_factor = exp(-dist_to_traj / 3.0)
    
    5. Others
       - shadow_depth = distance(occluder, cell)
       - velocity_boost = compute_velocity_boost(ego_velocity)
2.7 Occlusion Strength 계산
pythondef compute_occlusion_strength(occluder):
    """
    객체 크기 기반 occlusion 능력
    
    Returns:
        strength: float (0~1)
    """
    구현 내용:
    
    객체 타입별 기본값:
    OCCLUSION_PROFILES = {
        'vehicle.bus': 1.0,
        'vehicle.truck': 0.9,
        'vehicle.car': 0.6,
        'vehicle.motorcycle': 0.3,
        'human.pedestrian.adult': 0.3,
        'human.pedestrian.child': 0.1,
        'vehicle.bicycle': 0.2,
    }
    
    1. 타입별 base strength
    2. 실제 크기로 조정
       - area = width * length
       - strength = base * min(area / expected_area, 1.5)
    3. Clip to [0, 1]
2.8 Type Diversity 계산
pythondef compute_type_diversity(occluder):
    """
    가려진 곳에서 나올 수 있는 객체 다양성
    
    Returns:
        diversity: float (0~1)
    """
    구현 내용:
    
    객체 크기 기반:
    
    area = occluder['size'][0] * occluder['size'][1]
    
    if area > 8.0:  # 큰 객체 (버스, 트럭)
        diversity = 1.0
        # 차량, 보행자, 자전거 모두 가능
    
    elif area > 3.0:  # 중간 (승용차)
        diversity = 0.7
        # 보행자, 자전거, 오토바이
    
    elif area > 1.0:  # 작은 객체
        diversity = 0.4
        # 보행자, 자전거만
    
    else:  # 매우 작음
        diversity = 0.1
        # 거의 못 가림
2.9 Longitudinal Urgency
pythondef compute_longitudinal_urgency(ttc, distance, lateral_offset=None):
    """
    거리/TTC 기반 긴급도
    
    Returns:
        urgency: float (0~1)
    """
    구현 내용:
    
    if ttc > 3.0 or distance > 30.0:
        # 멀리, 시간 여유
        return 0.4
    
    elif ttc > 1.5 or distance > 15.0:
        # 중간, 주의
        return 0.7
    
    elif ttc > 0.8 or distance > 8.0:
        # 가까움, 위험
        return 0.9
    
    else:  # < 8m, 매우 가까움
        # lateral_offset으로 판단
        if lateral_offset is None:
            return 1.0
        
        if lateral_offset < 3.0:
            # 아직 안 지나감 → Critical!
            return 1.0
        else:
            # 많이 지나감 → 이미 봄
            return 0.2
2.10 Lateral Risk
pythondef compute_lateral_risk(lateral_dist, relative_velocity, is_collision_course):
    """
    좌우 거리 기반 위험도
    
    Returns:
        risk: float (0~1)
    """
    구현 내용:
    
    if lateral_dist < 1.5:
        # 같은 차선
        if relative_velocity < 2.0:
            # 함께 이동
            risk = 0.3
        else:
            # 속도 차이
            risk = 0.6
    
    elif lateral_dist < 5.0:
        # 옆 차선
        num_lanes = int(lateral_dist / 3.5)
        
        if num_lanes == 1:
            # 바로 옆 차선
            risk = 0.9
            if is_collision_course:
                risk = 1.0  # Boost!
        else:
            # 2개 차선
            risk = 0.5
    
    else:
        # 멀리
        risk = 0.2 * np.exp(-(lateral_dist - 5.0) / 5.0)
    
    return risk
2.11 최종 Risk Score
pythondef compute_risk_score(features):
    """
    모든 features를 조합하여 최종 risk
    
    Returns:
        risk: float (0~1)
    """
    구현 내용:
    
    # Step 1: Base score (occlusion)
    if not features['is_occluded']:
        return 0.0
    
    base = 0.3 * features['occlusion_strength'] * features['type_diversity']
    
    # Step 2: Urgency (거리 기반)
    score = base * features['longitudinal_urgency']
    
    # Step 3: Lateral risk
    score = score * features['lateral_risk']
    
    # Step 4: Trajectory alignment
    score = score * (0.3 + 0.7 * features['ego_alignment'])
    
    # Step 5: Trajectory proximity
    score = score * (0.5 + 0.5 * features['trajectory_factor'])
    
    # Step 6: Shadow depth
    shadow_depth = features['shadow_depth']
    if shadow_depth < 3.0:
        depth_factor = 1.0
    elif shadow_depth < 8.0:
        depth_factor = 0.7
    else:
        depth_factor = 0.4
    score = score * depth_factor
    
    # Step 7: Velocity boost
    score = score * features['velocity_boost']
    
    # Step 8: Collision course boost
    if features['is_collision_course']:
        score = score * 1.5
    
    # Final clip
    return np.clip(score, 0.0, 1.0)

3. Main Pipeline
pythondef generate_risk_map(nusc, sample, config):
    """
    한 sample에 대한 risk map 생성
    
    Returns:
        {
            'sample_token': str,
            'risk_map': np.array([200, 200]),
            'ego_state': dict,
            'metadata': dict,
        }
    """
    구현 내용:
    
    # 1. Extract state
    ego_state = get_ego_state(nusc, sample)
    objects = get_detected_objects(nusc, sample)
    
    # 2. Initialize risk map
    risk_map = np.zeros((config['bev_h'], config['bev_w']), dtype=np.float32)
    
    # 3. For each cell
    for grid_y in range(config['bev_h']):
        for grid_x in range(config['bev_w']):
            # World coordinates
            cell_world = grid_to_world(grid_x, grid_y, config)
            
            # Features
            features = compute_cell_features(cell_world, ego_state, objects)
            
            # Risk
            risk = compute_risk_score(features)
            
            risk_map[grid_y, grid_x] = risk
    
    # 4. Metadata
    metadata = {
        'max_risk': risk_map.max(),
        'mean_risk': risk_map.mean(),
        'high_risk_cells': (risk_map > 0.7).sum(),
        'medium_risk_cells': ((risk_map > 0.3) & (risk_map <= 0.7)).sum(),
    }
    
    # 5. Return
    return {
        'sample_token': sample['token'],
        'scene_token': sample['scene_token'],
        'risk_map': risk_map,
        'ego_state': ego_state,
        'metadata': metadata,
    }

4. 전체 데이터셋 처리
pythondef create_all_risk_labels(nusc, config, output_path):
    """
    전체 nuScenes 데이터셋 처리
    """
    구현 내용:
    
    all_labels = {}
    
    # Train/val split
    train_scenes = [s for s in nusc.scene if s['name'] in nusc.scene_split['train']]
    val_scenes = [s for s in nusc.scene if s['name'] in nusc.scene_split['val']]
    
    # Process train
    print("Processing train set...")
    train_labels = process_scenes(nusc, train_scenes, config)
    
    # Process val
    print("Processing val set...")
    val_labels = process_scenes(nusc, val_scenes, config)
    
    # Save
    with open(f'{output_path}/risk_labels_train.pkl', 'wb') as f:
        pickle.dump(train_labels, f)
    
    with open(f'{output_path}/risk_labels_val.pkl', 'wb') as f:
        pickle.dump(val_labels, f)
    
    # Statistics
    print_statistics(train_labels, val_labels)


def process_scenes(nusc, scenes, config):
    """
    여러 scene 처리
    """
    all_labels = {}
    
    for scene in tqdm(scenes, desc="Processing scenes"):
        scene_labels = []
        
        # Get all samples
        sample_tokens = get_scene_samples(nusc, scene['token'])
        
        for sample_token in sample_tokens:
            sample = nusc.get('sample', sample_token)
            
            try:
                label = generate_risk_map(nusc, sample, config)
                scene_labels.append(label)
            except Exception as e:
                print(f"Error on {sample_token}: {e}")
                continue
        
        all_labels[scene['token']] = scene_labels
    
    return all_labels

📊 통계 분석 & 시각화
통계 스크립트
pythondef analyze_risk_labels(labels_path):
    """
    생성된 risk labels 분석
    """
    
    분석 항목:
    
    1. Risk distribution
       - Mean, median, std of risk values
       - Histogram of risk scores
    
    2. High risk cells
       - % of cells with risk > 0.7
       - % of cells with risk > 0.5
       - % of cells with risk > 0.3
    
    3. Spatial distribution
       - Heatmap of average risk across BEV
       - Where are high-risk areas typically?
    
    4. Per-scene statistics
       - Scenes with highest avg risk
       - Scenes with most high-risk cells
    
    Output:
    - risk_statistics.json
    - risk_distribution.png (plots)
시각화 스크립트
pythondef visualize_risk_samples(nusc, labels, num_samples=20):
    """
    Risk map 시각화
    """
    
    각 sample마다:
    
    ┌─────────────────────────────────────┐
    │ Front Camera Image                  │
    ├─────────────────────────────────────┤
    │ BEV with Detected Objects           │
    │ + Ego vehicle + orientation arrow   │
    ├─────────────────────────────────────┤
    │ Risk Map Heatmap (0~1)              │
    │ + High risk regions highlighted     │
    └─────────────────────────────────────┘
    
    Color scheme:
    - Blue: Low risk (0.0-0.3)
    - Yellow: Medium (0.3-0.7)
    - Red: High (0.7-1.0)
```

---

## 📝 **파일 구조**
```
tools/
├── create_risk_labels.py          # Main script
├── risk_utils.py                  # Utility functions
│   ├── get_ego_state()
│   ├── get_detected_objects()
│   ├── find_occluding_object()
│   ├── compute_cell_features()
│   ├── compute_risk_score()
│   └── ...
├── analyze_risk_labels.py         # Analysis script
└── visualize_risk_samples.py      # Visualization

configs/
└── risk_config.py                 # Configuration

출력:
data/emergence_risk/
├── risk_labels_train.pkl
├── risk_labels_val.pkl
├── risk_statistics.json
└── risk_config.json

🚀 실행 방법
bash# 1. Risk labels 생성
python tools/create_risk_labels.py \
    --dataroot /path/to/nuscenes \
    --version v1.0-trainval \
    --output_dir data/emergence_risk

# 2. 통계 분석
python tools/analyze_risk_labels.py \
    --labels data/emergence_risk/risk_labels_train.pkl \
    --output_dir analysis/

# 3. 시각화
python tools/visualize_risk_samples.py \
    --labels data/emergence_risk/risk_labels_train.pkl \
    --num_samples 20 \
    --output_dir visualizations/risk_maps
```

---

## ✅ **검증 사항**

생성 후 확인:
```
1. Risk 범위
   - 모든 값이 [0, 1] 범위인가?
   - NaN이나 Inf 없는가?

2. Risk 분포
   - High risk cells: 5-15%?
   - Mean risk: 0.1-0.3?
   - 너무 uniform하거나 extreme하지 않은가?

3. 시각적 검증
   - 큰 객체 뒤에 high risk?
   - Ego 진행 방향에 집중?
   - 멀리 떨어진 곳은 low risk?

4. Edge cases
   - Ego 정지 중: risk가 낮은가?
   - 빈 scene: risk가 0인가?
   - 많은 객체: reasonable한가?
```

---

## 💡 **중요 노트**

### **최적화 팁:**
```
1. 모든 cell을 다 계산하지 말고:
   - Ego 근처만 (예: 40m 이내)
   - Coarse-to-fine: 먼저 큰 grid로

2. Occlusion check 최적화:
   - KD-tree로 가까운 객체만
   - Bounding box로 빠른 reject

3. Parallel processing:
   - Scene별로 병렬 처리
   - Multiprocessing 사용
```

### **예상 실행 시간:**
```
- Single sample: ~0.5-1초
- Full train set (28k): ~8-14시간
- 병렬화 (4 cores): ~2-4시간

🎯 최종 Output 예시
python# risk_labels_train.pkl 내용
{
    'scene-0001': [
        {
            'sample_token': 'xxx',
            'risk_map': np.array([200, 200]),  # 0~1
            'ego_state': {
                'position': (10.5, -2.3),
                'velocity': 8.5,
                'heading': 1.57,
            },
            'metadata': {
                'max_risk': 0.85,
                'mean_risk': 0.12,
                'high_risk_cells': 145,
            }
        },
        ...
    ],
    'scene-0002': [...],
    ...
}