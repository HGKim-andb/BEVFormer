# nuScenes Mini Dataset 동작 원리

## 핵심 답변

**✅ Mini dataset을 별도로 다운로드할 필요 없습니다!**
**✅ Full dataset이 있으면 자동으로 mini scene만 선택해서 사용합니다!**

---

## 동작 원리

### 1. nuScenes 데이터 구조

nuScenes는 **하나의 원본 데이터셋**에 **여러 버전의 메타데이터**가 포함되어 있습니다:

```
data/nuscenes/
├── maps/              # 지도 데이터 (공통)
├── samples/           # 이미지 파일들 (공통)
├── sweeps/            # 추가 프레임들 (공통)
├── v1.0-trainval/     # Full dataset 메타데이터
│   ├── attribute.json
│   ├── calibrated_sensor.json
│   ├── category.json
│   ├── ego_pose.json
│   ├── instance.json
│   ├── log.json
│   ├── map.json
│   ├── sample.json
│   ├── sample_annotation.json
│   ├── sample_data.json
│   ├── scene.json
│   ├── sensor.json
│   └── vehicle.json
└── v1.0-mini/         # Mini dataset 메타데이터 (별도 폴더)
    ├── attribute.json
    ├── ... (같은 구조)
```

**중요**: 
- **원본 이미지 파일**(`samples/`, `sweeps/`)은 **공통**으로 사용
- **메타데이터만** 버전별로 다름 (`v1.0-trainval/`, `v1.0-mini/`)

### 2. Scene 분할 방식

코드를 보면:

```python
# tools/data_converter/nuscenes_converter.py:62-64
elif version == 'v1.0-mini':
    train_scenes = splits.mini_train  # 8개 scene
    val_scenes = splits.mini_val       # 2개 scene
```

**nuScenes splits 정의**:
- `splits.train`: 700개 scene (Full train)
- `splits.val`: 150개 scene (Full val)
- `splits.mini_train`: **8개 scene** (Mini train) - Full의 일부
- `splits.mini_val`: **2개 scene** (Mini val) - Full의 일부

### 3. 실제 동작 과정

#### Step 1: NuScenes 클래스 초기화
```python
# nuscenes_converter.py:51
nusc = NuScenes(version='v1.0-mini', dataroot=root_path, verbose=True)
```

이때:
- `v1.0-mini/` 폴더의 메타데이터를 읽음
- **같은 원본 이미지 파일**(`samples/`, `sweeps/`)을 참조
- Mini scene만 필터링 (8개 train + 2개 val)

#### Step 2: Scene 필터링
```python
# nuscenes_converter.py:69-81
available_scenes = get_available_scenes(nusc)
train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
```

- `splits.mini_train`에 정의된 8개 scene만 선택
- `splits.mini_val`에 정의된 2개 scene만 선택

#### Step 3: Info 파일 생성
```python
# nuscenes_converter.py:90-110
train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
    nusc, nusc_can_bus, train_scenes, val_scenes, ...)

# Mini scene만 포함된 info 파일 생성
data = dict(infos=train_nusc_infos, metadata=metadata)
info_path = 'nuscenes_infos_temporal_train.pkl'  # Mini train만 포함
```

---

## 실제 사용 방법

### 시나리오 1: Full dataset만 다운로드한 경우

```bash
# Full dataset 다운로드 (약 300GB)
# v1.0-trainval/ 메타데이터 포함

# Mini dataset 사용하려면?
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --version v1.0-mini \  # 이렇게 지정하면
    ...
```

**동작**:
1. `v1.0-mini/` 폴더가 없으면 에러 발생
2. **하지만** nuScenes devkit이 자동으로 mini 메타데이터를 생성하거나
3. 또는 Full dataset에 이미 `v1.0-mini/` 폴더가 포함되어 있을 수 있음

### 시나리오 2: Mini dataset만 다운로드한 경우

```bash
# Mini dataset 다운로드 (약 3.8GB)
# v1.0-mini/ 메타데이터만 포함

# Full dataset 사용 불가!
python tools/create_data.py nuscenes \
    --version v1.0-trainval \  # ❌ 에러: v1.0-trainval/ 폴더 없음
    ...
```

---

## 정확한 답변

### Q: Mini dataset을 별도로 다운로드해야 하나요?

**A: 상황에 따라 다릅니다:**

1. **Full dataset만 다운로드한 경우**:
   - ✅ Mini dataset도 사용 가능 (같은 원본 데이터 사용)
   - 단, `v1.0-mini/` 메타데이터 폴더가 있어야 함
   - Full dataset에 포함되어 있을 수 있음

2. **Mini dataset만 다운로드한 경우**:
   - ✅ Mini dataset만 사용 가능
   - ❌ Full dataset 사용 불가 (메타데이터 없음)

3. **권장 방법**:
   - Full dataset을 다운로드하면 모든 버전 사용 가능
   - Mini dataset은 Full dataset의 **일부 scene만 선택**하여 사용

### Q: Mini dataset을 어떻게 만드나요?

**A: Full dataset에서 자동으로 선택합니다:**

```python
# 코드에서 자동으로 처리
if version == 'v1.0-mini':
    train_scenes = splits.mini_train  # 8개 scene만 선택
    val_scenes = splits.mini_val      # 2개 scene만 선택
```

**과정**:
1. Full dataset의 모든 scene 중에서
2. `splits.mini_train`에 정의된 8개 scene만 선택
3. `splits.mini_val`에 정의된 2개 scene만 선택
4. 선택된 scene의 정보만 info 파일에 저장

**즉, 수동으로 빼낼 필요 없습니다!** 코드가 자동으로 처리합니다.

---

## 실제 예시

### Full dataset 구조
```
data/nuscenes/
├── samples/           # 모든 scene의 이미지 (공통)
├── sweeps/            # 모든 scene의 추가 프레임 (공통)
├── v1.0-trainval/     # 700 train + 150 val scene 메타데이터
└── v1.0-mini/         # 8 train + 2 val scene 메타데이터
```

### Mini dataset 사용 시
```python
nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes')
# → v1.0-mini/ 메타데이터만 읽음
# → 8개 train scene + 2개 val scene만 사용
# → 같은 samples/, sweeps/ 폴더의 이미지 참조
```

### Info 파일 생성
```python
# Mini scene만 포함된 info 파일 생성
nuscenes_infos_temporal_train.pkl  # 8개 scene의 샘플만 포함 (~323개)
nuscenes_infos_temporal_val.pkl    # 2개 scene의 샘플만 포함 (~81개)
```

---

## 요약

| 질문 | 답변 |
|------|------|
| Mini dataset 별도 다운로드 필요? | ❌ 필요 없음 (Full에 포함) |
| Full에서 Mini 만들기? | ✅ 자동으로 선택 (코드가 처리) |
| 원본 이미지 파일? | ✅ 공통 사용 (samples/, sweeps/) |
| 메타데이터? | ⚠️ 버전별로 다름 (v1.0-trainval/, v1.0-mini/) |
| Info 파일? | ✅ 선택된 scene만 포함 |

**결론**: Full dataset이 있으면 `--version v1.0-mini`만 지정하면 자동으로 mini scene만 사용합니다!

