# Full Dataset Risk Labels Split 가이드

## 상황
다른 컴퓨터에 Full dataset (v1.0-trainval)의 risk labels pkl 파일이 있음

## Split 방법

### 준비물
1. Risk labels pkl 파일 (예: `risk_labels_all.pkl`)
2. nuScenes full dataset 접근 (또는 최소한 scene 정보)

---

## 방법 1: split_risk_labels_official.py 사용 (권장)

### Step 1: pkl 파일 복사
```bash
# 다른 컴퓨터에서 이 컴퓨터로 복사
scp other_computer:/path/to/risk_labels_all.pkl data/emergence_risk_v5_full/
```

### Step 2: Split 실행
```bash
python tools/split_risk_labels_official.py \
    --input data/emergence_risk_v5_full/risk_labels_all.pkl \
    --dataroot /path/to/nuscenes \
    --version v1.0-trainval \
    --output_dir data/emergence_risk_v5_full
```

**출력**:
- `data/emergence_risk_v5_full/risk_labels_train.pkl` (700 scenes)
- `data/emergence_risk_v5_full/risk_labels_val.pkl` (150 scenes)

---

## 방법 2: nuScenes 없이 Split (Scene 정보만 있으면 가능)

nuScenes dataset에 접근할 수 없는 경우:

```python
#!/usr/bin/env python3
"""
Split risk labels without nuScenes dataset
Only needs scene name to split mapping
"""
import pickle
from pathlib import Path

# nuScenes official train split scene names (처음 700개)
# 전체 리스트는 nuscenes.utils.splits.create_splits_scenes() 참고
TRAIN_SCENES = [
    'scene-0001', 'scene-0002', 'scene-0004', 'scene-0005', 'scene-0006',
    # ... 700 scenes total
    # Full list: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py
]

VAL_SCENES = [
    'scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015',
    # ... 150 scenes total
]

def split_without_nuscenes(input_pkl, output_dir):
    """Split risk labels using hardcoded scene lists"""
    
    # Load all labels
    with open(input_pkl, 'rb') as f:
        all_labels = pickle.load(f)
    
    train_labels = {}
    val_labels = {}
    
    # Get scene names from labels
    # Assuming label structure: {scene_token: [labels...], ...}
    # and each label has 'scene_name' field
    
    for scene_token, labels in all_labels.items():
        if len(labels) > 0:
            scene_name = labels[0].get('scene_name', 'unknown')
            
            if scene_name in TRAIN_SCENES:
                train_labels[scene_token] = labels
            elif scene_name in VAL_SCENES:
                val_labels[scene_token] = labels
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'risk_labels_train.pkl', 'wb') as f:
        pickle.dump(train_labels, f)
    
    with open(output_dir / 'risk_labels_val.pkl', 'wb') as f:
        pickle.dump(val_labels, f)
    
    print(f"Train: {len(train_labels)} scenes")
    print(f"Val: {len(val_labels)} scenes")

if __name__ == '__main__':
    split_without_nuscenes(
        'data/emergence_risk_v5_full/risk_labels_all.pkl',
        'data/emergence_risk_v5_full'
    )
```

---

## 방법 3: 네트워크를 통해 직접 Split

다른 컴퓨터에서 직접 split 후 전송:

### 다른 컴퓨터에서:
```bash
# 1. BEVFormer 코드 복사 (split 스크립트만)
scp -r tools/split_risk_labels_official.py other_computer:/path/to/

# 2. 다른 컴퓨터에서 split 실행
ssh other_computer
cd /path/to/BEVFormer
python tools/split_risk_labels_official.py \
    --input /path/to/risk_labels_all.pkl \
    --dataroot /path/to/nuscenes \
    --version v1.0-trainval \
    --output_dir /path/to/output

# 3. Split된 파일만 복사
exit
scp other_computer:/path/to/output/risk_labels_train.pkl data/emergence_risk_v5_full/
scp other_computer:/path/to/output/risk_labels_val.pkl data/emergence_risk_v5_full/
```

---

## 방법 4: pkl 파일 구조 확인 후 수동 Split

### Step 1: pkl 구조 확인
```python
import pickle

with open('risk_labels_all.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Type: {type(data)}")
print(f"Keys (first 5): {list(data.keys())[:5]}")

# 첫 번째 scene 확인
first_scene_token = list(data.keys())[0]
first_scene_data = data[first_scene_token]
print(f"\nFirst scene:")
print(f"  Token: {first_scene_token}")
print(f"  Samples: {len(first_scene_data)}")
if len(first_scene_data) > 0:
    print(f"  Sample keys: {first_scene_data[0].keys()}")
    print(f"  Scene name: {first_scene_data[0].get('scene_name', 'N/A')}")
```

### Step 2: Scene name 확인 후 Split
위 정보를 바탕으로 적절한 split 스크립트 작성

---

## 간단한 Split 스크립트 (nuScenes 없이)

```python
#!/usr/bin/env python3
import pickle
from pathlib import Path
from nuscenes.utils.splits import create_splits_scenes

def split_risk_labels_simple(input_pkl, output_dir):
    """
    Simple split without loading full nuScenes
    Uses nuscenes-devkit's split info only
    """
    # Get official splits (doesn't need dataset, just returns scene names)
    splits = create_splits_scenes()
    train_scene_names = set(splits['train'])
    val_scene_names = set(splits['val'])
    
    # Load labels
    print(f"Loading {input_pkl}...")
    with open(input_pkl, 'rb') as f:
        all_labels = pickle.load(f)
    
    print(f"Total scenes in pkl: {len(all_labels)}")
    
    # Split
    train_labels = {}
    val_labels = {}
    unknown = {}
    
    for scene_token, labels in all_labels.items():
        # Get scene name from first label
        if len(labels) > 0 and 'scene_name' in labels[0]:
            scene_name = labels[0]['scene_name']
            
            if scene_name in train_scene_names:
                train_labels[scene_token] = labels
            elif scene_name in val_scene_names:
                val_labels[scene_token] = labels
            else:
                unknown[scene_token] = labels
                print(f"Warning: Unknown scene {scene_name}")
    
    print(f"\nSplit results:")
    print(f"  Train: {len(train_labels)} scenes")
    print(f"  Val:   {len(val_labels)} scenes")
    print(f"  Unknown: {len(unknown)} scenes")
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / 'risk_labels_train.pkl'
    val_path = output_dir / 'risk_labels_val.pkl'
    
    with open(train_path, 'wb') as f:
        pickle.dump(train_labels, f)
    print(f"\n✓ Saved: {train_path}")
    
    with open(val_path, 'wb') as f:
        pickle.dump(val_labels, f)
    print(f"✓ Saved: {val_path}")
    
    return train_labels, val_labels

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    
    split_risk_labels_simple(args.input, args.output_dir)
```

**사용**:
```bash
python split_simple.py \
    --input risk_labels_all.pkl \
    --output_dir data/emergence_risk_v5_full
```

---

## 예상 결과

Full dataset (v1.0-trainval) split 후:

```
data/emergence_risk_v5_full/
├── risk_labels_train.pkl  (~5-6GB, 700 scenes, ~28k samples)
└── risk_labels_val.pkl    (~1-1.5GB, 150 scenes, ~6k samples)
```

---

## Config 업데이트

Split 후 config 파일 수정:

```python
# projects/configs/bevformer/bevformer_risk_base.py
data = dict(
    train=dict(
        type='NuScenesRiskDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_infos_temporal_train.pkl',
        risk_labels_path='data/emergence_risk_v5_full/risk_labels_train.pkl',
        ...
    ),
    val=dict(
        type='NuScenesRiskDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_infos_temporal_val.pkl',
        risk_labels_path='data/emergence_risk_v5_full/risk_labels_val.pkl',
        ...
    ),
)
```

---

## Troubleshooting

### Issue: "scene_name not in labels"
**원인**: Risk label 생성 시 scene_name을 저장하지 않음  
**해결**: Scene token으로 nuScenes dataset에서 scene name 찾기

### Issue: "nuscenes-devkit not installed"
**해결**: 
```bash
pip install nuscenes-devkit
```

### Issue: "Memory error loading large pkl"
**해결**: Chunk 단위로 처리
```python
import pickle

# 메모리 효율적인 로딩
with open('large.pkl', 'rb') as f:
    while True:
        try:
            chunk = pickle.load(f)
            process(chunk)
        except EOFError:
            break
```

---

## 요약

**가장 쉬운 방법**:
1. `split_risk_labels_official.py`를 다른 컴퓨터로 복사
2. 다른 컴퓨터에서 split 실행
3. 결과 파일만 복사해오기

**nuScenes 없이**:
- `nuscenes.utils.splits.create_splits_scenes()` 만 사용
- Scene name 기반으로 split
- Full dataset 없어도 가능!
