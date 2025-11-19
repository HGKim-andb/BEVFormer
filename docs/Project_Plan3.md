## 요청사항

Emergence labels를 시각적으로 확인하는 스크립트 작성

### 기능
- Random으로 emergence 있는 샘플 선택 (N개)
- 각 샘플마다 visualization 생성

### Visualization 구성 (1 sample = 1 image)
```
┌─────────────────────────────────────────────┐
│  Front Camera Image                         │
├─────────────────────────────────────────────┤
│  BEV with Current Detections                │
│  (ego vehicle + detected objects)           │
├──────────────┬──────────────┬───────────────┤
│ Emergence    │ Emergence    │ Emergence     │
│ t+1          │ t+2          │ t+3           │
│ (heatmap)    │ (heatmap)    │ (heatmap)     │
└──────────────┴──────────────┴───────────────┘
```

### 구현 디테일
```python
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def visualize_sample(nusc, sample_token, label_data, save_path):
    """
    Create visualization for one sample
    """
    sample = nusc.get('sample', sample_token)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Front camera (top)
    ax1 = plt.subplot(3, 1, 1)
    cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    img = plt.imread(f"data/nuscenes/{cam_data['filename']}")
    ax1.imshow(img)
    ax1.set_title(f'Sample: {sample_token[:12]}...', fontsize=14)
    ax1.axis('off')
    
    # 2. BEV with detections (middle)
    ax2 = plt.subplot(3, 3, 4)
    ax2.set_xlim(-50, 50)
    ax2.set_ylim(-50, 50)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Current BEV', fontsize=12)
    
    # Plot ego
    ego_rect = Rectangle((-2, -1), 4, 2, linewidth=2, 
                         edgecolor='blue', facecolor='blue', alpha=0.3)
    ax2.add_patch(ego_rect)
    
    # Plot detections
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        x, y = ann['translation'][:2]
        ax2.plot(x, y, 'go', markersize=8)
    
    # Plot future emergences
    for info in label_data['emergence_info']:
        x, y = info['position']
        frame = info['frame']
        color = ['red', 'orange', 'yellow'][frame-1]
        ax2.plot(x, y, '*', color=color, markersize=15)
        ax2.text(x, y-2, f"t+{frame}", fontsize=8, color=color)
    
    # 3-5. Emergence heatmaps (bottom)
    for t in range(3):
        ax = plt.subplot(3, 3, 5+t)
        mask = label_data['emergence_mask'][t]
        
        im = ax.imshow(mask, cmap='hot', vmin=0, vmax=1, 
                      extent=[-50, 50, -50, 50])
        ax.set_title(f'Emergence t+{t+1}', fontsize=12)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        # Mark actual emergences
        for info in label_data['emergence_info']:
            if info['frame'] == t+1:
                x, y = info['position']
                ax.plot(x, y, 'g*', markersize=12)
        
        plt.colorbar(im, ax=ax)
    
    # Info text
    fig.text(0.5, 0.02, 
             f"Emergences: {label_data['num_emergences']} | "
             f"Scene: {label_data['scene_name']}", 
             ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# Main
def main():
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes')
    
    with open('data/emergence_labels_train.pkl', 'rb') as f:
        all_labels = pickle.load(f)
    
    # Get positive samples
    positive_samples = []
    for scene_labels in all_labels.values():
        for label in scene_labels:
            if label['num_emergences'] > 0:
                positive_samples.append(label)
    
    # Random sample
    import random
    random.seed(42)
    selected = random.sample(positive_samples, min(20, len(positive_samples)))
    
    # Generate visualizations
    output_dir = Path('visualizations/emergence_samples')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, label in enumerate(tqdm(selected, desc="Visualizing")):
        save_path = output_dir / f"sample_{i:03d}_{label['sample_token'][:8]}.png"
        visualize_sample(nusc, label['sample_token'], label, save_path)
    
    print(f"✅ Generated {len(selected)} visualizations in {output_dir}")
```

### 실행
```bash
python tools/visualize_emergence_samples.py \
    --labels data/emergence_labels_train.pkl \
    --num_samples 20 \
    --output_dir visualizations/
```