## 요청사항

생성된 emergence labels의 품질을 분석하는 스크립트를 작성해주세요.

### 입력
- `emergence_labels_train.pkl`
- `emergence_labels_val.pkl`

### 출력
1. **Console 출력**: 통계 리포트
2. **label_statistics.json**: 수치 데이터
3. **distribution_plots.png**: 시각화 (4 subplots)

### 분석 항목

#### 1. 전체 통계
```python
{
    'total_scenes': int,
    'total_samples': int,
    'samples_with_emergence': int,
    'positive_ratio': float,
    'total_emergences': int,
    'avg_per_positive_sample': float
}
```

#### 2. Per-frame 분포
```python
{
    'frame_distribution': {
        't+1': count,
        't+2': count,
        't+3': count
    }
}
```

#### 3. Category 분포
```python
{
    'category_distribution': {
        'pedestrian': count,
        'vehicle': count,
        'bicycle': count,
        'motorcycle': count
    }
}
```

#### 4. Distance 분석
```python
{
    'distance_stats': {
        'mean': float,
        'median': float,
        'min': float,
        'max': float,
        'std': float
    }
}
```

#### 5. Spatial 분석
```python
# Heatmap [200, 200] - 어디서 emergence가 많이 발생하는가
spatial_heatmap = np.zeros((200, 200))
for each emergence:
    spatial_heatmap[grid_y, grid_x] += 1
```

### 시각화 (2x2 subplot)
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Frame distribution (Bar chart)
ax[0, 0].bar(['t+1', 't+2', 't+3'], frame_counts)
ax[0, 0].set_title('Emergence by Future Frame')

# 2. Category distribution (Pie chart)
ax[0, 1].pie(category_counts, labels=categories, autopct='%1.1f%%')
ax[0, 1].set_title('Emergence by Category')

# 3. Distance distribution (Histogram)
ax[1, 0].hist(distances, bins=30)
ax[1, 0].axvline(mean_distance, color='red', linestyle='--')
ax[1, 0].set_title('Distance Distribution')
ax[1, 0].set_xlabel('Distance (m)')

# 4. Spatial heatmap
im = ax[1, 1].imshow(spatial_heatmap, cmap='hot')
ax[1, 1].plot(100, 100, 'b*', markersize=20)  # Ego position
ax[1, 1].set_title('Spatial Distribution')
plt.colorbar(im, ax=ax[1, 1])

plt.tight_layout()
plt.savefig('distribution_plots.png', dpi=150)
```

### 실행
```bash
python tools/analyze_emergence_labels.py \
    --train_labels data/emergence_labels_train.pkl \
    --val_labels data/emergence_labels_val.pkl \
    --output_dir analysis/
```

### 판단 기준 (자동 체크)
```python
def validate_statistics(stats):
    issues = []
    
    if not (0.05 <= stats['positive_ratio'] <= 0.20):
        issues.append(f"⚠️  Positive ratio {stats['positive_ratio']:.1%} outside 5-20%")
    
    if stats['frame_distribution']['t+1'] < stats['frame_distribution']['t+2']:
        issues.append("⚠️  t+1 should have most emergences")
    
    if stats['distance_stats']['mean'] < 5 or stats['distance_stats']['mean'] > 35:
        issues.append(f"⚠️  Mean distance {stats['distance_stats']['mean']:.1f}m unusual")
    
    if len(issues) == 0:
        print("✅ All statistics look reasonable!")
    else:
        print("Issues found:")
        for issue in issues:
            print(issue)
```