# Emergence Labels - Quick Start Guide

## 빠른 실행 (3단계)

### 1단계: Mini Dataset 테스트 (필수)
```bash
python tools/create_emergence_labels.py \
    --dataroot data/nuscenes \
    --version v1.0-mini \
    --output_dir data/emergence_labels_test
```
**소요 시간**: ~5초  
**예상 결과**: Train 603개, Val 451개 emergences

### 2단계: 분석 및 시각화
```bash
# 통계 분석
python tools/analyze_emergence_labels.py \
    --train_labels data/emergence_labels_test/emergence_labels_train.pkl \
    --val_labels data/emergence_labels_test/emergence_labels_val.pkl \
    --output_dir data/emergence_labels_test

# 시각화 (선택)
python tools/visualize_emergence_samples.py \
    --labels data/emergence_labels_test/emergence_labels_train.pkl \
    --dataroot data/nuscenes \
    --num_samples 10 \
    --output_dir visualizations/emergence_samples_mini
```

### 3단계: Full Dataset 실행
```bash
python tools/create_emergence_labels.py \
    --dataroot data/nuscenes \
    --version v1.0-trainval \
    --output_dir data/emergence_labels
```
**소요 시간**: 30-60분  
**출력**: emergence_labels_train.pkl, emergence_labels_val.pkl

---

## 생성된 파일

### 스크립트
- `tools/create_emergence_labels.py` - 메인 생성 스크립트
- `tools/analyze_emergence_labels.py` - 통계 분석
- `tools/visualize_emergence_samples.py` - 시각화

### 문서
- `tools/emergence_labels_README.md` - 전체 문서
- `tools/EMERGENCE_QUICK_START.md` - 본 파일

### 출력 (실행 후 생성됨)
- `data/emergence_labels/emergence_labels_train.pkl`
- `data/emergence_labels/emergence_labels_val.pkl`
- `data/emergence_labels/label_config.json`
- `data/emergence_labels/label_statistics.json`

---

## 주요 설정 (tools/create_emergence_labels.py)

```python
CONFIG = {
    'lookback_frames': 5,          # 과거 5 프레임
    'lookahead_frames': 3,         # 미래 3 프레임
    'min_distance': 5.0,           # 최소 5m
    'max_distance': 40.0,          # 최대 40m
    'bev_range': [-50, 50, -50, 50],  # BEV 범위
    'bev_resolution': 0.5,         # 0.5m/pixel
}
```

---

## Emergence 정의

**과거(t-5 ~ t-1)**: visibility = 1 (0-40%, 가려짐)  
**미래(t+1 ~ t+3)**: visibility ≥ 2 (40%+, 나타남)  
**거리**: 5-40m 범위 내

---

## 테스트 완료 ✅

- Mini dataset (v1.0-mini): 테스트 완료
- Coordinate 변환: 검증 완료
- Visibility 기반 감지: 정상 작동

**다음 단계**: Full dataset 실행 또는 모델 학습 시작

상세 문서는 `tools/emergence_labels_README.md` 참고
