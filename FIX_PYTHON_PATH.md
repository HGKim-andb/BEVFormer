# Python Path Fix for Server

## Problem
서버에 mmdet3d가 egg 파일로 설치되어 있어서, 로컬 코드 변경사항이 반영되지 않습니다:
```
/home/sp/miniconda3/envs/uniad_test/lib/python3.8/site-packages/mmdet3d-0.17.1-py3.8-linux-x86_64.egg
```

## Solution

**중요**: BEVFormer 프로젝트 루트를 PYTHONPATH에 추가하면 안 됩니다!
- 프로젝트에 `mmdetection3d/` 소스 폴더가 있어서 설치된 mmdet3d와 충돌합니다
- **오직 `projects/` 디렉토리만** PYTHONPATH에 추가해야 합니다

### 방법 1: 스크립트 사용 (권장 - 이미 수정됨)

`dist_train.sh`와 `dist_test.sh`가 자동으로 `projects/` 디렉토리를 PYTHONPATH에 추가합니다.
```bash
cd ~/Project/BEVFormer2/BEVFormer

# Training - 그냥 실행
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
bash tools/dist_train.sh \
    projects/configs/bevformer/bevformer_risk_tiny_attention.py \
    4 \
    --work-dir ./work_dirs/bevformer_risk_tiny_attention

# Evaluation - 그냥 실행
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash tools/dist_test.sh \
    projects/configs/bevformer/bevformer_risk_tiny_attention.py \
    work_dirs/bevformer_risk_tiny_attention_1126/epoch_1.pth \
    4 \
    --eval bbox
```

### 방법 2: 환경 변수 설정

서버에서 ~/.bashrc에 추가 (프로젝트 루트가 아닌 projects 폴더):
```bash
# ~/.bashrc 파일 편집
nano ~/.bashrc

# 파일 끝에 다음 라인 추가 (projects 폴더만!):
export PYTHONPATH=/home/sp/Project/BEVFormer2/BEVFormer/projects:$PYTHONPATH

# 저장 후 적용
source ~/.bashrc
```

### 설치 확인
```bash
# Custom plugin이 projects 폴더에서 로드되는지 확인
python -c "import projects.mmdet3d_plugin.datasets.nuscenes_risk_dataset as m; print(m.__file__)"

# 출력이 다음과 같아야 합니다:
# /home/sp/Project/BEVFormer2/BEVFormer/projects/mmdet3d_plugin/datasets/nuscenes_risk_dataset.py

# mmdet3d는 설치된 패키지에서 로드되는지 확인
python -c "import mmdet3d; print(mmdet3d.__file__)"

# 출력이 site-packages 또는 .egg 파일이어야 합니다 (이게 정상!):
# /home/sp/miniconda3/envs/uniad_test/lib/python3.8/site-packages/mmdet3d-0.17.1-py3.8.egg/mmdet3d/__init__.py
```

## 확인사항

재실행 시 다음 디버그 메시지들이 보여야 합니다:
```
[Dataset Init] Loading risk labels from: .../risk_labels_val.pkl
[Dataset Init] File exists: True
[Dataset Init] Total risk labels: 80
[Dataset Init] First 5 sample tokens: [...]

[Risk Eval] Starting risk evaluation with 80 results
[Risk Eval] Collected 80 valid samples for evaluation
```

그리고 최종 결과에 다음 메트릭들이 포함되어야 합니다:
```
Risk Evaluation Metrics:
  Risk MSE: 0.XXXX
  Risk RMSE: 0.XXXX
  Risk MAE: 0.XXXX
  Risk Precision@0.5: 0.XXXX
  Risk Recall@0.5: 0.XXXX
  Risk F1@0.5: 0.XXXX
  Max Risk Correlation: 0.XXXX
```

## Troubleshooting

만약 여전히 site-packages를 사용한다면:
```bash
# PYTHONPATH를 명시적으로 설정
export PYTHONPATH=/home/hg-main/data2/BEVFormer:$PYTHONPATH

# 또는 실행 시마다 설정
PYTHONPATH=/home/hg-main/data2/BEVFormer:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash tools/dist_test.sh ...
```
