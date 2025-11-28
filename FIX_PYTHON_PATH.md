# Python Path Fix for Server

## Problem
서버에 mmdet3d가 egg 파일로 설치되어 있어서, 로컬 코드 변경사항이 반영되지 않습니다:
```
/home/sp/miniconda3/envs/uniad_test/lib/python3.8/site-packages/mmdet3d-0.17.1-py3.8-linux-x86_64.egg
```

## Solution

BEVFormer는 setup.py가 없으므로 PYTHONPATH로 해결합니다.

### 방법 1: 환경 변수 설정 (권장)

서버에서 ~/.bashrc에 추가:
```bash
# ~/.bashrc 파일 편집
nano ~/.bashrc

# 파일 끝에 다음 라인 추가:
export PYTHONPATH=/home/sp/Project/BEVFormer2/BEVFormer:$PYTHONPATH

# 저장 후 적용
source ~/.bashrc
```

### 방법 2: 실행 시마다 PYTHONPATH 설정

매번 실행할 때 명시적으로 설정:
```bash
cd ~/Project/BEVFormer2/BEVFormer

# Training
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=/home/sp/Project/BEVFormer2/BEVFormer:$PYTHONPATH \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
bash tools/dist_train.sh \
    projects/configs/bevformer/bevformer_risk_tiny_attention.py \
    4 \
    --work-dir ./work_dirs/bevformer_risk_tiny_attention

# Evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=/home/sp/Project/BEVFormer2/BEVFormer:$PYTHONPATH \
bash tools/dist_test.sh \
    projects/configs/bevformer/bevformer_risk_tiny_attention.py \
    work_dirs/bevformer_risk_tiny_attention_1126/epoch_1.pth \
    4 \
    --eval bbox
```

### 설치 확인
```bash
# PYTHONPATH 설정 후 확인
PYTHONPATH=/home/sp/Project/BEVFormer2/BEVFormer:$PYTHONPATH \
python -c "import projects.mmdet3d_plugin.datasets.nuscenes_risk_dataset as m; print(m.__file__)"

# 출력이 다음과 같아야 합니다:
# /home/sp/Project/BEVFormer2/BEVFormer/projects/mmdet3d_plugin/datasets/nuscenes_risk_dataset.py
# (site-packages가 아님!)
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
