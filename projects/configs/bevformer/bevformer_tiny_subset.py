_base_ = './bevformer_tiny.py'

# Full dataset 사용하되 샘플 수를 제한
# 전체 약 28,000 samples 중 일부만 사용
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        # 10% 샘플만 사용 (약 2,800 samples)
        # indices를 사용하거나 dataset을 subset으로 제한
        dataset=dict(
            times=1,  # 중복 없이 1번만
        ),
    ),
)

# Epoch 수를 늘려서 학습량 확보
total_epochs = 48  # 24 -> 48로 증가

# 평가 주기 조정
evaluation = dict(interval=2)  # 2 epoch마다 평가
checkpoint_config = dict(interval=2)  # 2 epoch마다 저장

