# Full dataset의 30%만 사용하는 설정
# bevformer_small 기반

_base_ = './bevformer_small.py'

dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'

# 30% subset pkl 파일 사용
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train_30percent.pkl',  # 30% subset
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',  # val은 그대로
    ),
)

# Epoch 수 조정
total_epochs = 32  # 24 -> 32
runner = dict(type='EpochBasedRunner', max_epochs=32)

# 평가/저장 주기
evaluation = dict(interval=2)
checkpoint_config = dict(interval=2)

# 로그
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

