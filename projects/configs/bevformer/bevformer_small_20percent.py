# Full dataset의 20%만 사용하는 설정
# bevformer_small 기반

_base_ = './bevformer_small.py'

dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'

# 20% subset pkl 파일 사용
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train_20percent_dense.pkl',  # 20% subset
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',  # val은 그대로
    ),
)

# Epoch 수 조정 (데이터가 적으니 약간 더 많이)
total_epochs = 36  # 24 -> 36
runner = dict(type='EpochBasedRunner', max_epochs=36)

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

