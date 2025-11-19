"""
BEVFormer with Risk Prediction - Tiny Version (20% Dataset)

This config extends bevformer_tiny.py to add risk prediction capability.
Uses 20% subsampled dataset for faster training.
"""

_base_ = ['./bevformer_tiny.py']

# Image normalization config (inherited from base but redefined for clarity)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Model modifications
model = dict(
    type='BEVFormerRisk',

    # Risk prediction head configuration
    risk_head=dict(
        type='RiskPredictionHead',
        in_channels=256,
        bev_h=50,
        bev_w=50,
        risk_h=200,
        risk_w=200,
        num_convs=3,
        conv_channels=128,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU'),
        use_sigmoid=True,
    ),

    # Risk configuration
    use_risk_guidance=False,  # Set to True to use risk-guided attention
    risk_loss_weight=100.0,   # Weight for risk loss (increased for sparse risk maps)
)

# Dataset modifications
dataset_type = 'NuScenesRiskDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(type='ObjectNameFilter', classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                                          'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                                                     'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_risk_map'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=False,
        use_valid_flag=True,
        bev_size=(200, 200),
        queue_length=3,
        # Risk-specific settings
        use_risk=True,
        risk_labels_path='data/emergence_risk_v5_full/risk_labels_train_20pct.pkl',
        risk_map_size=(200, 200),
        risk_threshold=0.0,  # Set > 0 to filter low-risk samples
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        bev_size=(200, 200),
        classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        # Risk-specific settings
        use_risk=True,
        risk_labels_path='data/emergence_risk_v5_full/risk_labels_val_20pct.pkl',
        risk_map_size=(200, 200),
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        bev_size=(200, 200),
        classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        use_risk=True,
        risk_labels_path='data/emergence_risk_v5_full/risk_labels_val_20pct.pkl',
        risk_map_size=(200, 200),
        box_type_3d='LiDAR'),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

# Learning rate and optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'risk_head': dict(lr_mult=1.0),  # Full learning rate for risk head
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# Learning rate scheduler
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

total_epochs = 24
evaluation = dict(interval=24, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)

# Runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/bevformer_risk_tiny'
load_from = None
resume_from = None
workflow = [('train', 1)]

# fp16 settings
fp16 = dict(loss_scale=512.)
