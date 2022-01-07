# model settings
norm_cfg = dict(type='BN', requires_grad=False)
model = dict(
    type='FasterRCNN',
    pretrained='open-mmlab://resnet50_caffe', #resnet50_caffe
    backbone=dict(
        type='ResNet',
        depth=50, #50
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(2, ),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=True,
        style='caffe'),
    shared_head=dict(
        type='ResLayer',
        depth=50, #50
        stage=3,
        stride=1,
        dilation=2,
        style='caffe',
        norm_cfg=norm_cfg,
        norm_eval=True),
    rpn_head=dict(
        type='RPNHead',
        in_channels=1024,
        feat_channels=512,
        anchor_scales=[2, 4, 8, 16, 32],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[16],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=1024,
        featmap_strides=[16]),
    bbox_head=dict(
        type='BBoxHead',
        with_avg_pool=False, #True
        roi_feat_size=7,
        in_channels=1024,
        num_classes=31,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    selsa_head=dict(
        type='SelsaHead',
        in_channels=256,
        out_channels=1024,
        nongt_dim=3,  # number of frames
        feat_dim=1024,  # 1024
        dim=[1024, 1024, 1024],
        norm_cfg=norm_cfg,
        norm_eval=True,
        apply=True)
)
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=12000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False)
)
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=6000,
        nms_post=1000,
        max_num=300, #1000
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.001, nms=dict(type='nms', iou_thr=0.3), max_per_img=300)) #0.05
# dataset settings
dataset_type = 'VIDDataset'
data_root = '/home/hschen/Sequence-Level-Semantics-Aggregation/data/ILSVRC/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False) #maybe need to change
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(600, 1000), keep_ratio=True),
    # when remove the crop, the gpu-util will be 100% then suspend
    dict(type='RandomCrop', crop_size=(3000, 3000)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(600, 1000),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomCrop', crop_size=(3000, 3000)),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'img_norm_cfg', 'img_info')),
        ])
]
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        image_set='DET_train_30classes+VID_train_15frames',#VID_train_15frames
        ann_file=data_root,
        img_prefix=data_root,
        pipeline=train_pipeline,
        selsa_offset=dict(MAX_OFFSET=9,MIN_OFFSET=-9)),
    val=dict(
        type=dataset_type,
        image_set='VID_val_frames',
        ann_file=data_root,
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        image_set='VID_val_videos', #VID_val_frames VID_val_frames_shuffle VID_val_videos
        ann_file=data_root,
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1) #control eval interval epoch
# optimizer
optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=1000,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/faster_rcnn_r50_caffe_c4_1x_selsa'
load_from = None
resume_from = None
# resume_from = '/media/data1/jliang_data/detection/third/mmdetection/work_dirs/faster_rcnn_r50_caffe_c4_1x_selsa/latest.pth'
workflow = [('train', 1)]
