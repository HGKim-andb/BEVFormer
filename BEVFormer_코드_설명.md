# BEVFormer 논문 구조와 코드 매핑 가이드

## 목차
1. [BEVFormer 개요](#1-bevformer-개요)
2. [전체 아키텍처](#2-전체-아키텍처)
3. [각 컴포넌트별 상세 설명](#3-각-컴포넌트별-상세-설명)
4. [핵심 함수 위치 정리](#4-핵심-함수-위치-정리)
5. [논문 용어와 코드 매핑](#5-논문-용어와-코드-매핑)

---

## 1. BEVFormer 개요

### 논문 정보
- **제목**: BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers
- **저자**: Zhiqi Li et al.
- **출판**: ECCV 2022
- **핵심 아이디어**: 다중 카메라 이미지에서 시공간 Transformer를 사용하여 BEV 표현을 학습하고 3D 객체 탐지 수행

### 주요 특징
- **Spatial Cross-Attention (SCA)**: 다중 카메라 뷰에서 공간 정보 수집
- **Temporal Self-Attention (TSA)**: 이전 프레임의 BEV 정보와 융합
- **BEV Queries**: BEV 공간을 격자로 나눈 학습 가능한 쿼리
- **Ego Motion Compensation**: 차량 움직임에 따른 좌표계 변환

---

## 2. 전체 아키텍처

### 논문 Figure 2 구조
```
다중 카메라 이미지 
    ↓
Image Backbone (ResNet + FPN)
    ↓
BEV Queries 초기화
    ↓
BEV Encoder (반복)
    ├─ Temporal Self-Attention (TSA)
    └─ Spatial Cross-Attention (SCA)
    ↓
BEV Features
    ↓
Transformer Decoder
    ↓
Detection Head (분류 + 회귀)
    ↓
3D 바운딩 박스 예측
```

### 코드 흐름
1. **입력**: 6개 카메라 이미지 (B, N_cam, C, H, W)
2. **특징 추출**: `extract_img_feat()` → 다중 스케일 특징 맵
3. **BEV 생성**: `get_bev_features()` → BEV 특징 생성
4. **객체 탐지**: `forward()` → 3D 바운딩 박스 예측

---

## 3. 각 컴포넌트별 상세 설명

### 3.1 Image Backbone (이미지 특징 추출)

#### 논문 설명
- ResNet 백본과 FPN 네크를 사용하여 다중 카메라 이미지에서 특징 추출
- 각 카메라별로 다중 스케일 특징 맵 생성 (일반적으로 4개 레벨)

#### 코드 위치
**파일**: `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py`

**함수**: `extract_img_feat()`
```python
# 라인 67-100
def extract_img_feat(self, img, img_metas, len_queue=None):
    """Extract features of images."""
    B = img.size(0)
    if img.dim() == 5:
        B, N, C, H, W = img.size()
        img = img.reshape(B * N, C, H, W)
    
    if self.use_grid_mask:
        img = self.grid_mask(img)  # 데이터 증강
    
    img_feats = self.img_backbone(img)  # ResNet backbone
    if isinstance(img_feats, dict):
        img_feats = list(img_feats.values())
    
    if self.with_img_neck:
        img_feats = self.img_neck(img_feats)  # FPN neck
    
    # 다중 스케일 특징 재구성: [B, N_cam, C, H, W]
    img_feats_reshaped = []
    for img_feat in img_feats:
        BN, C, H, W = img_feat.size()
        img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
    
    return img_feats_reshaped
```

**역할**:
- 입력 이미지를 ResNet 백본으로 통과시켜 특징 추출
- FPN으로 다중 스케일 특징 맵 생성
- 각 카메라별 특징을 분리하여 반환

**입력 형식**: `(B, N_cam, C, H, W)` - 배치, 카메라 수, 채널, 높이, 너비
**출력 형식**: `List[Tensor]` - 각 스케일별 `(B, N_cam, C, H_i, W_i)`

---

### 3.2 BEV Queries 초기화

#### 논문 설명
- BEV 공간을 격자로 나눔 (예: 200×200)
- 각 격자점에 대해 학습 가능한 쿼리 임베딩 사용
- BEV 쿼리는 BEV 공간의 특정 위치를 나타냄

#### 코드 위치
**파일**: `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py`

**함수**: `_init_layers()`
```python
# 라인 103-107
if not self.as_two_stage:
    self.bev_embedding = nn.Embedding(
        self.bev_h * self.bev_w, self.embed_dims)  # BEV 쿼리
    self.query_embedding = nn.Embedding(
        self.num_query, self.embed_dims * 2)  # 객체 쿼리
```

**사용 위치**: `forward()`
```python
# 라인 136-137
bev_queries = self.bev_embedding.weight.to(dtype)  # (bev_h*bev_w, embed_dims)
object_query_embeds = self.query_embedding.weight.to(dtype)  # (num_query, embed_dims*2)
```

**역할**:
- BEV 공간의 각 격자점에 대한 학습 가능한 임베딩 생성
- 객체 탐지를 위한 객체 쿼리 임베딩 생성

**파라미터**:
- `bev_h`, `bev_w`: BEV 공간의 높이와 너비 (예: 200, 200)
- `embed_dims`: 임베딩 차원 (일반적으로 256)
- `num_query`: 객체 쿼리 개수 (일반적으로 900)

---

### 3.3 BEV Encoder (핵심 컴포넌트)

#### 논문 설명
- BEV Encoder는 Temporal Self-Attention과 Spatial Cross-Attention을 반복
- 여러 레이어를 쌓아 BEV 특징을 점진적으로 개선

#### 코드 위치
**파일**: `projects/mmdet3d_plugin/bevformer/modules/encoder.py`

**클래스**: `BEVFormerEncoder`
```python
# 라인 24-239
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):
    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, ...):
        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
```

**Forward 함수**: `forward()`
```python
# 라인 151-239
def forward(self, bev_query, key, value, bev_h, bev_w, bev_pos, 
            prev_bev=None, shift=0., **kwargs):
    # 1. 3D reference points 생성 (SCA용)
    ref_3d = self.get_reference_points(bev_h, bev_w, ..., dim='3d', ...)
    
    # 2. 2D reference points 생성 (TSA용)
    ref_2d = self.get_reference_points(bev_h, bev_w, dim='2d', ...)
    
    # 3. 3D 점을 카메라 이미지로 투영
    reference_points_cam, bev_mask = self.point_sampling(
        ref_3d, self.pc_range, kwargs['img_metas'])
    
    # 4. Ego motion 보정
    shift_ref_2d = ref_2d.clone()
    shift_ref_2d += shift[:, None, None, :]
    
    # 5. 이전 BEV와 현재 쿼리 결합
    if prev_bev is not None:
        prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
        hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(...)
    
    # 6. 각 레이어 통과
    for lid, layer in enumerate(self.layers):
        output = layer(bev_query, key, value, 
                      ref_2d=hybird_ref_2d, ref_3d=ref_3d,
                      reference_points_cam=reference_points_cam, ...)
        bev_query = output
    
    return output
```

**역할**:
- BEV 쿼리를 입력받아 BEV 특징 생성
- TSA와 SCA를 반복하여 공간적/시간적 정보 융합

---

### 3.3.1 Temporal Self-Attention (TSA)

#### 논문 설명
- **목적**: 이전 프레임의 BEV 특징과 현재 BEV 쿼리를 융합하여 시간적 일관성 확보
- **방법**: Deformable attention을 사용하여 이전 BEV에서 관련 위치 샘플링
- **Ego Motion**: 차량 움직임에 따라 이전 BEV를 현재 좌표계로 변환

#### 코드 위치
**파일**: `projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py`

**클래스**: `TemporalSelfAttention`
```python
# 라인 25-272
@ATTENTION.register_module()
class TemporalSelfAttention(BaseModule):
    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, 
                 num_points=4, num_bev_queue=2, ...):
        # num_bev_queue=2: 이전 BEV + 현재 BEV
        self.sampling_offsets = nn.Linear(
            embed_dims*self.num_bev_queue, 
            num_bev_queue*num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims*self.num_bev_queue,
            num_bev_queue*num_heads * num_levels * num_points)
```

**Forward 함수**: `forward()`
```python
# 라인 128-272
def forward(self, query, key=None, value=None, reference_points=None, 
            prev_bev=None, **kwargs):
    # 1. 이전 BEV와 현재 쿼리 결합
    query = torch.cat([value[:bs], query], -1)  # (bs, num_query, embed_dims*2)
    value = self.value_proj(value)  # (bs*2, num_query, embed_dims)
    
    # 2. Sampling offsets와 attention weights 예측
    sampling_offsets = self.sampling_offsets(query)  # (bs, num_query, ...)
    attention_weights = self.attention_weights(query)
    attention_weights = attention_weights.softmax(-1)
    
    # 3. Reference points 기반 샘플링 위치 계산
    sampling_locations = reference_points[:, :, None, :, None, :] + \
                        sampling_offsets / offset_normalizer
    
    # 4. Deformable attention 수행
    output = MultiScaleDeformableAttnFunction.apply(
        value, spatial_shapes, level_start_index, 
        sampling_locations, attention_weights, ...)
    
    # 5. 이전/현재 BEV 특징 융합 (평균)
    output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
    output = output.mean(-1)  # (num_query, embed_dims, bs)
    
    output = self.output_proj(output)
    return self.dropout(output) + identity
```

**호출 위치**: `projects/mmdet3d_plugin/bevformer/modules/encoder.py`
```python
# 라인 357-373 (BEVFormerLayer 내부)
if layer == 'self_attn':
    query = self.attentions[attn_index](
        query, prev_bev, prev_bev,
        reference_points=ref_2d,  # 2D BEV 평면 기준
        spatial_shapes=torch.tensor([[bev_h, bev_w]], ...),
        ...
    )
```

**역할**:
- 이전 프레임의 BEV 특징을 현재 프레임에 통합
- Ego motion을 고려하여 좌표계 변환
- 시간적 정보를 활용하여 동적 객체 탐지 성능 향상

**입력**:
- `query`: 현재 BEV 쿼리 `(bs, num_query, embed_dims)`
- `prev_bev`: 이전 프레임 BEV 특징 `(bs, num_query, embed_dims)`
- `reference_points`: 2D BEV 평면 기준 reference points

**출력**: 융합된 BEV 쿼리 `(bs, num_query, embed_dims)`

---

### 3.3.2 Spatial Cross-Attention (SCA)

#### 논문 설명
- **목적**: 각 BEV 쿼리가 다중 카메라 뷰에서 해당하는 3D 위치의 특징을 수집
- **방법**: 
  1. BEV 쿼리 위치를 3D 공간으로 변환
  2. 3D 점을 각 카메라 이미지 평면으로 투영
  3. Deformable attention으로 해당 위치의 특징 샘플링
  4. 모든 카메라에서 수집한 특징을 평균

#### 코드 위치
**파일**: `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py`

**클래스**: `SpatialCrossAttention`
```python
# 라인 31-175
@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    def __init__(self, embed_dims=256, num_cams=6, pc_range=None, ...):
        self.deformable_attention = build_attention(deformable_attention)
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
```

**Forward 함수**: `forward()`
```python
# 라인 75-175
def forward(self, query, key, value, reference_points, 
            reference_points_cam, bev_mask, ...):
    # 1. 각 카메라별로 해당하는 BEV 쿼리만 선택 (메모리 절약)
    indexes = []
    for i, mask_per_img in enumerate(bev_mask):
        index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
        indexes.append(index_query_per_img)
    
    queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
    reference_points_rebatch = reference_points_cam.new_zeros(
        [bs, self.num_cams, max_len, D, 2])
    
    # 2. 각 카메라별 쿼리 재구성
    for j in range(bs):
        for i, reference_points_per_img in enumerate(reference_points_cam):
            index_query_per_img = indexes[i]
            queries_rebatch[j, i, :len(index_query_per_img)] = \
                query[j, index_query_per_img]
            reference_points_rebatch[j, i, :len(index_query_per_img)] = \
                reference_points_per_img[j, index_query_per_img]
    
    # 3. 각 카메라와 해당 BEV 쿼리 간 deformable attention
    queries = self.deformable_attention(
        query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims),
        key=key, value=value,
        reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2),
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index
    ).view(bs, self.num_cams, max_len, self.embed_dims)
    
    # 4. 모든 카메라에서 수집한 특징을 평균
    slots = torch.zeros_like(query)
    for j in range(bs):
        for i, index_query_per_img in enumerate(indexes):
            slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]
    
    count = bev_mask.sum(-1) > 0
    count = count.permute(1, 2, 0).sum(-1)
    slots = slots / count[..., None]  # 평균화
    
    slots = self.output_proj(slots)
    return self.dropout(slots) + inp_residual
```

**호출 위치**: `projects/mmdet3d_plugin/bevformer/modules/encoder.py`
```python
# 라인 381-397 (BEVFormerLayer 내부)
elif layer == 'cross_attn':
    query = self.attentions[attn_index](
        query, key, value,
        reference_points=ref_3d,  # 3D 공간 기준
        reference_points_cam=reference_points_cam,  # 카메라 투영 좌표
        mask=bev_mask,
        spatial_shapes=spatial_shapes,
        ...
    )
```

**역할**:
- BEV 쿼리를 3D 공간으로 변환하여 각 카메라 이미지에서 특징 수집
- 다중 카메라 뷰의 정보를 융합하여 BEV 표현 생성

**입력**:
- `query`: BEV 쿼리 `(bs, num_query, embed_dims)`
- `key`, `value`: 다중 카메라 특징 `(num_cam, H*W, bs, embed_dims)`
- `reference_points_cam`: 카메라 투영 좌표 `(num_cam, bs, num_query, D, 2)`
- `bev_mask`: 유효한 영역 마스크

**출력**: 업데이트된 BEV 쿼리 `(bs, num_query, embed_dims)`

---

### 3.3.3 3D Reference Points 생성 및 Camera Projection

#### 논문 설명
- BEV 쿼리 위치를 3D 공간으로 변환
- 각 BEV 쿼리에 대해 여러 높이(z) 레벨 생성 (pillar 구조)
- 3D 점을 각 카메라 이미지 평면으로 투영

#### 코드 위치
**파일**: `projects/mmdet3d_plugin/bevformer/modules/encoder.py`

**함수**: `get_reference_points()`
```python
# 라인 46-85
@staticmethod
def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', ...):
    """Get the reference points used in SCA and TSA."""
    
    # 3D reference points (SCA용)
    if dim == '3d':
        zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, ...) / Z
        xs = torch.linspace(0.5, W - 0.5, W, ...) / W
        ys = torch.linspace(0.5, H - 0.5, H, ...) / H
        ref_3d = torch.stack((xs, ys, zs), -1)  # (num_points_in_pillar, H, W, 3)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
        return ref_3d  # (bs, H*W, num_points_in_pillar, 3)
    
    # 2D reference points (TSA용)
    elif dim == '2d':
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, ...),
            torch.linspace(0.5, W - 0.5, W, ...)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        return ref_2d  # (bs, H*W, 1, 2)
```

**함수**: `point_sampling()`
```python
# 라인 88-149
@force_fp32(apply_to=('reference_points', 'img_metas'))
def point_sampling(self, reference_points, pc_range, img_metas):
    """3D 점을 카메라 이미지 평면으로 투영"""
    
    # 1. lidar2img 변환 행렬 가져오기
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)  # (B, N, 4, 4)
    
    # 2. Normalized 좌표를 실제 3D 좌표로 변환
    reference_points[..., 0:1] = reference_points[..., 0:1] * \
        (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * \
        (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * \
        (pc_range[5] - pc_range[2]) + pc_range[2]
    
    # 3. Homogeneous 좌표로 변환
    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1)
    
    # 4. 3D -> 2D 투영 (lidar2img 변환)
    reference_points_cam = torch.matmul(
        lidar2img.to(torch.float32),
        reference_points.to(torch.float32)
    ).squeeze(-1)
    
    # 5. 유효한 영역 마스킹
    bev_mask = (reference_points_cam[..., 2:3] > eps)  # depth > 0
    reference_points_cam = reference_points_cam[..., 0:2] / \
        torch.maximum(reference_points_cam[..., 2:3], ...)
    
    # 이미지 크기로 정규화
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    
    # 유효 영역 체크 (0~1 범위)
    bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0) &
                (reference_points_cam[..., 1:2] < 1.0) &
                (reference_points_cam[..., 0:1] < 1.0) &
                (reference_points_cam[..., 0:1] > 0.0))
    
    return reference_points_cam, bev_mask
```

**역할**:
- BEV 쿼리를 3D 공간 좌표로 변환
- 3D 점을 각 카메라 이미지 평면으로 투영
- 유효한 영역만 마스킹하여 처리

**파라미터**:
- `num_points_in_pillar`: 각 BEV 쿼리당 높이 방향 샘플링 포인트 수 (기본값: 4)
- `pc_range`: Point cloud 범위 `[x_min, y_min, z_min, x_max, y_max, z_max]`

---

### 3.3.4 Ego Motion Compensation

#### 논문 설명
- 차량의 움직임(translation, rotation)을 고려하여 이전 프레임의 BEV를 현재 좌표계로 변환
- CAN bus 신호를 사용하여 ego motion 정보 획득

#### 코드 위치
**파일**: `projects/mmdet3d_plugin/bevformer/modules/transformer.py`

**함수**: `get_bev_features()`
```python
# 라인 103-200
def get_bev_features(self, mlvl_feats, bev_queries, bev_h, bev_w, 
                     bev_pos=None, prev_bev=None, **kwargs):
    # 1. Ego motion 정보 추출
    delta_x = np.array([each['can_bus'][0] for each in kwargs['img_metas']])
    delta_y = np.array([each['can_bus'][1] for each in kwargs['img_metas']])
    ego_angle = np.array([each['can_bus'][-2] / np.pi * 180 
                          for each in kwargs['img_metas']])
    
    # 2. Translation과 rotation 계산
    translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
    translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
    bev_angle = ego_angle - translation_angle
    
    # 3. BEV 공간에서의 shift 계산
    shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / \
              grid_length_y / bev_h
    shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / \
              grid_length_x / bev_w
    shift = bev_queries.new_tensor([shift_x, shift_y]).permute(1, 0)
    
    # 4. 이전 BEV 회전 (rotation)
    if prev_bev is not None and self.rotate_prev_bev:
        for i in range(bs):
            rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
            tmp_prev_bev = prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
            tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, 
                                 center=self.rotate_center)
            tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(bev_h * bev_w, 1, -1)
            prev_bev[:, i] = tmp_prev_bev[:, 0]
    
    # 5. CAN bus 신호를 BEV 쿼리에 추가
    can_bus = bev_queries.new_tensor(
        [each['can_bus'] for each in kwargs['img_metas']])
    can_bus = self.can_bus_mlp(can_bus)[None, :, :]
    bev_queries = bev_queries + can_bus * self.use_can_bus
```

**역할**:
- 차량의 움직임을 고려하여 이전 프레임 BEV를 현재 좌표계로 변환
- CAN bus 신호를 활용하여 ego motion 정보 인코딩

**CAN bus 신호**: 18차원 벡터
- `can_bus[0:3]`: Translation (x, y, z)
- `can_bus[3:6]`: Velocity
- `can_bus[6:9]`: Acceleration
- `can_bus[9:12]`: Angular velocity
- `can_bus[12:15]`: Angular acceleration
- `can_bus[15:18]`: Rotation (yaw, pitch, roll)

---

### 3.4 Transformer Decoder (객체 탐지)

#### 논문 설명
- BEV 특징을 객체 쿼리로 변환하여 3D 바운딩 박스 예측
- DETR 스타일의 decoder 구조 사용
- Reference points를 점진적으로 개선

#### 코드 위치
**파일**: `projects/mmdet3d_plugin/bevformer/modules/decoder.py`

**클래스**: `DetectionTransformerDecoder`
```python
# 라인 52-129
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetectionTransformerDecoder(TransformerLayerSequence):
    def forward(self, query, *args, reference_points=None, 
                reg_branches=None, **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = []
        
        for lid, layer in enumerate(self.layers):
            # Reference points 입력 준비
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            
            # Decoder layer 통과
            output = layer(output, *args, 
                          reference_points=reference_points_input, ...)
            output = output.permute(1, 0, 2)
            
            # Reference points 개선
            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + \
                    inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[..., 4:5] + \
                    inverse_sigmoid(reference_points[..., 2:3])
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        
        return output, reference_points
```

**호출 위치**: `projects/mmdet3d_plugin/bevformer/modules/transformer.py`
```python
# 라인 275-285
inter_states, inter_references = self.decoder(
    query=query,  # 객체 쿼리
    key=None,
    value=bev_embed,  # BEV 특징
    query_pos=query_pos,
    reference_points=reference_points,  # 초기 3D 위치
    reg_branches=reg_branches,
    cls_branches=cls_branches,
    ...
)
```

**역할**:
- BEV 특징을 객체 쿼리로 변환
- Reference points를 점진적으로 개선하여 정확한 3D 바운딩 박스 예측

---

### 3.5 Detection Head

#### 논문 설명
- 최종 분류 및 바운딩 박스 회귀 수행
- 각 decoder layer에서 중간 예측 생성 (auxiliary loss)

#### 코드 위치
**파일**: `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py`

**클래스**: `BEVFormerHead`
```python
# 라인 16-213
@HEADS.register_module()
class BEVFormerHead(DETRHead):
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        # 1. BEV 특징 생성
        outputs = self.transformer(mlvl_feats, bev_queries, 
                                   object_query_embeds, ...)
        bev_embed, hs, init_reference, inter_references = outputs
        
        # 2. 각 decoder layer에서 예측
        hs = hs.permute(0, 2, 1, 3)  # (num_layers, bs, num_query, embed_dims)
        outputs_classes = []
        outputs_coords = []
        
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            
            reference = inverse_sigmoid(reference)
            
            # 분류 예측
            outputs_class = self.cls_branches[lvl](hs[lvl])
            
            # 회귀 예측
            tmp = self.reg_branches[lvl](hs[lvl])
            
            # Reference points 기반 바운딩 박스 계산
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]  # x, y
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]  # z
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            
            # 실제 좌표로 변환
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + 
                            self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + 
                            self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + 
                            self.pc_range[2])
            
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        
        return {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            ...
        }
```

**Loss 계산**: `loss()`
```python
# 라인 395-480
def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts, ...):
    all_cls_scores = preds_dicts['all_cls_scores']
    all_bbox_preds = preds_dicts['all_bbox_preds']
    
    # 각 decoder layer에 대해 loss 계산
    losses_cls, losses_bbox = multi_apply(
        self.loss_single, all_cls_scores, all_bbox_preds,
        all_gt_bboxes_list, all_gt_labels_list, ...)
    
    loss_dict = dict()
    loss_dict['loss_cls'] = losses_cls[-1]  # 마지막 layer
    loss_dict['loss_bbox'] = losses_bbox[-1]
    
    # Auxiliary loss (중간 layer들)
    for num_dec_layer, (loss_cls_i, loss_bbox_i) in enumerate(
        zip(losses_cls[:-1], losses_bbox[:-1])):
        loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
        loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
    
    return loss_dict
```

**역할**:
- Transformer decoder의 출력을 분류 및 회귀 예측으로 변환
- 각 decoder layer에서 중간 예측 생성 (auxiliary loss)
- 최종 3D 바운딩 박스 좌표 계산

**출력 형식**:
- `all_cls_scores`: `(num_layers, bs, num_query, num_classes)` - 분류 점수
- `all_bbox_preds`: `(num_layers, bs, num_query, 10)` - 바운딩 박스 (x, y, z, w, l, h, yaw, vx, vy, vz)

---

### 3.6 전체 Forward 흐름

#### 학습 시 (forward_train)

**코드 위치**: `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py`

**함수**: `forward_train()`
```python
# 라인 179-234
def forward_train(self, points=None, img_metas=None, gt_bboxes_3d=None,
                  gt_labels_3d=None, img=None, ...):
    # 1. 시간적 큐에서 이전 프레임과 현재 프레임 분리
    len_queue = img.size(1)
    prev_img = img[:, :-1, ...]  # 이전 프레임들
    img = img[:, -1, ...]  # 현재 프레임
    
    # 2. 이전 프레임들로부터 BEV 특징 생성
    prev_img_metas = copy.deepcopy(img_metas)
    prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
    
    # 3. 현재 프레임 메타데이터 추출
    img_metas = [each[len_queue-1] for each in img_metas]
    if not img_metas[0]['prev_bev_exists']:
        prev_bev = None
    
    # 4. 현재 프레임 특징 추출
    img_feats = self.extract_feat(img=img, img_metas=img_metas)
    
    # 5. Transformer로 BEV 특징 생성 및 객체 탐지
    losses = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d,
                                    img_metas, gt_bboxes_ignore, prev_bev)
    
    return losses
```

**함수**: `obtain_history_bev()`
```python
# 라인 158-177
def obtain_history_bev(self, imgs_queue, img_metas_list):
    """Obtain history BEV features iteratively."""
    self.eval()
    
    with torch.no_grad():
        prev_bev = None
        bs, len_queue, num_cams, C, H, W = imgs_queue.shape
        imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
        
        # 다중 프레임 특징 추출
        img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
        
        # 각 프레임 순차적으로 처리
        for i in range(len_queue):
            img_metas = [each[i] for each in img_metas_list]
            if not img_metas[0]['prev_bev_exists']:
                prev_bev = None
            
            img_feats = [each_scale[:, i] for each_scale in img_feats_list]
            prev_bev = self.pts_bbox_head(
                img_feats, img_metas, prev_bev, only_bev=True)
    
    self.train()
    return prev_bev
```

#### 추론 시 (forward_test)

**함수**: `forward_test()`
```python
# 라인 236-269
def forward_test(self, img_metas, img=None, **kwargs):
    # 1. Scene token 체크 (새로운 scene이면 이전 BEV 초기화)
    if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
        self.prev_frame_info['prev_bev'] = None
    
    self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']
    
    # 2. Temporal 정보 사용 여부 체크
    if not self.video_test_mode:
        self.prev_frame_info['prev_bev'] = None
    
    # 3. Ego motion 보정
    tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
    tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
    
    if self.prev_frame_info['prev_bev'] is not None:
        img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
        img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
    else:
        img_metas[0][0]['can_bus'][-1] = 0
        img_metas[0][0]['can_bus'][:3] = 0
    
    # 4. BEV 특징 생성 및 탐지
    new_prev_bev, bbox_results = self.simple_test(
        img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], ...)
    
    # 5. 다음 프레임을 위해 저장
    self.prev_frame_info['prev_pos'] = tmp_pos
    self.prev_frame_info['prev_angle'] = tmp_angle
    self.prev_frame_info['prev_bev'] = new_prev_bev
    
    return bbox_results
```

---

## 4. 핵심 함수 위치 정리

### 4.1 메인 모델 클래스

| 클래스/함수 | 파일 위치 | 라인 | 설명 |
|-----------|----------|------|------|
| `BEVFormer` | `detectors/bevformer.py` | 20-293 | 메인 모델 클래스 |
| `extract_img_feat()` | `detectors/bevformer.py` | 67-100 | 이미지 특징 추출 |
| `forward_train()` | `detectors/bevformer.py` | 179-234 | 학습 시 forward |
| `forward_test()` | `detectors/bevformer.py` | 236-269 | 추론 시 forward |
| `obtain_history_bev()` | `detectors/bevformer.py` | 158-177 | 이전 프레임 BEV 생성 |

### 4.2 Transformer 관련

| 클래스/함수 | 파일 위치 | 라인 | 설명 |
|-----------|----------|------|------|
| `PerceptionTransformer` | `modules/transformer.py` | 26-289 | 전체 Transformer 구조 |
| `get_bev_features()` | `modules/transformer.py` | 103-200 | BEV 특징 생성 |
| `BEVFormerEncoder` | `modules/encoder.py` | 24-239 | BEV Encoder |
| `BEVFormerLayer` | `modules/encoder.py` | 242-406 | Encoder 레이어 |
| `get_reference_points()` | `modules/encoder.py` | 46-85 | Reference points 생성 |
| `point_sampling()` | `modules/encoder.py` | 88-149 | 3D->2D 투영 |
| `DetectionTransformerDecoder` | `modules/decoder.py` | 52-129 | Detection Decoder |

### 4.3 Attention 모듈

| 클래스/함수 | 파일 위치 | 라인 | 설명 |
|-----------|----------|------|------|
| `TemporalSelfAttention` | `modules/temporal_self_attention.py` | 25-272 | 시간적 Self-Attention |
| `SpatialCrossAttention` | `modules/spatial_cross_attention.py` | 31-175 | 공간적 Cross-Attention |
| `MSDeformableAttention3D` | `modules/spatial_cross_attention.py` | 178-399 | 3D Deformable Attention |

### 4.4 Detection Head

| 클래스/함수 | 파일 위치 | 라인 | 설명 |
|-----------|----------|------|------|
| `BEVFormerHead` | `dense_heads/bevformer_head.py` | 16-213 | Detection Head |
| `forward()` | `dense_heads/bevformer_head.py` | 117-213 | Head forward |
| `loss()` | `dense_heads/bevformer_head.py` | 395-480 | Loss 계산 |
| `get_bboxes()` | `dense_heads/bevformer_head.py` | 482-509 | 바운딩 박스 디코딩 |

---

## 5. 논문 용어와 코드 매핑

### 5.1 주요 용어

| 논문 용어 | 코드 변수/함수 | 설명 |
|---------|--------------|------|
| **BEV Queries** | `bev_queries`, `bev_embedding` | BEV 공간의 격자 쿼리 |
| **Spatial Cross-Attention (SCA)** | `SpatialCrossAttention` | 다중 카메라에서 특징 수집 |
| **Temporal Self-Attention (TSA)** | `TemporalSelfAttention` | 이전 프레임 BEV와 융합 |
| **Reference Points** | `ref_3d`, `ref_2d` | 3D/2D 참조 점 |
| **Camera Projection** | `point_sampling()` | 3D 점을 이미지로 투영 |
| **Ego Motion** | `can_bus`, `shift` | 차량 움직임 정보 |
| **BEV Features** | `bev_embed` | BEV 공간 특징 |
| **Object Queries** | `object_query_embed` | 객체 탐지 쿼리 |
| **Deformable Attention** | `MSDeformableAttention3D` | 변형 가능한 attention |
| **Pillar** | `num_points_in_pillar` | 높이 방향 샘플링 포인트 |

### 5.2 수식과 코드 매핑

#### 논문 수식 1: Spatial Cross-Attention
```
SCA(Q_p, F_t) = (1/|V_hit|) * Σ_{k∈V_hit} Σ_{j=1}^{N_ref} DeformAttn(Q_p, P(p, j), F_t^k)
```

**코드 구현**:
```python
# spatial_cross_attention.py:162-167
queries = self.deformable_attention(
    query=queries_rebatch, key=key, value=value,
    reference_points=reference_points_rebatch, ...
)
# 평균화
slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]
slots = slots / count[..., None]
```

#### 논문 수식 2: Temporal Self-Attention
```
TSA(Q_t, B_{t-1}) = DeformAttn(Q_t, p, B_{t-1})
```

**코드 구현**:
```python
# temporal_self_attention.py:197-262
query = torch.cat([value[:bs], query], -1)  # 이전 BEV + 현재 쿼리
output = MultiScaleDeformableAttnFunction.apply(
    value, spatial_shapes, level_start_index,
    sampling_locations, attention_weights, ...
)
output = output.mean(-1)  # 이전/현재 융합
```

#### 논문 수식 3: Ego Motion Compensation
```
B_{t-1}' = Rotate(B_{t-1}, Δθ) + Shift(Δx, Δy)
```

**코드 구현**:
```python
# transformer.py:122-156
# Shift 계산
shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w

# Rotation
tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
```

### 5.3 하이퍼파라미터

| 파라미터 | 기본값 | 설명 | 위치 |
|---------|-------|------|------|
| `bev_h`, `bev_w` | 200, 200 | BEV 공간 크기 | `bevformer_head.py:37-38` |
| `embed_dims` | 256 | 임베딩 차원 | Config 파일 |
| `num_query` | 900 | 객체 쿼리 개수 | Config 파일 |
| `num_points_in_pillar` | 4 | 높이 방향 샘플링 포인트 | `encoder.py:36` |
| `num_cams` | 6 | 카메라 개수 | Config 파일 |
| `num_feature_levels` | 4 | FPN 특징 레벨 수 | Config 파일 |
| `num_bev_queue` | 2 | BEV 큐 길이 (이전+현재) | `temporal_self_attention.py:60` |

---

## 6. 데이터 흐름 요약

### 6.1 입력 → 출력

```
입력: 
  - img: (B, N_cam, C, H, W) - 다중 카메라 이미지
  - img_metas: 메타데이터 (카메라 파라미터, ego motion 등)
  - prev_bev: (B, H*W, C) - 이전 프레임 BEV (선택적)

처리:
  1. Image Backbone: img → mlvl_feats (다중 스케일 특징)
  2. BEV Encoder: mlvl_feats + bev_queries → bev_embed
     - TSA: 이전 BEV와 융합
     - SCA: 다중 카메라에서 특징 수집
  3. Detection Decoder: bev_embed + object_queries → hs
  4. Detection Head: hs → cls_scores, bbox_preds

출력:
  - all_cls_scores: (num_layers, B, num_query, num_classes)
  - all_bbox_preds: (num_layers, B, num_query, 10)
  - bev_embed: (B, H*W, C)
```

### 6.2 주요 텐서 Shape

| 텐서 | Shape | 설명 |
|------|-------|------|
| `img` | `(B, N_cam, C, H, W)` | 입력 이미지 |
| `mlvl_feats` | `List[(B, N_cam, C_i, H_i, W_i)]` | 다중 스케일 특징 |
| `bev_queries` | `(H*W, C)` | BEV 쿼리 |
| `bev_embed` | `(B, H*W, C)` | BEV 특징 |
| `ref_3d` | `(B, H*W, num_points_in_pillar, 3)` | 3D 참조 점 |
| `ref_2d` | `(B, H*W, 1, 2)` | 2D 참조 점 |
| `reference_points_cam` | `(N_cam, B, H*W, num_points_in_pillar, 2)` | 카메라 투영 좌표 |
| `object_queries` | `(num_query, C*2)` | 객체 쿼리 |
| `hs` | `(num_layers, B, num_query, C)` | Decoder 출력 |
| `cls_scores` | `(num_layers, B, num_query, num_classes)` | 분류 점수 |
| `bbox_preds` | `(num_layers, B, num_query, 10)` | 바운딩 박스 |

---

## 7. 참고 자료

### 7.1 논문
- **BEVFormer**: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers
- **arXiv**: https://arxiv.org/abs/2203.17270
- **GitHub**: https://github.com/fundamentalvision/BEVFormer

### 7.2 관련 논문
- **DETR**: End-to-End Object Detection with Transformers
- **Deformable DETR**: Deformable Transformers for End-to-End Object Detection
- **DETR3D**: 3D Object Detection from Multi-view Images via 3D-to-2D Queries

### 7.3 코드베이스 구조
```
BEVFormer/
├── projects/
│   ├── mmdet3d_plugin/
│   │   └── bevformer/
│   │       ├── detectors/          # 메인 모델
│   │       ├── modules/             # Transformer 모듈
│   │       ├── dense_heads/         # Detection Head
│   │       └── apis/               # 학습/추론 API
│   └── configs/
│       └── bevformer/               # 설정 파일
└── tools/                           # 유틸리티
```

---

## 8. 주요 개선 포인트

### 8.1 메모리 최적화
- **카메라별 쿼리 선택**: 각 카메라는 해당하는 BEV 쿼리만 처리 (`spatial_cross_attention.py:143-153`)
- **Gradient Checkpointing**: 학습 시 메모리 절약

### 8.2 성능 최적화
- **Deformable Attention**: 고정된 위치가 아닌 학습 가능한 샘플링
- **Multi-scale Features**: FPN의 다중 스케일 특징 활용

### 8.3 시간적 일관성
- **Ego Motion Compensation**: 차량 움직임 보정
- **Temporal Fusion**: 이전 프레임 정보 활용

---

**작성일**: 2024
**버전**: 1.0
**작성자**: BEVFormer 코드 분석

