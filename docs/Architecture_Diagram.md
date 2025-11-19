# Risk-Guided BEVFormer Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MULTI-CAMERA INPUT (6 cameras)                      │
│                         [B, 6, 3, H, W] (e.g., 1600×900)                    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IMAGE BACKBONE (ResNet-50/101)                        │
│                                                                               │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│  │ Conv Block  │ → │ Conv Block  │ → │ Conv Block  │ → │ Conv Block  │    │
│  │   Layer 1   │   │   Layer 2   │   │   Layer 3   │   │   Layer 4   │    │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘    │
│                                                                               │
│  Output: [B×6, 256, H/32, W/32] Multi-scale features                        │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           IMAGE NECK (FPN)                                   │
│                                                                               │
│  Multi-level features: [B×6, 256, H/8, W/8]                                 │
│                        [B×6, 256, H/16, W/16]                               │
│                        [B×6, 256, H/32, W/32]                               │
│                        [B×6, 256, H/64, W/64]                               │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       BEV TRANSFORMER ENCODER                                │
│                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    BEV Query Initialization                            │  │
│  │              [H_bev × W_bev, B, C] (e.g., 50×50, 256)                 │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │                                           │
│                                  ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                   Transformer Encoder Layers (6 layers)                │  │
│  │                                                                         │  │
│  │  Each Layer:                                                            │  │
│  │  ┌─────────────────────────────────────────────────────────┐           │  │
│  │  │  1. Temporal Self-Attention                             │           │  │
│  │  │     - Align with previous BEV (if available)            │           │  │
│  │  │     - Ego motion compensation                           │           │  │
│  │  └─────────────────────────────────────────────────────────┘           │  │
│  │                          ↓                                              │  │
│  │  ┌─────────────────────────────────────────────────────────┐           │  │
│  │  │  2. Spatial Cross-Attention (Deformable)                │           │  │
│  │  │     - Query BEV features from multi-camera images       │           │  │
│  │  │     - 3D-to-2D projection with camera parameters        │           │  │
│  │  │     - Multi-scale, Multi-camera aggregation             │           │  │
│  │  └─────────────────────────────────────────────────────────┘           │  │
│  │                          ↓                                              │  │
│  │  ┌─────────────────────────────────────────────────────────┐           │  │
│  │  │  3. Feed-Forward Network                                │           │  │
│  │  └─────────────────────────────────────────────────────────┘           │  │
│  │                                                                         │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │                                           │
│  Output: BEV Features [H_bev×W_bev, B, C] = [2500, B, 256]                 │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BEV FEATURES (Shared Representation)                      │
│                         [B, H_bev×W_bev, C] or [B, C, H_bev, W_bev]         │
│                              [B, 2500, 256] or [B, 256, 50, 50]             │
└───────────────┬──────────────────────────────────────────┬──────────────────┘
                │                                          │
                │                                          │
       ┌────────▼─────────┐                       ┌────────▼─────────┐
       │                  │                       │                  │
       │  DETECTION PATH  │                       │   RISK PATH      │
       │                  │                       │   (NEW!)         │
       └────────┬─────────┘                       └────────┬─────────┘
                │                                          │
                ▼                                          ▼
┌───────────────────────────────┐         ┌───────────────────────────────────┐
│   TRANSFORMER DECODER         │         │   RISK PREDICTION HEAD            │
│                               │         │                                   │
│ Object Queries (900)          │         │ ┌─────────────────────────────┐   │
│        ↓                      │         │ │  Conv 256→128 (3×3)         │   │
│ Multi-scale Deformable Attn   │         │ │  BN + ReLU                  │   │
│        ↓                      │         │ └──────────────┬──────────────┘   │
│ Self-Attention                │         │                ↓                  │
│        ↓                      │         │ ┌─────────────────────────────┐   │
│ FFN                           │         │ │  Conv 128→128 (3×3)         │   │
│        ↓                      │         │ │  BN + ReLU                  │   │
│ 6 Decoder Layers              │         │ └──────────────┬──────────────┘   │
│                               │         │                ↓                  │
│ Output: [B, 900, 256]         │         │ ┌─────────────────────────────┐   │
└───────────┬───────────────────┘         │ │  Conv 128→64 (3×3)          │   │
            │                             │ │  BN + ReLU                  │   │
            ▼                             │ └──────────────┬──────────────┘   │
┌───────────────────────────────┐         │                ↓                  │
│   DETECTION HEADS             │         │ ┌─────────────────────────────┐   │
│                               │         │ │  Conv 64→1 (1×1)            │   │
│ ┌─────────────────────────┐   │         │ │  (Risk prediction)          │   │
│ │  Classification Head    │   │         │ └──────────────┬──────────────┘   │
│ │  (10 classes)           │   │         │                ↓                  │
│ └───────────┬─────────────┘   │         │ ┌─────────────────────────────┐   │
│             │                 │         │ │  Bilinear Upsample          │   │
│ ┌───────────▼─────────────┐   │         │ │  50×50 → 200×200            │   │
│ │  Regression Head        │   │         │ └──────────────┬──────────────┘   │
│ │  (x,y,z,w,l,h,θ,v_x,v_y)│   │         │                ↓                  │
│ └─────────────────────────┘   │         │ ┌─────────────────────────────┐   │
│                               │         │ │  Sigmoid Activation         │   │
│ Output:                       │         │ │  (values → [0, 1])          │   │
│ - Classes: [B, 900, 10]       │         │ └──────────────┬──────────────┘   │
│ - Boxes: [B, 900, 10]         │         │                                   │
└───────────┬───────────────────┘         │ Output: [B, 1, 200, 200]          │
            │                             └────────────┬──────────────────────┘
            │                                          │
            ▼                                          ▼
┌───────────────────────────────┐         ┌───────────────────────────────────┐
│   DETECTION OUTPUTS           │         │   RISK MAP OUTPUT                 │
│                               │         │                                   │
│ • 3D Bounding Boxes           │         │ • Risk values [0, 1]              │
│ • Class Labels                │         │ • 200×200 BEV grid                │
│ • Confidence Scores           │         │ • 0.5m resolution                 │
│ • Velocity Vectors            │         │ • Range: [-50m, +50m]             │
└───────────────────────────────┘         └───────────────────────────────────┘
```

## Multi-Task Loss Function

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LOSS COMPUTATION                                │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        DETECTION LOSSES                                 │  │
│  │                                                                          │  │
│  │  L_cls   = Focal Loss (predicted_classes, gt_classes)                  │  │
│  │  L_bbox  = L1 Loss (predicted_boxes, gt_boxes)                         │  │
│  │                                                                          │  │
│  │  L_detection = L_cls + L_bbox                                           │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                         RISK LOSSES (NEW!)                              │  │
│  │                                                                          │  │
│  │  L_mse = MSE(predicted_risk, gt_risk) × focal_weight                   │  │
│  │  L_mae = MAE(predicted_risk, gt_risk) × focal_weight                   │  │
│  │                                                                          │  │
│  │  where focal_weight = 2.0 if gt_risk > 0.5 else 1.0                    │  │
│  │                                                                          │  │
│  │  L_risk = L_mse + 0.5 × L_mae                                           │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                         TOTAL LOSS                                      │  │
│  │                                                                          │  │
│  │  L_total = L_detection + λ_risk × L_risk                                │  │
│  │                                                                          │  │
│  │  where λ_risk = 1.0 (configurable risk loss weight)                    │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Risk-Guided Attention Variant

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              RISK-GUIDED ATTENTION HEAD (Optional Enhancement)               │
│                                                                               │
│  BEV Features [B, 256, 50, 50]                                               │
│         │                                                                     │
│         ├──────────────┬──────────────────────────────────────────────┐      │
│         │              │                                              │      │
│         ▼              ▼                                              ▼      │
│  ┌─────────────┐  ┌────────────────────────────┐            ┌──────────┐    │
│  │  Standard   │  │  Risk Prediction           │            │ Spatial  │    │
│  │  Conv Path  │  │  (same as before)          │            │ Attention│    │
│  └──────┬──────┘  └──────────┬─────────────────┘            │ Generator│    │
│         │                    │                               └────┬─────┘    │
│         │                    ▼                                    │          │
│         │         ┌──────────────────────┐                        │          │
│         │         │ Risk Map             │                        │          │
│         │         │ [B, 1, 200, 200]     │                        │          │
│         │         └──────────┬───────────┘                        │          │
│         │                    │                                    │          │
│         │                    │ Downsample to BEV size             │          │
│         │                    ▼                                    │          │
│         │         ┌──────────────────────┐                        │          │
│         │         │ Risk Map (small)     │────────────────────────┘          │
│         │         │ [B, 1, 50, 50]       │                                   │
│         │         └──────────┬───────────┘                                   │
│         │                    │                                               │
│         │                    ▼                                               │
│         │         ┌──────────────────────────────────┐                       │
│         │         │ Attention Weight Generation      │                       │
│         │         │ Conv(1→32→1) + Sigmoid           │                       │!1
│         │         └──────────┬───────────────────────┘                       │
│         │                    │                                               │
│         │                    │                                               │
│         │                    ▼                                               │
│         │         ┌──────────────────────┐                                   │
│         │         │ Attention Weights    │                                   │
│         │         │ [B, 1, 50, 50]       │                                   │
│         │         └──────────┬───────────┘                                   │
│         │                    │                                               │
│         └────────────────────┴──► Element-wise Multiplication                │
│                              │                                               │
│                              ▼                                               │
│                   ┌──────────────────────┐                                   │
│                   │ Risk-Attended        │                                   │
│                   │ Features             │                                   │
│                   │ [B, 256, 50, 50]     │                                   │
│                   └──────────┬───────────┘                                   │
│                              │                                               │
│                              ▼                                               │
│                   Can be used for enhanced detection                         │
│                   or as auxiliary supervision                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

## BEV Grid Coordinate System

```
                    Y (left) [meters]
                         ↑
                         │
        -50m ────────────┼────────────── +50m
          │              │              │
          │      Q2      │      Q1      │
          │              │              │
    -50m  ├──────────────┼──────────────┤  ← +50m (forward)
          │              │              │
          │      Q3      │   ✱ EGO     │
          │              │  (0, 0)      │
          │              │      ↑       │
    0m    ├──────────────┼──────────────┤
          │              │   FORWARD    │
          │      Q4      │      Q5      │
          │              │              │
    +50m  └──────────────┴──────────────┘
                         │
                         └──────────────────→ X (forward) [meters]

Risk Map Grid: 200×200 pixels
Resolution: 0.5m per pixel
Range: [-50, +50] meters in both X and Y

Pixel Coordinates:
- (0, 0) = top-left = (-50m, -50m) in world
- (100, 100) = center = (0m, 0m) = Ego vehicle
- (199, 199) = bottom-right = (+50m, +50m) in world
```

## Data Flow Example

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING DATA FLOW                                   │
│                                                                               │
│  1. nuScenes Sample Loading                                                  │
│     ├─ 6 Camera Images (CAM_FRONT, CAM_FRONT_LEFT, etc.)                    │
│     ├─ Camera Calibration (intrinsics, extrinsics)                           │
│     ├─ GT 3D Bboxes (from annotations)                                       │
│     └─ GT Risk Map (from pre-computed risk labels pkl)                       │
│            [200, 200] float32, values in [0, 1]                              │
│                                                                               │
│  2. Preprocessing                                                             │
│     ├─ Image Normalization                                                   │
│     ├─ Data Augmentation (PhotoMetric, Flip, etc.)                           │
│     ├─ BBox Transformation                                                    │
│     └─ Risk Map to Tensor                                                     │
│                                                                               │
│  3. Forward Pass                                                              │
│     ├─ Images → Backbone → Features                                          │
│     ├─ Features → BEV Transformer → BEV Features                             │
│     ├─ BEV Features → Detection Head → Pred Boxes                            │
│     └─ BEV Features → Risk Head → Pred Risk Map                              │
│                                                                               │
│  4. Loss Calculation                                                          │
│     ├─ Detection Loss = Focal Loss + L1 Loss                                 │
│     ├─ Risk Loss = MSE Loss + MAE Loss (focal weighted)                      │
│     └─ Total Loss = Detection Loss + λ × Risk Loss                           │
│                                                                               │
│  5. Backpropagation                                                           │
│     └─ Gradients flow to both detection and risk heads                       │
│                                                                               │
│  6. Optimizer Step                                                            │
│     └─ AdamW with differential learning rates:                               │
│        - Backbone: 0.1× base LR                                              │
│        - Risk Head: 1.0× base LR                                             │
│        - Other modules: 1.0× base LR                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Shape Transformation Pipeline

```
Stage                           Shape                       Notes
─────────────────────────────────────────────────────────────────────────────
Input Images                    [B, 6, 3, 900, 1600]        6 cameras
                                      ↓
Backbone (ResNet-50)            [B×6, 256, 29, 50]          Feature maps
                                      ↓
FPN Neck                        [B×6, 256, H, W]            Multi-scale
                                      ↓  (4 scales)
BEV Queries                     [2500, B, 256]              50×50 grid
                                      ↓
Temporal Self-Attention         [2500, B, 256]              With prev BEV
                                      ↓
Spatial Cross-Attention         [2500, B, 256]              From images
                                      ↓  (6 encoder layers)
BEV Features                    [B, 2500, 256]              Reshape
                                      │
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
            Detection Path                       Risk Path
                    │                                   │
          Object Queries                      Reshape to 4D
          [900, B, 256]                       [B, 256, 50, 50]
                    ↓                                   ↓
          Decoder (6 layers)                  Conv 256→128
          [900, B, 256]                       [B, 128, 50, 50]
                    ↓                                   ↓
          ┌─────────┴──────────┐              Conv 128→128
          │                    │              [B, 128, 50, 50]
    Classification        Regression               ↓
    [B, 900, 10]         [B, 900, 10]         Conv 128→64
                                               [B, 64, 50, 50]
                                                    ↓
                                               Conv 64→1
                                               [B, 1, 50, 50]
                                                    ↓
                                               Upsample 4×
                                               [B, 1, 200, 200]
                                                    ↓
                                               Sigmoid
                                               [B, 1, 200, 200]
                                               values ∈ [0, 1]
```

## Model Variants Comparison

```
┌──────────────────┬─────────────────┬────────────────────┬───────────────────┐
│ Model Variant    │ Detection Head  │ Risk Head          │ Risk Guidance     │
├──────────────────┼─────────────────┼────────────────────┼───────────────────┤
│ BEVFormer        │ ✅ Yes          │ ❌ No              │ ❌ No             │
│ (Baseline)       │                 │                    │                   │
├──────────────────┼─────────────────┼────────────────────┼───────────────────┤
│ BEVFormerRisk    │ ✅ Yes          │ ✅ Yes             │ ❌ No             │
│                  │                 │ (RiskPredictionHead│                   │
│                  │                 │  only)             │                   │
├──────────────────┼─────────────────┼────────────────────┼───────────────────┤
│ BEVFormerRisk    │ ✅ Yes          │ ✅ Yes             │ ✅ Yes            │
│ Attention        │                 │ (RiskGuidedAttn    │ (Spatial/Channel/ │
│                  │                 │  Head)             │  Both)            │
└──────────────────┴─────────────────┴────────────────────┴───────────────────┘
```

## Risk Label Generation (Pre-processing)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   RISK LABEL GENERATION (Offline Process)                    │
│                                                                               │
│  nuScenes Sample                                                              │
│         │                                                                     │
│         ├─ Ego Position, Velocity, Heading                                   │
│         ├─ GT 3D Bboxes (all objects)                                        │
│         └─ Timestamp                                                          │
│                ↓                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  For each BEV grid cell (200×200):                                    │    │
│  │                                                                        │    │
│  │  1. Calculate temporal position (TTC)                                 │    │
│  │     - Distance to cell from ego                                       │    │
│  │     - Time to reach cell at current velocity                          │    │
│  │                                                                        │    │
│  │  2. Check trajectory alignment                                        │    │
│  │     - Is cell within trajectory corridor? (±20m laterally)            │    │
│  │     - Is cell in forward direction?                                   │    │
│  │                                                                        │    │
│  │  3. Find occluding objects                                            │    │
│  │     - Objects between ego and cell                                    │    │
│  │     - Calculate occlusion factor O                                    │    │
│  │                                                                        │    │
│  │  4. Calculate risk score (V5 algorithm):                              │    │
│  │                                                                        │    │
│  │     R = I_traj × O × U × P                                            │    │
│  │                                                                        │    │
│  │     where:                                                            │    │
│  │     I_traj = 1 if on trajectory and forward, 0 otherwise              │    │
│  │     O = min(occluder_area / 10.0, 1.0)                                │    │
│  │     U = (T_safe - TTC) / (T_safe - T_critical)  [clamped to [0,1]]   │    │
│  │     P = (d_far - lateral_dist) / (d_far - d_close)  [clamped to [0,1]│    │
│  │                                                                        │    │
│  │  5. Store risk value for cell                                         │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                ↓                                                              │
│  Risk Map [200, 200] with values in [0, 1]                                   │
│         │                                                                     │
│         ↓                                                                     │
│  Save to pickle file:                                                         │
│  {                                                                            │
│    'scene_token': {                                                           │
│      'sample_token': str,                                                     │
│      'risk_map': np.ndarray [200, 200],                                       │
│      'ego_state': {...},                                                      │
│      'metadata': {                                                            │
│        'max_risk': float,                                                     │
│        'mean_risk': float,                                                    │
│        'high_risk_cells': int                                                 │
│      }                                                                         │
│    }                                                                           │
│  }                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

