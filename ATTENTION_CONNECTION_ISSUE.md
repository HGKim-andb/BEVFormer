# Attention과 Detection Head 연결 문제

## 🔍 발견된 문제

현재 구현에서 **attended_features가 생성되지만 detection에 사용되지 않습니다!**

### 현재 코드:

```python
# bevformer_risk.py, line 70-106

# 1. Detection forward (original BEV)
outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev)

# 2. Detection loss 계산 (original BEV 사용)
losses = self.pts_bbox_head.loss([gt_bboxes_3d, gt_labels_3d, outs], ...)

# 3. Attention 생성
if self.use_risk_guidance:
    pred_risk_map, attention_weights, attended_features = \
        self.risk_head.forward_with_attention(bev_embed)

    # ❌ attended_features가 여기서 버려짐!

# 4. Risk loss 계산
risk_losses = self.risk_head.loss(pred_risk_map, gt_risk_maps)
```

## 📊 현재 데이터 흐름:

```
Images
  ↓
BEV Transformer
  ↓
BEV Features
  ├─→ Detection Head → Boxes (original BEV 사용)
  │
  └─→ Risk Head
       ├─→ Risk Map
       └─→ Attention Weights → attended_features
                                      ↓
                                  (버려짐! ❌)
```

## 🎯 올바른 흐름 (목표):

```
Images
  ↓
BEV Transformer
  ↓
BEV Features
  ↓
Risk Head
  ├─→ Risk Map
  └─→ Attention Weights
       ↓
BEV × Attention = Attended BEV
       ↓
Detection Head → Boxes (attended BEV 사용! ✅)
```

## 🔧 해결 방법

### 방법 1: Detection Forward 수정 (추천)

```python
def forward_pts_train(self, pts_feats, gt_bboxes_3d, gt_labels_3d,
                      img_metas, gt_bboxes_ignore=None, prev_bev=None,
                      gt_risk_maps=None):

    # Step 1: BEV transformer forward
    outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev)
    bev_embed = outs['bev_embed']

    # Step 2: Risk-guided attention
    if self.use_risk_guidance and self.risk_head is not None:
        # 형상 변환
        if bev_embed.dim() == 3 and bev_embed.shape[1] < bev_embed.shape[0]:
            bev_embed = bev_embed.permute(1, 0, 2)

        # Attention 생성
        pred_risk_map, attention_weights, attended_bev = \
            self.risk_head.forward_with_attention(bev_embed)

        # ⭐ 핵심: Attended BEV를 다시 올바른 형상으로
        # [B, 256, 50, 50] → [2500, B, 256]
        B, C, H, W = attended_bev.shape
        attended_bev = attended_bev.view(B, C, H*W).permute(2, 0, 1)

        # ⭐ Detection head forward with attended BEV
        # BEVFormer head의 transformer decoder에 attended_bev 전달
        outs_attended = self.pts_bbox_head.forward_decoder(
            attended_bev, img_metas
        )

        # Detection loss with attended features
        losses = self.pts_bbox_head.loss(
            [gt_bboxes_3d, gt_labels_3d, outs_attended],
            img_metas=img_metas
        )

        # Risk loss
        if gt_risk_maps is not None:
            gt_risk_maps = torch.stack([rm.data for rm in gt_risk_maps], dim=0)
            risk_losses = self.risk_head.loss(pred_risk_map, gt_risk_maps)
            for key, value in risk_losses.items():
                losses[key] = value * self.risk_loss_weight
    else:
        # No attention: use original
        losses = self.pts_bbox_head.loss(
            [gt_bboxes_3d, gt_labels_3d, outs],
            img_metas=img_metas
        )

        # Risk loss (no attention)
        if self.risk_head is not None and gt_risk_maps is not None:
            # ... (기존 코드)

    return losses
```

### 방법 2: Detection Head 내부 수정

BEVFormer detection head가 attended BEV를 받도록 수정:

```python
# projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py

class BEVFormerHead(nn.Module):
    def forward(self, mlvl_feats, img_metas, prev_bev=None,
                attended_bev=None):  # ← 파라미터 추가

        if attended_bev is not None:
            # Attended BEV 사용
            bev_embed = attended_bev
        else:
            # Original transformer forward
            bev_queries = self.bev_embedding.weight.to(dtype)
            bev_embed = self.transformer(
                mlvl_feats,
                bev_queries,
                prev_bev=prev_bev,
                **kwargs
            )

        # Decoder는 동일
        outs = self.transformer.decoder(
            query=object_query_embeds,
            key=bev_embed,  # Attended or original
            ...
        )
        return outs
```

### 방법 3: 현재 상태 유지 (간접 효과)

Attention을 직접 사용하지 않고, loss를 통한 간접 학습:

```python
# 현재 구현 그대로
# - BEV features가 detection과 risk 모두에 사용
# - Multi-task learning으로 BEV가 두 task 모두 잘하도록 학습
# - Attention weights는 risk loss에만 영향

장점:
  - 코드 수정 최소
  - 안정적 학습

단점:
  - Attention의 직접적 효과 없음
  - "Risk-guided" 이름과 불일치
```

## 📝 권장 사항

### 단기 (현재):
**방법 3** 사용 - 현재 코드 그대로 학습
- Multi-task learning 효과 확인
- Baseline 비교
- 개념 증명

### 중기 (다음):
**방법 1** 구현 - Detection에 attended BEV 직접 사용
- 진정한 risk-guided attention
- 더 강한 효과 기대
- Ablation study 가능:
  - No risk head
  - + Risk head (no attention to detection)
  - + Risk-guided attention (attended BEV to detection)

## 🎯 실험 계획

### Experiment 1: 현재 구현 (간접)
```
Baseline: Detection only
+ Risk prediction (no attention)
+ Risk with attention (indirect via loss)
```

### Experiment 2: 수정 구현 (직접)
```
Baseline: Detection only
+ Risk prediction (no attention)
+ Risk-guided attention (direct to detection) ← 새로운!
```

### 기대 결과:
```
Experiment 1 (간접):
  mAP 향상: +0.5% ~ +1.5%
  (Multi-task learning 효과)

Experiment 2 (직접):
  mAP 향상: +1.5% ~ +3.0%
  (Attention 직접 효과 + Multi-task)
```

## 💡 다음 단계

### Option A: 현재 코드로 실험
```bash
1. 현재 구현으로 학습
2. Baseline 비교
3. 결과 분석
4. 논문: "Multi-task learning with risk prediction"
```

### Option B: 수정 후 실험
```bash
1. 방법 1 구현 (attended BEV to detection)
2. 학습 및 비교
3. Ablation study
4. 논문: "Risk-guided attention for detection"
```

### Option C: 둘 다!
```bash
1. 현재 구현 실험 (빠름)
2. 수정 구현 (시간 소요)
3. 두 방법 비교
4. 논문: 포괄적 분석
```

## 🔍 현재 코드의 의미

현재 구현은:
- ❌ "Risk-guided attention" (직접)은 아님
- ✅ "Multi-task learning with attention" (간접)임

Attention weights는:
- Risk prediction에만 직접 영향
- Detection에는 gradient를 통한 간접 영향

## 📊 성능 예측

### Multi-task learning (현재):
```
BEV features 개선:
  Detection task ←─┐
                   ├─→ BEV ← 두 task 공유
  Risk task ←──────┘

결과: +0.5% ~ +1.5% mAP
```

### Risk-guided attention (수정 후):
```
BEV features 개선:
  Detection task
       ↑
  Attention weights (위험 영역 강조)
       ↑
  Risk task

결과: +1.5% ~ +3.0% mAP
```

## 🎓 결론

**현재 구현의 문제**:
- Attended features가 생성되지만 사용되지 않음
- Detection은 original BEV만 사용
- Attention의 직접 효과 없음

**해결 방법**:
1. 간단: 현재 상태 유지 (간접 효과)
2. 중간: Detection forward에 attended BEV 전달
3. 복잡: Detection head 내부 수정

**추천**:
- 현재 코드로 먼저 실험 → 빠른 결과
- 나중에 수정 구현 → 더 강한 효과
- 둘 다 비교 → 완전한 분석
