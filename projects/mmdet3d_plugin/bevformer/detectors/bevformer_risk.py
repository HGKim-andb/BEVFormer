"""
Risk-Guided BEVFormer
Extends BEVFormer with risk prediction capability
"""

import torch
import copy
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from .bevformer import BEVFormer


@DETECTORS.register_module()
class BEVFormerRisk(BEVFormer):
    """
    BEVFormer with Risk Prediction Head.

    This model extends BEVFormer to predict both 3D object detection
    and BEV risk maps simultaneously.

    Args:
        risk_head (dict): Config dict for risk prediction head
        use_risk_guidance (bool): Whether to use risk for attention guidance. Default: False
        risk_loss_weight (float): Weight for risk loss. Default: 1.0
        All other args from BEVFormer
    """

    def __init__(self,
                 risk_head=None,
                 use_risk_guidance=False,
                 risk_loss_weight=1.0,
                 *args,
                 **kwargs):
        super(BEVFormerRisk, self).__init__(*args, **kwargs)

        # Build risk head
        if risk_head is not None:
            from mmdet.models import build_head
            self.risk_head = build_head(risk_head)
        else:
            self.risk_head = None

        self.use_risk_guidance = use_risk_guidance
        self.risk_loss_weight = risk_loss_weight

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None,
                          gt_risk_maps=None):
        """
        Forward function for point cloud branch training.

        Args:
            pts_feats (list[torch.Tensor]): Features from image backbone
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth boxes
            gt_labels_3d (list[torch.Tensor]): Ground truth labels
            img_metas (list[dict]): Meta information
            gt_bboxes_ignore (list[torch.Tensor], optional): Boxes to ignore
            prev_bev (torch.Tensor, optional): Previous BEV features
            gt_risk_maps (torch.Tensor, optional): Ground truth risk maps [B, 200, 200]

        Returns:
            dict: Losses
        """
        # Forward through BEV transformer
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev)

        # Detection losses
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

        # Risk prediction losses
        if self.risk_head is not None and gt_risk_maps is not None:
            # Get BEV features from transformer output
            bev_embed = outs['bev_embed']  # Original: [H*W, B, C] from transformer

            # BEVFormer transformer returns [H*W, B, C], need to convert to [B, H*W, C]
            if bev_embed.dim() == 3 and bev_embed.shape[1] < bev_embed.shape[0]:
                # Likely [H*W, B, C] format, transpose to [B, H*W, C]
                bev_embed = bev_embed.permute(1, 0, 2)  # [H*W, B, C] -> [B, H*W, C]

            # Convert gt_risk_maps from list of DataContainer to tensor
            if isinstance(gt_risk_maps, list):
                # Extract data from DataContainer and stack
                gt_risk_maps = torch.stack([rm.data for rm in gt_risk_maps], dim=0)

            # Predict risk map
            if self.use_risk_guidance and hasattr(self.risk_head, 'forward_with_attention'):
                pred_risk_map, attention_weights, attended_features = \
                    self.risk_head.forward_with_attention(bev_embed)
            else:
                pred_risk_map = self.risk_head(bev_embed)

            # Calculate risk loss
            risk_losses = self.risk_head.loss(pred_risk_map, gt_risk_maps)

            # Weight and add risk losses
            for key, value in risk_losses.items():
                losses[key] = value * self.risk_loss_weight

        return losses

    @auto_fp16(apply_to=('img',))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      gt_risk_maps=None,
                      gt_risk_map=None,
                      **kwargs):
        """
        Forward training function.

        Args:
            gt_risk_maps (torch.Tensor, optional): Ground truth risk maps
                Shape: [B, 200, 200] or [B, 1, 200, 200]
            gt_risk_map (torch.Tensor, optional): Alternative name for gt_risk_maps
            ... (other args same as BEVFormer)

        Returns:
            dict: Losses
        """
        # Handle both gt_risk_map (from dataset) and gt_risk_maps (preferred)
        if gt_risk_maps is None and gt_risk_map is not None:
            gt_risk_maps = gt_risk_map

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None

        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()

        # Forward pts branch with risk maps
        losses_pts = self.forward_pts_train(
            img_feats,
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas,
            gt_bboxes_ignore,
            prev_bev,
            gt_risk_maps=gt_risk_maps
        )

        losses.update(losses_pts)
        return losses

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """
        Test function with risk prediction.

        Returns:
            tuple: (bev_embed, bbox_results, risk_map)
        """
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        # Predict risk map if risk head exists
        risk_map = None
        if self.risk_head is not None:
            bev_embed = outs['bev_embed']
            risk_map = self.risk_head(bev_embed)  # [B, 1, 200, 200]

        return outs['bev_embed'], bbox_results, risk_map

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """
        Test function without augmentation.

        Returns:
            tuple: (new_prev_bev, bbox_list)
                Each element in bbox_list contains 'pts_bbox' and optionally 'risk_map'
        """
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts, risk_map = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        # Add risk map to results if available
        if risk_map is not None:
            for i, result_dict in enumerate(bbox_list):
                result_dict['risk_map'] = risk_map[i]  # [1, 200, 200]

        return new_prev_bev, bbox_list


@DETECTORS.register_module()
class BEVFormerRiskAttention(BEVFormerRisk):
    """
    BEVFormer with Risk-Guided Attention.

    This model uses predicted risk maps to guide the attention mechanism
    in the BEV transformer.

    Note: This requires modifications to the transformer layers to accept
    attention guidance, which is handled separately.
    """

    def __init__(self, *args, **kwargs):
        # Force use_risk_guidance to True
        kwargs['use_risk_guidance'] = True
        super(BEVFormerRiskAttention, self).__init__(*args, **kwargs)

        # Ensure risk head supports attention
        if self.risk_head is not None:
            assert hasattr(self.risk_head, 'forward_with_attention'), \
                "Risk head must support forward_with_attention method"

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None,
                          gt_risk_maps=None):
        """
        Forward with risk-guided attention.
        """
        # First pass: Get initial BEV features and risk prediction
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev)
        bev_embed = outs['bev_embed']

        # Generate risk-based attention
        if self.risk_head is not None:
            pred_risk_map, attention_weights, attended_features = \
                self.risk_head.forward_with_attention(bev_embed)

            # TODO: Use attended_features in subsequent processing
            # This would require modifying the transformer decoder
            # For now, we just compute the risk loss

        # Detection losses
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

        # Risk losses
        if self.risk_head is not None and gt_risk_maps is not None:
            risk_losses = self.risk_head.loss(pred_risk_map, gt_risk_maps)
            for key, value in risk_losses.items():
                losses[key] = value * self.risk_loss_weight

        return losses
