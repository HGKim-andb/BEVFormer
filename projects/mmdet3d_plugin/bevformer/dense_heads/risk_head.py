"""
Risk Prediction Head for BEVFormer
Predicts risk maps from BEV features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import HEADS
from mmcv.cnn import ConvModule


@HEADS.register_module()
class RiskPredictionHead(nn.Module):
    """
    Risk Prediction Head that converts BEV features to risk maps.

    Args:
        in_channels (int): Number of input channels from BEV features. Default: 256
        bev_h (int): Height of BEV feature map. Default: 50
        bev_w (int): Width of BEV feature map. Default: 50
        risk_h (int): Height of output risk map. Default: 200
        risk_w (int): Width of output risk map. Default: 200
        num_convs (int): Number of convolutional layers. Default: 3
        conv_channels (int): Number of channels in intermediate conv layers. Default: 128
        norm_cfg (dict): Config dict for normalization layer. Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer. Default: dict(type='ReLU')
        use_sigmoid (bool): Whether to use sigmoid activation. Default: True
    """

    def __init__(self,
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
                 **kwargs):
        super(RiskPredictionHead, self).__init__()

        self.in_channels = in_channels
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.risk_h = risk_h
        self.risk_w = risk_w
        self.num_convs = num_convs
        self.conv_channels = conv_channels
        self.use_sigmoid = use_sigmoid
        self.fp16_enabled = False

        # Build convolutional layers
        self.convs = nn.ModuleList()
        in_ch = in_channels
        for i in range(num_convs):
            out_ch = conv_channels if i < num_convs - 1 else conv_channels // 2
            self.convs.append(
                ConvModule(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )
            in_ch = out_ch

        # Final prediction layer (1 channel for risk map)
        self.risk_conv = nn.Conv2d(
            conv_channels // 2,
            1,
            kernel_size=1,
            padding=0
        )

        # Upsampling to match risk map resolution
        # From [B, 1, 50, 50] to [B, 1, 200, 200]
        self.upsample_scale = risk_h // bev_h
        assert risk_h == bev_h * self.upsample_scale, \
            f"risk_h ({risk_h}) must be divisible by bev_h ({bev_h})"
        assert risk_w == bev_w * self.upsample_scale, \
            f"risk_w ({risk_w}) must be divisible by bev_w ({bev_w})"

        self._init_weights()

    def _init_weights(self):
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @auto_fp16(apply_to=('bev_features',))
    def forward(self, bev_features):
        """
        Forward function.

        Args:
            bev_features (torch.Tensor): BEV features from transformer encoder.
                Shape: [B, H*W, C] or [B, C, H, W]

        Returns:
            torch.Tensor: Predicted risk map. Shape: [B, 1, risk_h, risk_w]
        """
        # Handle different input formats
        if bev_features.dim() == 3:
            # Shape: [B, H*W, C] -> [B, C, H, W]
            B, HW, C = bev_features.shape

            # Debug: print shape if assertion fails
            if HW != self.bev_h * self.bev_w:
                print(f"WARNING: BEV shape mismatch!")
                print(f"  Input shape: {bev_features.shape}")
                print(f"  Expected HW: {self.bev_h * self.bev_w} ({self.bev_h}x{self.bev_w})")
                print(f"  Got HW: {HW}")

                # If HW=1, this might be a summary token - try to reshape differently
                if HW == 1:
                    raise ValueError(
                        f"BEV features have HW=1, which suggests wrong feature extraction. "
                        f"Expected shape: [B, {self.bev_h * self.bev_w}, C], got: {bev_features.shape}. "
                        f"Make sure to extract BEV features before pooling/aggregation."
                    )

            bev_features = bev_features.permute(0, 2, 1).reshape(B, C, self.bev_h, self.bev_w)
        elif bev_features.dim() == 4:
            # Already in [B, C, H, W] format
            B, C, H, W = bev_features.shape
            if H != self.bev_h or W != self.bev_w:
                print(f"WARNING: BEV spatial size mismatch!")
                print(f"  Input shape: {bev_features.shape}")
                print(f"  Expected: [B, C, {self.bev_h}, {self.bev_w}]")
        else:
            raise ValueError(f"Unexpected bev_features dim: {bev_features.dim()}, shape: {bev_features.shape}")

        # Apply convolutional layers
        x = bev_features
        for conv in self.convs:
            x = conv(x)

        # Predict risk map
        risk_map = self.risk_conv(x)  # [B, 1, bev_h, bev_w]

        # Upsample to target resolution
        if self.upsample_scale > 1:
            risk_map = F.interpolate(
                risk_map,
                size=(self.risk_h, self.risk_w),
                mode='bilinear',
                align_corners=False
            )

        # Apply sigmoid to get values in [0, 1]
        if self.use_sigmoid:
            risk_map = torch.sigmoid(risk_map)

        return risk_map

    @force_fp32(apply_to=('pred_risk', 'gt_risk'))
    def loss(self, pred_risk, gt_risk, valid_mask=None):
        """
        Calculate risk prediction loss.

        Args:
            pred_risk (torch.Tensor): Predicted risk map. Shape: [B, 1, H, W]
            gt_risk (torch.Tensor): Ground truth risk map. Shape: [B, 1, H, W] or [B, H, W]
            valid_mask (torch.Tensor, optional): Valid mask for risk map. Shape: [B, 1, H, W] or [B, H, W]

        Returns:
            dict: Dictionary containing loss values
        """
        # Ensure gt_risk has channel dimension
        if gt_risk.dim() == 3:
            gt_risk = gt_risk.unsqueeze(1)

        # Ensure shapes match
        assert pred_risk.shape == gt_risk.shape, \
            f"Shape mismatch: pred_risk {pred_risk.shape} vs gt_risk {gt_risk.shape}"

        # MSE Loss for continuous risk values
        mse_loss = F.mse_loss(pred_risk, gt_risk, reduction='none')

        # MAE Loss for robustness
        mae_loss = F.l1_loss(pred_risk, gt_risk, reduction='none')

        # Focal loss component for high-risk regions (risk > 0.5)
        # This helps focus on important high-risk areas
        high_risk_mask = (gt_risk > 0.5).float()
        focal_weight = torch.where(
            high_risk_mask > 0,
            torch.ones_like(gt_risk) * 2.0,  # Higher weight for high-risk regions
            torch.ones_like(gt_risk) * 1.0
        )

        # Apply valid mask if provided
        if valid_mask is not None:
            if valid_mask.dim() == 3:
                valid_mask = valid_mask.unsqueeze(1)
            mse_loss = mse_loss * valid_mask
            mae_loss = mae_loss * valid_mask
            focal_weight = focal_weight * valid_mask
            num_valid = valid_mask.sum().clamp(min=1.0)
        else:
            num_valid = pred_risk.numel()

        # Weighted losses
        weighted_mse = (mse_loss * focal_weight).sum() / num_valid
        weighted_mae = (mae_loss * focal_weight).sum() / num_valid

        # Combined loss
        total_loss = weighted_mse + 0.5 * weighted_mae

        losses = {
            'loss_risk_mse': weighted_mse,
            'loss_risk_mae': weighted_mae,
            'loss_risk': total_loss
        }

        return losses

    def get_risk_map(self, bev_features):
        """
        Convenience method to get risk map during inference.

        Args:
            bev_features (torch.Tensor): BEV features

        Returns:
            torch.Tensor: Risk map [B, 1, H, W] with values in [0, 1]
        """
        return self.forward(bev_features)


@HEADS.register_module()
class RiskGuidedAttentionHead(RiskPredictionHead):
    """
    Extended Risk Head with attention guidance capability.
    Predicts risk maps and provides risk-based attention weights.

    Additional Args:
        attention_type (str): Type of attention guidance.
            Options: 'spatial', 'channel', 'both'. Default: 'spatial'
        attention_temp (float): Temperature for attention softmax. Default: 1.0
    """

    def __init__(self,
                 attention_type='spatial',
                 attention_temp=1.0,
                 *args,
                 **kwargs):
        super(RiskGuidedAttentionHead, self).__init__(*args, **kwargs)

        self.attention_type = attention_type
        self.attention_temp = attention_temp

        # Additional layers for attention generation
        if attention_type in ['spatial', 'both']:
            self.spatial_attention_conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),
            )

        if attention_type in ['channel', 'both']:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.in_channels, self.in_channels // 16, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channels // 16, self.in_channels, kernel_size=1),
            )

    @auto_fp16(apply_to=('bev_features',))
    def forward_with_attention(self, bev_features):
        """
        Forward with attention weights.

        Args:
            bev_features (torch.Tensor): BEV features [B, H*W, C] or [B, C, H, W]

        Returns:
            tuple: (risk_map, attention_weights, attended_features)
                - risk_map: [B, 1, risk_h, risk_w]
                - attention_weights: [B, 1, bev_h, bev_w] or [B, C, 1, 1]
                - attended_features: [B, C, bev_h, bev_w]
        """
        # Get risk map
        risk_map = self.forward(bev_features)

        # Convert bev_features to 4D if needed
        if bev_features.dim() == 3:
            B, HW, C = bev_features.shape
            bev_features = bev_features.permute(0, 2, 1).reshape(B, C, self.bev_h, self.bev_w)

        # Downsample risk map to BEV resolution for attention
        risk_map_small = F.interpolate(
            risk_map,
            size=(self.bev_h, self.bev_w),
            mode='bilinear',
            align_corners=False
        )

        attention_weights = None
        attended_features = bev_features

        # Spatial attention based on risk
        if self.attention_type in ['spatial', 'both']:
            spatial_attn = self.spatial_attention_conv(risk_map_small)
            spatial_attn = torch.sigmoid(spatial_attn / self.attention_temp)
            attention_weights = spatial_attn
            attended_features = attended_features * spatial_attn

        # Channel attention based on risk
        if self.attention_type in ['channel', 'both']:
            channel_attn = self.channel_attention(attended_features)
            channel_attn = torch.sigmoid(channel_attn / self.attention_temp)
            if attention_weights is None:
                attention_weights = channel_attn
            attended_features = attended_features * channel_attn

        return risk_map, attention_weights, attended_features
