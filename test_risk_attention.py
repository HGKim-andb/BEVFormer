#!/usr/bin/env python3
"""
Quick test to verify risk-guided attention mechanism works
"""

import torch
import sys
sys.path.insert(0, '.')
import projects.mmdet3d_plugin

from mmcv import Config
from mmdet.models import build_detector

def test_risk_attention():
    print("="*80)
    print("TESTING RISK-GUIDED ATTENTION MECHANISM")
    print("="*80)

    # Load config
    print("\n1. Loading config...")
    cfg = Config.fromfile('projects/configs/bevformer/bevformer_risk_tiny_attention.py')
    print(f"   ✓ Model type: {cfg.model.type}")
    print(f"   ✓ Risk head type: {cfg.model.risk_head.type}")
    print(f"   ✓ Use risk guidance: {cfg.model.use_risk_guidance}")
    print(f"   ✓ Attention type: {cfg.model.risk_head.attention_type}")

    # Build model
    print("\n2. Building model...")
    try:
        model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
        print("   ✓ Model built successfully")
    except Exception as e:
        print(f"   ✗ Error building model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check risk head type
    print("\n3. Checking risk head...")
    print(f"   Risk head class: {type(model.risk_head).__name__}")
    print(f"   Has forward_with_attention: {hasattr(model.risk_head, 'forward_with_attention')}")
    print(f"   Attention type: {model.risk_head.attention_type}")
    print(f"   Use risk guidance: {model.use_risk_guidance}")

    # Test forward pass with dummy data
    print("\n4. Testing forward pass with dummy data...")
    model.eval()

    # Create dummy BEV features
    B = 1  # Batch size
    C = 256  # Channels
    H, W = 50, 50  # BEV size

    dummy_bev = torch.randn(B, H*W, C)
    print(f"   Input BEV features: {dummy_bev.shape}")

    try:
        # Test risk prediction
        with torch.no_grad():
            risk_map = model.risk_head(dummy_bev)
        print(f"   ✓ Risk map output: {risk_map.shape}")
        print(f"   ✓ Risk map range: [{risk_map.min():.3f}, {risk_map.max():.3f}]")

        # Test attention mechanism
        if hasattr(model.risk_head, 'forward_with_attention'):
            with torch.no_grad():
                risk_map, attn_weights, attended_features = model.risk_head.forward_with_attention(dummy_bev)
            print(f"   ✓ Risk map: {risk_map.shape}")
            print(f"   ✓ Attention weights: {attn_weights.shape}")
            print(f"   ✓ Attended features: {attended_features.shape}")
            print(f"   ✓ Attention range: [{attn_weights.min():.3f}, {attn_weights.max():.3f}]")
    except Exception as e:
        print(f"   ✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED - RISK-GUIDED ATTENTION READY")
    print("="*80)
    print("\nNext steps:")
    print("  1. Start training with: CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh \\")
    print("     projects/configs/bevformer/bevformer_risk_tiny_attention.py 1")
    print("  2. Compare with baseline (no attention): bevformer_risk_tiny.py")

    return True


if __name__ == '__main__':
    success = test_risk_attention()
    sys.exit(0 if success else 1)
