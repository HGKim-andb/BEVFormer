"""
Model Architecture Validation Tests

Tests to ensure all components work together correctly:
- Risk head forward pass
- BEV feature shape consistency
- Multi-task output format
- Gradient flow
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_risk_head_shape():
    """Test risk head input/output shapes"""
    print("\n" + "="*80)
    print("TEST: Risk Head Shape Validation")
    print("="*80)

    from projects.mmdet3d_plugin.bevformer.dense_heads.risk_head import RiskPredictionHead

    # Create risk head
    risk_head = RiskPredictionHead(
        in_channels=256,
        bev_h=50,
        bev_w=50,
        risk_h=200,
        risk_w=200,
        num_convs=3,
        conv_channels=128
    )

    # Test with 3D input [B, H*W, C]
    print("\n📋 Test 1: 3D BEV features [B, H*W, C]")
    B, H, W, C = 2, 50, 50, 256
    bev_features_3d = torch.randn(B, H*W, C)
    print(f"  Input shape: {bev_features_3d.shape}")

    risk_map = risk_head(bev_features_3d)
    print(f"  Output shape: {risk_map.shape}")
    print(f"  Expected: ({B}, 1, 200, 200)")

    assert risk_map.shape == (B, 1, 200, 200), \
        f"Shape mismatch: {risk_map.shape} != ({B}, 1, 200, 200)"
    assert risk_map.min() >= 0 and risk_map.max() <= 1, \
        f"Risk values out of range [0, 1]: [{risk_map.min():.3f}, {risk_map.max():.3f}]"

    print(f"  ✅ Shape correct!")
    print(f"  ✅ Values in range [0, 1]: [{risk_map.min():.3f}, {risk_map.max():.3f}]")

    # Test with 4D input [B, C, H, W]
    print("\n📋 Test 2: 4D BEV features [B, C, H, W]")
    bev_features_4d = torch.randn(B, C, H, W)
    print(f"  Input shape: {bev_features_4d.shape}")

    risk_map = risk_head(bev_features_4d)
    print(f"  Output shape: {risk_map.shape}")

    assert risk_map.shape == (B, 1, 200, 200), \
        f"Shape mismatch: {risk_map.shape} != ({B}, 1, 200, 200)"

    print(f"  ✅ Shape correct!")

    print("\n✅ Risk Head Shape Test PASSED!\n")
    return True


def test_risk_head_loss():
    """Test risk head loss calculation"""
    print("\n" + "="*80)
    print("TEST: Risk Head Loss Calculation")
    print("="*80)

    from projects.mmdet3d_plugin.bevformer.dense_heads.risk_head import RiskPredictionHead

    risk_head = RiskPredictionHead(
        in_channels=256,
        bev_h=50,
        bev_w=50,
        risk_h=200,
        risk_w=200
    )

    B = 2
    bev_features = torch.randn(B, 256, 50, 50)
    gt_risk = torch.rand(B, 200, 200)  # Random GT in [0, 1]

    # Forward
    pred_risk = risk_head(bev_features)
    print(f"\n📋 Predicted risk shape: {pred_risk.shape}")
    print(f"📋 GT risk shape: {gt_risk.shape}")

    # Calculate loss
    losses = risk_head.loss(pred_risk, gt_risk)

    print(f"\n📋 Loss components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")
        assert not torch.isnan(value), f"NaN detected in {key}"
        assert not torch.isinf(value), f"Inf detected in {key}"
        assert value >= 0, f"Negative loss in {key}: {value.item()}"

    print("\n✅ Risk Head Loss Test PASSED!\n")
    return True


def test_risk_guided_attention():
    """Test risk-guided attention head"""
    print("\n" + "="*80)
    print("TEST: Risk-Guided Attention Head")
    print("="*80)

    from projects.mmdet3d_plugin.bevformer.dense_heads.risk_head import RiskGuidedAttentionHead

    risk_head = RiskGuidedAttentionHead(
        in_channels=256,
        bev_h=50,
        bev_w=50,
        risk_h=200,
        risk_w=200,
        attention_type='spatial'
    )

    B, C, H, W = 2, 256, 50, 50
    bev_features = torch.randn(B, C, H, W)

    print(f"\n📋 Input BEV features: {bev_features.shape}")

    # Forward with attention
    risk_map, attention_weights, attended_features = risk_head.forward_with_attention(bev_features)

    print(f"\n📋 Outputs:")
    print(f"  Risk map: {risk_map.shape}")
    print(f"  Attention weights: {attention_weights.shape if attention_weights is not None else None}")
    print(f"  Attended features: {attended_features.shape}")

    assert risk_map.shape == (B, 1, 200, 200), f"Risk map shape mismatch"
    assert attended_features.shape == (B, C, H, W), f"Attended features shape mismatch"

    print("\n✅ Risk-Guided Attention Test PASSED!\n")
    return True


def test_gradient_flow():
    """Test that gradients flow through all components"""
    print("\n" + "="*80)
    print("TEST: Gradient Flow")
    print("="*80)

    from projects.mmdet3d_plugin.bevformer.dense_heads.risk_head import RiskPredictionHead

    risk_head = RiskPredictionHead(
        in_channels=256,
        bev_h=50,
        bev_w=50,
        risk_h=200,
        risk_w=200
    )

    B = 2
    bev_features = torch.randn(B, 256, 50, 50, requires_grad=True)
    gt_risk = torch.rand(B, 200, 200)

    # Forward
    pred_risk = risk_head(bev_features)
    losses = risk_head.loss(pred_risk, gt_risk)
    total_loss = losses['loss_risk']

    # Backward
    total_loss.backward()

    print(f"\n📋 Checking gradients...")

    # Check input gradient
    assert bev_features.grad is not None, "No gradient for input features"
    assert not torch.isnan(bev_features.grad).any(), "NaN in input gradient"
    assert not torch.isinf(bev_features.grad).any(), "Inf in input gradient"
    print(f"  ✅ Input gradient: OK (norm={bev_features.grad.norm().item():.6f})")

    # Check parameter gradients
    params_with_grad = 0
    params_without_grad = 0
    nan_gradients = []
    inf_gradients = []

    for name, param in risk_head.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                params_without_grad += 1
                print(f"  ⚠️  No gradient: {name}")
            elif torch.isnan(param.grad).any():
                nan_gradients.append(name)
                print(f"  ❌ NaN gradient: {name}")
            elif torch.isinf(param.grad).any():
                inf_gradients.append(name)
                print(f"  ❌ Inf gradient: {name}")
            else:
                params_with_grad += 1

    print(f"\n📋 Gradient statistics:")
    print(f"  Parameters with gradient: {params_with_grad}")
    print(f"  Parameters without gradient: {params_without_grad}")
    print(f"  Parameters with NaN gradient: {len(nan_gradients)}")
    print(f"  Parameters with Inf gradient: {len(inf_gradients)}")

    assert len(nan_gradients) == 0, f"NaN gradients detected in {nan_gradients}"
    assert len(inf_gradients) == 0, f"Inf gradients detected in {inf_gradients}"
    assert params_without_grad == 0, f"Some parameters have no gradient"

    print("\n✅ Gradient Flow Test PASSED!\n")
    return True


def test_memory_usage():
    """Test memory usage for different batch sizes"""
    print("\n" + "="*80)
    print("TEST: Memory Usage")
    print("="*80)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping memory test")
        return True

    from projects.mmdet3d_plugin.bevformer.dense_heads.risk_head import RiskPredictionHead

    risk_head = RiskPredictionHead(
        in_channels=256,
        bev_h=50,
        bev_w=50,
        risk_h=200,
        risk_w=200
    ).cuda()

    batch_sizes = [1, 2, 4, 8]
    memory_usage = []

    for B in batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        bev_features = torch.randn(B, 256, 50, 50).cuda()
        gt_risk = torch.rand(B, 200, 200).cuda()

        # Forward + backward
        pred_risk = risk_head(bev_features)
        losses = risk_head.loss(pred_risk, gt_risk)
        losses['loss_risk'].backward()

        mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        memory_usage.append(mem_mb)

        print(f"  Batch size {B}: {mem_mb:.2f} MB")

    print(f"\n📋 Memory scaling:")
    for i in range(1, len(batch_sizes)):
        scaling = memory_usage[i] / memory_usage[i-1]
        print(f"  {batch_sizes[i-1]} -> {batch_sizes[i]}: {scaling:.2f}x")

    print("\n✅ Memory Usage Test PASSED!\n")
    return True


def test_deterministic_output():
    """Test that same input produces same output (with fixed seed)"""
    print("\n" + "="*80)
    print("TEST: Deterministic Output")
    print("="*80)

    from projects.mmdet3d_plugin.bevformer.dense_heads.risk_head import RiskPredictionHead

    def set_seed(seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    risk_head = RiskPredictionHead(
        in_channels=256,
        bev_h=50,
        bev_w=50,
        risk_h=200,
        risk_w=200
    )
    risk_head.eval()  # Evaluation mode

    B = 2
    bev_features = torch.randn(B, 256, 50, 50)

    # Run 1
    set_seed(42)
    with torch.no_grad():
        output1 = risk_head(bev_features)

    # Run 2
    set_seed(42)
    with torch.no_grad():
        output2 = risk_head(bev_features)

    # Check if outputs are the same
    diff = (output1 - output2).abs().max().item()
    print(f"\n📋 Max difference between runs: {diff:.10f}")

    assert diff < 1e-6, f"Non-deterministic output detected: diff={diff}"

    print("✅ Deterministic Output Test PASSED!\n")
    return True


def run_all_tests():
    """Run all model tests"""
    print("\n" + "="*80)
    print("RUNNING ALL MODEL TESTS")
    print("="*80)

    tests = [
        ("Risk Head Shape", test_risk_head_shape),
        ("Risk Head Loss", test_risk_head_loss),
        ("Risk-Guided Attention", test_risk_guided_attention),
        ("Gradient Flow", test_gradient_flow),
        ("Memory Usage", test_memory_usage),
        ("Deterministic Output", test_deterministic_output),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASSED" if result else "FAILED"
        except Exception as e:
            results[test_name] = f"ERROR: {str(e)}"
            print(f"\n❌ {test_name} failed with error:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for r in results.values() if r == "PASSED")
    total = len(results)

    for test_name, result in results.items():
        status_icon = "✅" if result == "PASSED" else "❌"
        print(f"{status_icon} {test_name}: {result}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed\n")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
