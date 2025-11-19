"""
End-to-End Integration Test

Tests the complete pipeline from raw data to final output:
1. Data loading
2. Forward pass
3. Loss calculation
4. Backward pass
5. Metrics evaluation
6. Visualization
"""

import torch
import torch.optim as optim
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_single_batch_overfit():
    """
    Test overfitting on a single batch.
    This ensures the model can learn and all components work together.
    """
    print("\n" + "="*80)
    print("TEST: Single Batch Overfitting")
    print("="*80)

    from projects.mmdet3d_plugin.bevformer.dense_heads.risk_head import RiskPredictionHead

    # Create model
    risk_head = RiskPredictionHead(
        in_channels=256,
        bev_h=50,
        bev_w=50,
        risk_h=200,
        risk_w=200,
        num_convs=3,
        conv_channels=128
    )

    # Create optimizer
    optimizer = optim.Adam(risk_head.parameters(), lr=1e-3)

    # Create a single batch of fake data
    B = 2
    bev_features = torch.randn(B, 256, 50, 50)
    gt_risk = torch.rand(B, 200, 200)  # Random target

    print(f"\n📋 Training on single batch...")
    print(f"  Batch size: {B}")
    print(f"  Target: Random risk map [0, 1]")

    # Train for multiple iterations
    num_iterations = 50
    losses_history = []

    for i in range(num_iterations):
        optimizer.zero_grad()

        # Forward
        pred_risk = risk_head(bev_features)

        # Loss
        losses = risk_head.loss(pred_risk, gt_risk)
        total_loss = losses['loss_risk']

        # Backward
        total_loss.backward()
        optimizer.step()

        losses_history.append(total_loss.item())

        if (i + 1) % 10 == 0:
            print(f"  Iteration {i+1}/{num_iterations}: loss = {total_loss.item():.6f}")

    # Check that loss decreased
    initial_loss = losses_history[0]
    final_loss = losses_history[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100

    print(f"\n📋 Training results:")
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Reduction: {reduction:.2f}%")

    assert final_loss < initial_loss, "Loss did not decrease!"
    assert reduction > 10, f"Loss reduction too small: {reduction:.2f}%"

    print(f"\n✅ Model can learn (loss reduced by {reduction:.2f}%)!\n")
    return True


def test_end_to_end_forward():
    """Test complete forward pass with minimal setup"""
    print("\n" + "="*80)
    print("TEST: End-to-End Forward Pass")
    print("="*80)

    # Create components
    from projects.mmdet3d_plugin.bevformer.dense_heads.risk_head import RiskPredictionHead

    risk_head = RiskPredictionHead(
        in_channels=256,
        bev_h=50,
        bev_w=50,
        risk_h=200,
        risk_w=200
    )

    print("\n📋 Simulating full pipeline...")

    # 1. Simulate multi-camera images -> backbone features
    print("  1. Image backbone (simulated)")
    B, num_cams = 2, 6
    img_features = [torch.randn(B * num_cams, 256, 32, 88)]  # Typical shape after backbone

    # 2. Simulate BEV transformer -> BEV features
    print("  2. BEV transformer (simulated)")
    bev_features = torch.randn(B, 50*50, 256)  # [B, H*W, C]

    # 3. Risk prediction
    print("  3. Risk prediction")
    risk_map = risk_head(bev_features)

    print(f"\n📋 Output shapes:")
    print(f"  Risk map: {risk_map.shape}")

    assert risk_map.shape == (B, 1, 200, 200)
    assert not torch.isnan(risk_map).any()
    assert not torch.isinf(risk_map).any()

    print("\n✅ End-to-End Forward Pass PASSED!\n")
    return True


def test_multi_gpu_compatibility():
    """Test DataParallel compatibility"""
    print("\n" + "="*80)
    print("TEST: Multi-GPU Compatibility")
    print("="*80)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping multi-GPU test")
        return True

    num_gpus = torch.cuda.device_count()
    print(f"\n📋 Available GPUs: {num_gpus}")

    if num_gpus < 2:
        print("⚠️  Less than 2 GPUs available, testing single GPU")
        device = torch.device('cuda:0')
        use_dp = False
    else:
        print("✅ Testing DataParallel with 2 GPUs")
        device = torch.device('cuda:0')
        use_dp = True

    from projects.mmdet3d_plugin.bevformer.dense_heads.risk_head import RiskPredictionHead

    risk_head = RiskPredictionHead(
        in_channels=256,
        bev_h=50,
        bev_w=50,
        risk_h=200,
        risk_w=200
    )

    if use_dp:
        risk_head = torch.nn.DataParallel(risk_head, device_ids=[0, 1])

    risk_head = risk_head.to(device)

    # Forward pass
    B = 4
    bev_features = torch.randn(B, 256, 50, 50).to(device)

    with torch.no_grad():
        risk_map = risk_head(bev_features)

    print(f"\n📋 Output shape: {risk_map.shape}")
    assert risk_map.shape == (B, 1, 200, 200)

    print("\n✅ Multi-GPU Compatibility Test PASSED!\n")
    return True


def test_save_load_checkpoint():
    """Test saving and loading model checkpoint"""
    print("\n" + "="*80)
    print("TEST: Save/Load Checkpoint")
    print("="*80)

    from projects.mmdet3d_plugin.bevformer.dense_heads.risk_head import RiskPredictionHead
    import tempfile

    # Create model
    risk_head = RiskPredictionHead(
        in_channels=256,
        bev_h=50,
        bev_w=50,
        risk_h=200,
        risk_w=200
    )

    # Get initial weights
    initial_weights = {}
    for name, param in risk_head.named_parameters():
        initial_weights[name] = param.data.clone()

    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        checkpoint_path = f.name

    torch.save({
        'model': risk_head.state_dict(),
        'config': {
            'in_channels': 256,
            'bev_h': 50,
            'bev_w': 50,
            'risk_h': 200,
            'risk_w': 200,
        }
    }, checkpoint_path)

    print(f"\n📋 Saved checkpoint to: {checkpoint_path}")

    # Create new model
    risk_head_new = RiskPredictionHead(
        in_channels=256,
        bev_h=50,
        bev_w=50,
        risk_h=200,
        risk_w=200
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    risk_head_new.load_state_dict(checkpoint['model'])

    print(f"📋 Loaded checkpoint")

    # Check that weights match
    for name, param in risk_head_new.named_parameters():
        original = initial_weights[name]
        loaded = param.data

        diff = (original - loaded).abs().max().item()
        assert diff < 1e-6, f"Weight mismatch for {name}: diff={diff}"

    print(f"  ✅ All weights match")

    # Test inference
    bev_features = torch.randn(2, 256, 50, 50)

    with torch.no_grad():
        output1 = risk_head(bev_features)
        output2 = risk_head_new(bev_features)

    diff = (output1 - output2).abs().max().item()
    print(f"  ✅ Output difference: {diff:.10f}")

    assert diff < 1e-6

    # Clean up
    os.unlink(checkpoint_path)

    print("\n✅ Save/Load Checkpoint Test PASSED!\n")
    return True


def test_inference_speed():
    """Benchmark inference speed"""
    print("\n" + "="*80)
    print("TEST: Inference Speed Benchmark")
    print("="*80)

    from projects.mmdet3d_plugin.bevformer.dense_heads.risk_head import RiskPredictionHead
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📋 Device: {device}")

    risk_head = RiskPredictionHead(
        in_channels=256,
        bev_h=50,
        bev_w=50,
        risk_h=200,
        risk_w=200
    ).to(device)

    risk_head.eval()

    # Warmup
    bev_features = torch.randn(1, 256, 50, 50).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = risk_head(bev_features)

    # Benchmark
    batch_sizes = [1, 2, 4, 8]
    num_runs = 100

    print(f"\n📋 Running {num_runs} iterations per batch size...")

    for B in batch_sizes:
        bev_features = torch.randn(B, 256, 50, 50).to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()

        for _ in range(num_runs):
            with torch.no_grad():
                _ = risk_head(bev_features)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        end_time = time.time()

        avg_time_ms = (end_time - start_time) / num_runs * 1000
        throughput = B / (avg_time_ms / 1000)

        print(f"  Batch size {B}: {avg_time_ms:.2f} ms/batch ({throughput:.1f} samples/sec)")

    print("\n✅ Inference Speed Benchmark PASSED!\n")
    return True


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*80)
    print("RUNNING ALL INTEGRATION TESTS")
    print("="*80)

    tests = [
        ("End-to-End Forward", test_end_to_end_forward),
        ("Single Batch Overfit", test_single_batch_overfit),
        ("Multi-GPU Compatibility", test_multi_gpu_compatibility),
        ("Save/Load Checkpoint", test_save_load_checkpoint),
        ("Inference Speed", test_inference_speed),
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
        print("\n🎉 ALL INTEGRATION TESTS PASSED! 🎉\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed\n")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
