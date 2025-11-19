#!/usr/bin/env python3
"""
Risk-Guided Attention Model Inference Script

Usage:
    python inference_risk_attention.py \
        --config projects/configs/bevformer/bevformer_risk_tiny_attention.py \
        --checkpoint work_dirs/bevformer_risk_attention_fixed/epoch_6.pth \
        --sample-idx 0
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with Risk-Guided Attention')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file path')
    parser.add_argument('--sample-idx', type=int, default=0, help='Sample index to visualize')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    parser.add_argument('--output-dir', default='inference_outputs', help='Output directory')
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading config from {args.config}")
    cfg = Config.fromfile(args.config)

    # Build dataset
    print("Building dataset...")
    dataset = build_dataset(cfg.data.val)

    # Build model
    print("Building model...")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # Move to device
    model = model.to(args.device)
    model.eval()

    # Get sample
    print(f"Loading sample {args.sample_idx}")
    data = dataset[args.sample_idx]

    # Prepare input
    img = data['img'].data.unsqueeze(0).to(args.device)  # [1, N_cams, 3, H, W]
    img_metas = [data['img_metas'].data]

    print("\nInput shapes:")
    print(f"  Images: {img.shape}")
    print(f"  Scene: {img_metas[0]['scene_token']}")
    print(f"  Sample: {img_metas[0]['sample_idx']}")

    # Inference
    print("\nRunning inference...")
    with torch.no_grad():
        # Forward pass
        prev_bev = None
        new_prev_bev, results = model.simple_test(
            img_metas=img_metas,
            img=img,
            prev_bev=prev_bev,
            rescale=True
        )

    # Extract results
    result = results[0]
    pts_bbox = result['pts_bbox']

    print("\nDetection Results:")
    print(f"  Boxes: {pts_bbox['boxes_3d']}")
    print(f"  Scores: {pts_bbox['scores_3d']}")
    print(f"  Labels: {pts_bbox['labels_3d']}")

    # Risk map (if available)
    if 'risk_map' in result:
        risk_map = result['risk_map']  # [1, 200, 200]
        print(f"\nRisk Map:")
        print(f"  Shape: {risk_map.shape}")
        print(f"  Range: [{risk_map.min():.4f}, {risk_map.max():.4f}]")
        print(f"  Mean: {risk_map.mean():.4f}")

        # Save risk map visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(risk_map.squeeze().cpu().numpy(), cmap='hot', vmin=0, vmax=1)
        plt.colorbar(label='Risk Score')
        plt.title(f'Predicted Risk Map - Sample {args.sample_idx}')
        plt.xlabel('X (BEV)')
        plt.ylabel('Y (BEV)')

        output_path = os.path.join(args.output_dir, f'risk_map_sample_{args.sample_idx}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nRisk map saved to: {output_path}")
        plt.close()

    # Save detection results
    output_file = os.path.join(args.output_dir, f'detection_sample_{args.sample_idx}.txt')
    with open(output_file, 'w') as f:
        f.write(f"Sample Index: {args.sample_idx}\n")
        f.write(f"Scene Token: {img_metas[0]['scene_token']}\n")
        f.write(f"Sample Token: {img_metas[0].get('sample_idx', 'N/A')}\n\n")

        f.write("Detections:\n")
        boxes = pts_bbox['boxes_3d'].tensor.cpu().numpy()
        scores = pts_bbox['scores_3d'].cpu().numpy()
        labels = pts_bbox['labels_3d'].cpu().numpy()

        class_names = dataset.CLASSES
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            f.write(f"  {i+1}. {class_names[label]}: score={score:.3f}, ")
            f.write(f"center=({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}), ")
            f.write(f"size=({box[3]:.2f}, {box[4]:.2f}, {box[5]:.2f})\n")

    print(f"Detection results saved to: {output_file}")
    print("\nInference completed!")


if __name__ == '__main__':
    main()
