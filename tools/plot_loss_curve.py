#!/usr/bin/env python3
"""
Loss Curve Visualization Script

Usage:
    python tools/plot_loss_curve.py --log-dir work_dirs/risk_w500_12ep
    python tools/plot_loss_curve.py --log-json work_dirs/risk_w500_12ep/20251201_XXXXXX.log.json
"""

import argparse
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Plot training loss curves')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory containing log files')
    parser.add_argument('--log-json', type=str, default=None,
                        help='Specific JSON log file path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path (default: loss_curve.png in log dir)')
    parser.add_argument('--smooth', type=float, default=0.9,
                        help='Smoothing factor (0-1, higher = smoother)')
    parser.add_argument('--figsize', type=int, nargs=2, default=[16, 12],
                        help='Figure size (width, height)')
    return parser.parse_args()


def smooth_curve(values, weight=0.9):
    """Apply exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def load_log_json(log_path):
    """Load and parse JSON log file."""
    logs = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get('mode') == 'train':
                    logs.append(data)
            except json.JSONDecodeError:
                continue
    return logs


def find_log_json(log_dir):
    """Find the most recent JSON log file in directory."""
    pattern = os.path.join(log_dir, '*.log.json')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No .log.json files found in {log_dir}")
    # Return most recent
    return max(files, key=os.path.getmtime)


def plot_loss_curves(logs, output_path, smooth_weight=0.9, figsize=(16, 12)):
    """Plot comprehensive loss curves."""

    # Extract data
    iterations = []
    epochs = []

    # Loss categories
    total_loss = []
    risk_loss = []
    risk_mse = []
    risk_mae = []
    detection_loss = []  # Sum of cls + bbox losses
    grad_norm = []
    lr = []

    for log in logs:
        iter_num = (log['epoch'] - 1) * 1000 + log['iter']  # Approximate
        iterations.append(iter_num)
        epochs.append(log['epoch'])

        total_loss.append(log.get('loss', 0))
        risk_loss.append(log.get('loss_risk', 0))
        risk_mse.append(log.get('loss_risk_mse', 0))
        risk_mae.append(log.get('loss_risk_mae', 0))
        grad_norm.append(log.get('grad_norm', 0))
        lr.append(log.get('lr', 0))

        # Calculate detection loss (main head losses)
        det_loss = log.get('loss_cls', 0) + log.get('loss_bbox', 0)
        # Add decoder layer losses
        for i in range(6):
            det_loss += log.get(f'd{i}.loss_cls', 0) + log.get(f'd{i}.loss_bbox', 0)
        detection_loss.append(det_loss)

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle('Training Loss Curves (12 Epochs)', fontsize=16, fontweight='bold')

    # 1. Total Loss
    ax = axes[0, 0]
    ax.plot(iterations, total_loss, alpha=0.3, color='blue', label='Raw')
    ax.plot(iterations, smooth_curve(total_loss, smooth_weight),
            color='blue', linewidth=2, label='Smoothed')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 2. Risk Loss
    ax = axes[0, 1]
    ax.plot(iterations, risk_loss, alpha=0.3, color='red', label='Raw')
    ax.plot(iterations, smooth_curve(risk_loss, smooth_weight),
            color='red', linewidth=2, label='Smoothed')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Risk Loss')
    ax.set_title('Risk Loss (weighted)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Risk MSE & MAE
    ax = axes[1, 0]
    ax.plot(iterations, risk_mse, alpha=0.3, color='orange', label='MSE (raw)')
    ax.plot(iterations, smooth_curve(risk_mse, smooth_weight),
            color='orange', linewidth=2, label='MSE (smoothed)')
    ax.plot(iterations, risk_mae, alpha=0.3, color='green', label='MAE (raw)')
    ax.plot(iterations, smooth_curve(risk_mae, smooth_weight),
            color='green', linewidth=2, label='MAE (smoothed)')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.set_title('Risk MSE & MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Detection Loss
    ax = axes[1, 1]
    ax.plot(iterations, detection_loss, alpha=0.3, color='purple', label='Raw')
    ax.plot(iterations, smooth_curve(detection_loss, smooth_weight),
            color='purple', linewidth=2, label='Smoothed')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Detection Loss')
    ax.set_title('Detection Loss (cls + bbox)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Gradient Norm
    ax = axes[2, 0]
    ax.plot(iterations, grad_norm, alpha=0.3, color='brown', label='Raw')
    ax.plot(iterations, smooth_curve(grad_norm, smooth_weight),
            color='brown', linewidth=2, label='Smoothed')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Learning Rate
    ax = axes[2, 1]
    ax.plot(iterations, lr, color='teal', linewidth=2)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)

    # Add epoch markers
    for ax_row in axes:
        for ax in ax_row:
            # Add vertical lines for epoch boundaries
            epoch_changes = []
            prev_epoch = 0
            for i, e in enumerate(epochs):
                if e != prev_epoch:
                    epoch_changes.append(iterations[i])
                    prev_epoch = e
            for ec in epoch_changes:
                ax.axvline(x=ec, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Loss curve saved to: {output_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Total iterations: {len(logs)}")
    print(f"Epochs: {min(epochs)} - {max(epochs)}")
    print(f"\nFinal Loss Values (last 100 iterations average):")
    print(f"  Total Loss: {np.mean(total_loss[-100:]):.4f}")
    print(f"  Risk Loss:  {np.mean(risk_loss[-100:]):.4f}")
    print(f"  Risk MSE:   {np.mean(risk_mse[-100:]):.6f}")
    print(f"  Risk MAE:   {np.mean(risk_mae[-100:]):.6f}")
    print(f"  Det Loss:   {np.mean(detection_loss[-100:]):.4f}")
    print(f"  Grad Norm:  {np.mean(grad_norm[-100:]):.2f}")

    print(f"\nImprovement (first 100 vs last 100 iterations):")
    if len(total_loss) > 200:
        print(f"  Total Loss: {np.mean(total_loss[:100]):.4f} -> {np.mean(total_loss[-100:]):.4f} "
              f"({(1 - np.mean(total_loss[-100:])/np.mean(total_loss[:100]))*100:.1f}% reduction)")
        print(f"  Risk Loss:  {np.mean(risk_loss[:100]):.4f} -> {np.mean(risk_loss[-100:]):.4f} "
              f"({(1 - np.mean(risk_loss[-100:])/np.mean(risk_loss[:100]))*100:.1f}% reduction)")


def main():
    args = parse_args()

    # Find log file
    if args.log_json:
        log_path = args.log_json
    elif args.log_dir:
        log_path = find_log_json(args.log_dir)
    else:
        raise ValueError("Either --log-dir or --log-json must be specified")

    print(f"Loading log from: {log_path}")

    # Load logs
    logs = load_log_json(log_path)
    print(f"Loaded {len(logs)} training iterations")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        log_dir = os.path.dirname(log_path)
        output_path = os.path.join(log_dir, 'loss_curve.png')

    # Plot
    plot_loss_curves(logs, output_path, args.smooth, tuple(args.figsize))


if __name__ == '__main__':
    main()
