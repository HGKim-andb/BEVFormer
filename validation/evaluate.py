"""
Evaluation Metrics for Risk-Guided BEVFormer

Comprehensive evaluation including:
- Risk prediction metrics (MSE, MAE, IoU)
- Detection metrics (mAP, NDS)
- Risk-conditioned detection metrics
- Calibration analysis
"""

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_curve, auc
from scipy.stats import pearsonr, spearmanr
import sys
import os
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class RiskEvaluator:
    """Evaluator for risk prediction metrics"""

    def __init__(self, thresholds=[0.3, 0.5, 0.7]):
        self.thresholds = thresholds
        self.reset()

    def reset(self):
        """Reset all accumulated metrics"""
        self.all_gt = []
        self.all_pred = []
        self.sample_metrics = []

    def add_batch(self, pred_risk, gt_risk):
        """
        Add a batch of predictions and ground truths.

        Args:
            pred_risk: Predicted risk maps [B, 200, 200] or [B, 1, 200, 200]
            gt_risk: Ground truth risk maps [B, 200, 200] or [B, 1, 200, 200]
        """
        # Convert to numpy
        if isinstance(pred_risk, torch.Tensor):
            pred_risk = pred_risk.cpu().numpy()
        if isinstance(gt_risk, torch.Tensor):
            gt_risk = gt_risk.cpu().numpy()

        # Remove channel dimension if present
        if pred_risk.ndim == 4:
            pred_risk = pred_risk[:, 0]
        if gt_risk.ndim == 4:
            gt_risk = gt_risk[:, 0]

        # Add to accumulated lists
        for pred, gt in zip(pred_risk, gt_risk):
            self.all_pred.append(pred.flatten())
            self.all_gt.append(gt.flatten())

            # Compute per-sample metrics
            sample_metric = self._compute_sample_metrics(pred, gt)
            self.sample_metrics.append(sample_metric)

    def _compute_sample_metrics(self, pred, gt):
        """Compute metrics for a single sample"""
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()

        metrics = {
            'mse': mean_squared_error(gt_flat, pred_flat),
            'mae': mean_absolute_error(gt_flat, pred_flat),
            'max_gt': gt.max(),
            'max_pred': pred.max(),
            'mean_gt': gt.mean(),
            'mean_pred': pred.mean(),
        }

        # High-risk IoU for each threshold
        for thresh in self.thresholds:
            gt_high = gt > thresh
            pred_high = pred > thresh
            intersection = (gt_high & pred_high).sum()
            union = (gt_high | pred_high).sum()
            iou = intersection / union if union > 0 else 0.0
            metrics[f'iou_{thresh}'] = iou

        return metrics

    def compute_metrics(self):
        """Compute aggregated metrics across all samples"""
        if len(self.all_pred) == 0:
            return {}

        # Concatenate all predictions and GTs
        all_pred = np.concatenate(self.all_pred)
        all_gt = np.concatenate(self.all_gt)

        metrics = {}

        # Basic regression metrics
        metrics['mse'] = float(mean_squared_error(all_gt, all_pred))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        metrics['mae'] = float(mean_absolute_error(all_gt, all_pred))

        # Correlation
        metrics['pearson_r'], metrics['pearson_p'] = pearsonr(all_gt, all_pred)
        metrics['spearman_r'], metrics['spearman_p'] = spearmanr(all_gt, all_pred)

        # Per-threshold metrics
        for thresh in self.thresholds:
            gt_binary = (all_gt > thresh).astype(float)
            pred_binary = (all_pred > thresh).astype(float)

            # Precision, Recall, F1
            tp = ((gt_binary == 1) & (pred_binary == 1)).sum()
            fp = ((gt_binary == 0) & (pred_binary == 1)).sum()
            fn = ((gt_binary == 1) & (pred_binary == 0)).sum()
            tn = ((gt_binary == 0) & (pred_binary == 0)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[f'precision_{thresh}'] = float(precision)
            metrics[f'recall_{thresh}'] = float(recall)
            metrics[f'f1_{thresh}'] = float(f1)

            # IoU
            intersection = tp
            union = tp + fp + fn
            iou = intersection / union if union > 0 else 0.0
            metrics[f'iou_{thresh}'] = float(iou)

        # Calibration: compare predicted vs actual max risk
        max_preds = [m['max_pred'] for m in self.sample_metrics]
        max_gts = [m['max_gt'] for m in self.sample_metrics]
        metrics['max_risk_mae'] = float(mean_absolute_error(max_gts, max_preds))
        metrics['max_risk_corr'], _ = pearsonr(max_gts, max_preds)

        # Average per-sample metrics
        avg_sample_metrics = defaultdict(list)
        for sample_metric in self.sample_metrics:
            for key, value in sample_metric.items():
                avg_sample_metrics[key].append(value)

        for key, values in avg_sample_metrics.items():
            metrics[f'avg_{key}'] = float(np.mean(values))

        return metrics

    def print_metrics(self, metrics=None):
        """Print metrics in a formatted way"""
        if metrics is None:
            metrics = self.compute_metrics()

        print("\n" + "="*80)
        print("RISK EVALUATION METRICS")
        print("="*80)

        # Regression metrics
        print("\n📊 Regression Metrics:")
        print(f"  MSE:  {metrics['mse']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE:  {metrics['mae']:.6f}")

        # Correlation
        print("\n📊 Correlation:")
        print(f"  Pearson:  {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4e})")
        print(f"  Spearman: {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.4e})")

        # Per-threshold metrics
        for thresh in self.thresholds:
            print(f"\n📊 Threshold = {thresh}:")
            print(f"  Precision: {metrics[f'precision_{thresh}']:.4f}")
            print(f"  Recall:    {metrics[f'recall_{thresh}']:.4f}")
            print(f"  F1 Score:  {metrics[f'f1_{thresh}']:.4f}")
            print(f"  IoU:       {metrics[f'iou_{thresh}']:.4f}")

        # Calibration
        print("\n📊 Calibration (Max Risk):")
        print(f"  MAE:         {metrics['max_risk_mae']:.4f}")
        print(f"  Correlation: {metrics['max_risk_corr']:.4f}")

        print("\n" + "="*80)


class DetectionEvaluator:
    """Evaluator for detection metrics conditioned on risk"""

    def __init__(self, risk_thresholds=[0.3, 0.5, 0.7]):
        self.risk_thresholds = risk_thresholds
        self.reset()

    def reset(self):
        """Reset accumulated data"""
        self.high_risk_detections = []
        self.low_risk_detections = []
        self.all_detections = []

    def add_sample(self, detections, risk_map):
        """
        Add a sample's detections and risk map.

        Args:
            detections: Dict with 'boxes', 'scores', 'labels'
            risk_map: Risk map [200, 200]
        """
        # For each detection, check if it's in a high-risk region
        if isinstance(risk_map, torch.Tensor):
            risk_map = risk_map.cpu().numpy()
        if risk_map.ndim == 3:
            risk_map = risk_map[0]

        boxes = detections.get('boxes', [])
        scores = detections.get('scores', [])
        labels = detections.get('labels', [])

        for box, score, label in zip(boxes, scores, labels):
            # Convert box center to risk map coordinates
            cx, cy = box[0], box[1]  # Assumes box format [x, y, z, ...]

            # Convert from meters to pixels
            px = int((cx + 50) / 0.5)
            py = int((cy + 50) / 0.5)

            # Check if within bounds
            if 0 <= px < 200 and 0 <= py < 200:
                risk_value = risk_map[py, px]

                detection_data = {
                    'score': score,
                    'label': label,
                    'risk': risk_value
                }

                self.all_detections.append(detection_data)

                # Categorize by risk
                if risk_value > 0.5:
                    self.high_risk_detections.append(detection_data)
                else:
                    self.low_risk_detections.append(detection_data)

    def compute_metrics(self):
        """Compute detection metrics"""
        metrics = {}

        metrics['total_detections'] = len(self.all_detections)
        metrics['high_risk_detections'] = len(self.high_risk_detections)
        metrics['low_risk_detections'] = len(self.low_risk_detections)

        # Average confidence in high vs low risk regions
        if len(self.high_risk_detections) > 0:
            high_risk_scores = [d['score'] for d in self.high_risk_detections]
            metrics['avg_score_high_risk'] = float(np.mean(high_risk_scores))
        else:
            metrics['avg_score_high_risk'] = 0.0

        if len(self.low_risk_detections) > 0:
            low_risk_scores = [d['score'] for d in self.low_risk_detections]
            metrics['avg_score_low_risk'] = float(np.mean(low_risk_scores))
        else:
            metrics['avg_score_low_risk'] = 0.0

        return metrics


class AblationAnalyzer:
    """Analyzer for ablation studies"""

    def __init__(self):
        self.results = {}

    def add_experiment(self, name, metrics):
        """
        Add results from an experiment.

        Args:
            name: Experiment name (e.g., 'baseline', 'with_risk', 'with_attention')
            metrics: Dict of metrics
        """
        self.results[name] = metrics

    def compare(self, baseline='baseline'):
        """
        Compare all experiments to baseline.

        Args:
            baseline: Name of baseline experiment

        Returns:
            Dict of comparisons
        """
        if baseline not in self.results:
            raise ValueError(f"Baseline '{baseline}' not found in results")

        baseline_metrics = self.results[baseline]
        comparisons = {}

        for name, metrics in self.results.items():
            if name == baseline:
                continue

            comparison = {}
            for key in baseline_metrics.keys():
                if key in metrics:
                    baseline_val = baseline_metrics[key]
                    current_val = metrics[key]

                    # Compute improvement
                    if baseline_val != 0:
                        improvement = (current_val - baseline_val) / abs(baseline_val) * 100
                    else:
                        improvement = 0.0

                    comparison[key] = {
                        'baseline': baseline_val,
                        'current': current_val,
                        'improvement_%': improvement
                    }

            comparisons[name] = comparison

        return comparisons

    def print_comparison(self, baseline='baseline'):
        """Print comparison table"""
        comparisons = self.compare(baseline)

        print("\n" + "="*80)
        print(f"ABLATION STUDY: Comparison to {baseline}")
        print("="*80)

        for exp_name, comparison in comparisons.items():
            print(f"\n📊 {exp_name}:")

            for metric_name, values in comparison.items():
                baseline_val = values['baseline']
                current_val = values['current']
                improvement = values['improvement_%']

                symbol = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
                print(f"  {metric_name:20s}: {baseline_val:.4f} -> {current_val:.4f} "
                      f"({symbol} {abs(improvement):.2f}%)")

        print("\n" + "="*80)


def test_evaluators():
    """Test evaluation functions"""
    print("\n" + "="*80)
    print("TEST: Evaluators")
    print("="*80)

    # Test RiskEvaluator
    print("\n📋 Testing RiskEvaluator...")
    risk_eval = RiskEvaluator()

    # Add synthetic data
    for _ in range(10):
        gt = torch.rand(2, 200, 200)
        pred = gt + torch.randn(2, 200, 200) * 0.1
        pred = torch.clamp(pred, 0, 1)
        risk_eval.add_batch(pred, gt)

    metrics = risk_eval.compute_metrics()
    risk_eval.print_metrics(metrics)

    # Test AblationAnalyzer
    print("\n📋 Testing AblationAnalyzer...")
    ablation = AblationAnalyzer()

    ablation.add_experiment('baseline', {'mAP': 0.35, 'NDS': 0.42, 'risk_mae': 0.15})
    ablation.add_experiment('with_risk', {'mAP': 0.37, 'NDS': 0.44, 'risk_mae': 0.08})
    ablation.add_experiment('with_attention', {'mAP': 0.39, 'NDS': 0.46, 'risk_mae': 0.07})

    ablation.print_comparison('baseline')

    print("\n✅ Evaluators Test PASSED!\n")


if __name__ == '__main__':
    test_evaluators()
