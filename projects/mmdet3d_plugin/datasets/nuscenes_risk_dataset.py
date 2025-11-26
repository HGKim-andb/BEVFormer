"""
NuScenes Dataset with Risk Labels
"""

import copy
import pickle
import numpy as np
import torch
from os import path as osp
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC

from .nuscenes_dataset import CustomNuScenesDataset


@DATASETS.register_module()
class NuScenesRiskDataset(CustomNuScenesDataset):
    """
    NuScenes Dataset with Risk Map Labels.

    This dataset extends CustomNuScenesDataset to load and provide
    risk map ground truth labels for training.

    Args:
        risk_labels_path (str): Path to risk labels pickle file
            Default: 'data/emergence_risk_v5_full/risk_labels_train.pkl'
        risk_map_size (tuple): Size of risk map (H, W). Default: (200, 200)
        use_risk (bool): Whether to load risk labels. Default: True
        risk_threshold (float): Threshold for high-risk filtering. Default: 0.0
            If > 0, only samples with max_risk > threshold are used
    """

    def __init__(self,
                 risk_labels_path='data/emergence_risk_v5_full/risk_labels_train.pkl',
                 risk_map_size=(200, 200),
                 use_risk=True,
                 risk_threshold=0.0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.use_risk = use_risk
        self.risk_map_size = risk_map_size
        self.risk_threshold = risk_threshold

        # Load risk labels
        if self.use_risk:
            self.risk_labels_dict = self._load_risk_labels(risk_labels_path)
            print(f"Loaded risk labels from {risk_labels_path}")

            # Build sample_token -> risk_label mapping for fast lookup
            self.risk_map_dict = {}
            total_samples = 0
            for scene_token, scene_labels in self.risk_labels_dict.items():
                for label in scene_labels:
                    sample_token = label['sample_token']
                    self.risk_map_dict[sample_token] = label
                    total_samples += 1

            print(f"Total risk labels: {total_samples}")

            # Filter samples by risk threshold if specified
            if self.risk_threshold > 0:
                self._filter_by_risk_threshold()
        else:
            self.risk_labels_dict = None
            self.risk_map_dict = {}

    def _load_risk_labels(self, risk_labels_path):
        """Load risk labels from pickle file."""
        if not osp.exists(risk_labels_path):
            raise FileNotFoundError(
                f"Risk labels file not found: {risk_labels_path}\n"
                f"Please generate risk labels first using tools/create_risk_labels.py"
            )

        with open(risk_labels_path, 'rb') as f:
            risk_labels = pickle.load(f)

        return risk_labels

    def _filter_by_risk_threshold(self):
        """Filter dataset to only include samples with high risk."""
        print(f"Filtering samples with max_risk > {self.risk_threshold}")

        # Get sample tokens that meet threshold
        valid_tokens = set()
        for sample_token, risk_label in self.risk_map_dict.items():
            if risk_label['metadata']['max_risk'] > self.risk_threshold:
                valid_tokens.add(sample_token)

        # Filter data_infos
        original_count = len(self.data_infos)
        self.data_infos = [
            info for info in self.data_infos
            if info['token'] in valid_tokens
        ]

        print(f"Filtered: {original_count} -> {len(self.data_infos)} samples "
              f"({len(self.data_infos) / original_count * 100:.1f}%)")

    def get_risk_label(self, sample_token):
        """
        Get risk label for a given sample token.

        Args:
            sample_token (str): Sample token

        Returns:
            dict or None: Risk label dict containing:
                - risk_map: np.ndarray (200, 200)
                - ego_state: dict
                - metadata: dict
        """
        if not self.use_risk:
            return None

        return self.risk_map_dict.get(sample_token, None)

    def get_data_info(self, index):
        """
        Get data info with risk label.

        Extends parent method to add risk map information.
        """
        input_dict = super().get_data_info(index)

        # Add risk label if available
        if self.use_risk:
            sample_token = input_dict['sample_idx']
            risk_label = self.get_risk_label(sample_token)

            if risk_label is not None:
                input_dict['risk_label'] = risk_label
            else:
                # Create zero risk map if label not found
                input_dict['risk_label'] = {
                    'risk_map': np.zeros(self.risk_map_size, dtype=np.float32),
                    'metadata': {'max_risk': 0.0, 'mean_risk': 0.0}
                }

        return input_dict

    def union2one(self, queue):
        """
        Union queue samples with risk maps.

        Extends parent method to handle risk map stacking.
        """
        # Call parent method
        result = super().union2one(queue)

        # If parent method returns None, return None
        if result is None:
            return None

        # Stack risk maps from queue
        if self.use_risk:
            # Get risk map from the last frame (current frame)
            risk_label = None
            if len(queue) > 0 and queue[-1] is not None and 'risk_label' in queue[-1]:
                risk_label = queue[-1].get('risk_label')

            if risk_label is not None and 'risk_map' in risk_label:
                risk_map = risk_label['risk_map']

                # Convert to tensor
                if isinstance(risk_map, np.ndarray):
                    risk_map = torch.from_numpy(risk_map).float()

                # Add to result
                result['gt_risk_map'] = DC(risk_map, cpu_only=False)

                # Also add metadata for analysis
                result['risk_metadata'] = DC(risk_label.get('metadata', {}), cpu_only=True)
            else:
                # Fallback: add zero risk map
                zero_risk = torch.zeros(self.risk_map_size, dtype=torch.float32)
                result['gt_risk_map'] = DC(zero_risk, cpu_only=False)
                result['risk_metadata'] = DC({'max_risk': 0.0, 'mean_risk': 0.0}, cpu_only=True)

        return result

    def prepare_train_data(self, index):
        """
        Training data preparation with risk labels.
        """
        data = super().prepare_train_data(index)

        # Ensure risk map is present
        if data is not None and self.use_risk:
            if 'gt_risk_map' not in data:
                # Add zero risk map as fallback
                zero_risk = torch.zeros(self.risk_map_size, dtype=torch.float32)
                data['gt_risk_map'] = DC(zero_risk, cpu_only=False)

        return data

    def evaluate(self, results, logger=None, **kwargs):
        """
        Evaluate both detection and risk prediction.

        Args:
            results (list[dict]): List of result dicts containing 'pts_bbox' and optionally 'risk_map'

        Returns:
            dict: Combined evaluation metrics from detection and risk
        """
        # Separate risk_map from results for detection evaluation
        results_for_detection = []
        risk_maps = []

        for result in results:
            # Extract pts_bbox for detection evaluation
            det_result = {'pts_bbox': result['pts_bbox']}
            results_for_detection.append(det_result)

            # Extract risk_map if available
            risk_map = result.get('risk_map', None)
            risk_maps.append(risk_map)

        # 1. Detection evaluation (parent class)
        det_metrics = super().evaluate(results_for_detection, logger=logger, **kwargs)

        # 2. Risk evaluation
        risk_metrics = {}
        if self.use_risk and any(rm is not None for rm in risk_maps):
            # Create results with only risk_map for risk evaluation
            risk_results = [{'risk_map': rm} if rm is not None else {} for rm in risk_maps]
            risk_metrics = self.evaluate_risk(risk_results, logger=logger, **kwargs)

        # 3. Merge metrics
        combined_metrics = {**det_metrics, **risk_metrics}

        return combined_metrics

    def evaluate_risk(self, results, logger=None, **kwargs):
        """
        Evaluate risk prediction performance.

        Args:
            results (list[dict]): List of result dicts containing 'risk_map'
            logger: Logger for output

        Returns:
            dict: Evaluation metrics
        """
        if not self.use_risk:
            print("[Risk Eval] use_risk=False, skipping risk evaluation")
            return {}

        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import numpy as np

        print(f"[Risk Eval] Starting risk evaluation with {len(results)} results")

        all_pred_risks = []
        all_gt_risks = []
        all_max_risks_pred = []
        all_max_risks_gt = []

        for i, result in enumerate(results):
            if 'risk_map' not in result:
                if i < 5:  # Only print first 5 to avoid spam
                    print(f"[Risk Eval] Sample {i}: risk_map not in result, keys={list(result.keys())}")
                continue

            # Get prediction
            pred_risk = result['risk_map']  # [1, 200, 200] or [200, 200]
            if isinstance(pred_risk, torch.Tensor):
                pred_risk = pred_risk.cpu().numpy()
            if pred_risk.ndim == 3:
                pred_risk = pred_risk[0]  # Remove channel dim

            # Get ground truth
            info = self.data_infos[i]
            risk_label = self.get_risk_label(info['token'])
            if risk_label is None:
                if i < 5:
                    print(f"[Risk Eval] Sample {i}: No GT risk label for token {info['token']}")
                continue

            gt_risk = risk_label['risk_map']

            # Collect for metrics
            all_pred_risks.append(pred_risk.flatten())
            all_gt_risks.append(gt_risk.flatten())
            all_max_risks_pred.append(pred_risk.max())
            all_max_risks_gt.append(gt_risk.max())

        print(f"[Risk Eval] Collected {len(all_pred_risks)} valid samples for evaluation")

        if len(all_pred_risks) == 0:
            print("[Risk Eval] No valid samples found, returning empty metrics")
            return {}

        # Concatenate all predictions and GTs
        all_pred_risks = np.concatenate(all_pred_risks)
        all_gt_risks = np.concatenate(all_gt_risks)

        # Calculate metrics
        mse = mean_squared_error(all_gt_risks, all_pred_risks)
        mae = mean_absolute_error(all_gt_risks, all_pred_risks)
        rmse = np.sqrt(mse)

        # High-risk cell metrics (threshold at 0.7)
        high_risk_mask_gt = all_gt_risks > 0.7
        high_risk_mask_pred = all_pred_risks > 0.7

        if high_risk_mask_gt.sum() > 0:
            # Precision and recall for high-risk cells
            tp = (high_risk_mask_gt & high_risk_mask_pred).sum()
            fp = (~high_risk_mask_gt & high_risk_mask_pred).sum()
            fn = (high_risk_mask_gt & ~high_risk_mask_pred).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1 = 0.0

        # Max risk correlation
        max_risk_corr = np.corrcoef(all_max_risks_pred, all_max_risks_gt)[0, 1]

        metrics = {
            'risk_mse': float(mse),
            'risk_rmse': float(rmse),
            'risk_mae': float(mae),
            'risk_precision': float(precision),
            'risk_recall': float(recall),
            'risk_f1': float(f1),
            'risk_max_corr': float(max_risk_corr),
        }

        print("[Risk Eval] Computed metrics:")
        for key, val in metrics.items():
            print(f"  {key}: {val:.4f}")

        if logger is not None:
            logger.info("Risk Evaluation Metrics:")
            for key, val in metrics.items():
                logger.info(f"  {key}: {val:.4f}")

        return metrics


@DATASETS.register_module()
class NuScenesRiskDatasetVal(NuScenesRiskDataset):
    """
    Validation dataset with risk labels.
    Uses validation split risk labels.
    """

    def __init__(self,
                 risk_labels_path='data/emergence_risk_v5_full/risk_labels_val.pkl',
                 *args,
                 **kwargs):
        super().__init__(risk_labels_path=risk_labels_path, *args, **kwargs)
