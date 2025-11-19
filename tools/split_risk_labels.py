#!/usr/bin/env python3
"""
Split existing risk labels into train/val based on nuScenes split
"""

import pickle
import argparse
from pathlib import Path
from nuscenes.nuscenes import NuScenes

def main():
    parser = argparse.ArgumentParser(description='Split risk labels into train/val')
    parser.add_argument('--input', type=str, required=True,
                        help='Input risk labels pkl file')
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                        help='nuScenes version')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--train_scenes', type=int, default=8,
                        help='Number of scenes for training (default: 8 for mini)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("RISK LABELS TRAIN/VAL SPLIT")
    print("="*80)
    
    # Load nuScenes to get scene info
    print(f"\n1. Loading nuScenes {args.version}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print(f"   Total scenes: {len(nusc.scene)}")
    
    # Load risk labels
    print(f"\n2. Loading risk labels from {args.input}...")
    with open(args.input, 'rb') as f:
        all_labels = pickle.load(f)
    
    total_samples = sum(len(labels) for labels in all_labels.values())
    print(f"   Total samples: {total_samples}")
    print(f"   Scenes in labels: {len(all_labels)}")
    
    # Get scene tokens for split
    train_scene_tokens = [scene['token'] for scene in nusc.scene[:args.train_scenes]]
    val_scene_tokens = [scene['token'] for scene in nusc.scene[args.train_scenes:]]
    
    print(f"\n3. Splitting...")
    print(f"   Train scenes: {args.train_scenes}")
    print(f"   Val scenes: {len(nusc.scene) - args.train_scenes}")
    
    # Split labels
    train_labels = {}
    val_labels = {}
    
    for scene_token, labels in all_labels.items():
        if scene_token in train_scene_tokens:
            train_labels[scene_token] = labels
        elif scene_token in val_scene_tokens:
            val_labels[scene_token] = labels
    
    train_samples = sum(len(labels) for labels in train_labels.values())
    val_samples = sum(len(labels) for labels in val_labels.values())
    
    print(f"   Train: {len(train_labels)} scenes, {train_samples} samples")
    print(f"   Val:   {len(val_labels)} scenes, {val_samples} samples")
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / 'risk_labels_train.pkl'
    val_path = output_dir / 'risk_labels_val.pkl'
    
    print(f"\n4. Saving...")
    with open(train_path, 'wb') as f:
        pickle.dump(train_labels, f)
    print(f"   ✓ Train: {train_path}")
    
    with open(val_path, 'wb') as f:
        pickle.dump(val_labels, f)
    print(f"   ✓ Val:   {val_path}")
    
    print("\n" + "="*80)
    print("✅ SPLIT COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
