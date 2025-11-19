#!/usr/bin/env python3
"""
Split risk labels using nuScenes official train/val split
"""

import pickle
import argparse
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

def main():
    parser = argparse.ArgumentParser(description='Split risk labels using official nuScenes split')
    parser.add_argument('--input', type=str, required=True,
                        help='Input risk labels pkl file')
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                        help='nuScenes version')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    
    args = parser.parse_args()
    
    print("="*80)
    print("RISK LABELS SPLIT (Using Official nuScenes Split)")
    print("="*80)
    
    # Load nuScenes
    print(f"\n1. Loading nuScenes {args.version}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    
    # Get official splits
    official_splits = create_splits_scenes()
    print(f"   Official split info:")
    print(f"     Total train scenes in full dataset: {len(official_splits['train'])}")
    print(f"     Total val scenes in full dataset: {len(official_splits['val'])}")
    
    # Load risk labels
    print(f"\n2. Loading risk labels from {args.input}...")
    with open(args.input, 'rb') as f:
        all_labels = pickle.load(f)
    
    # Map scene token to scene name
    scene_token_to_name = {scene['token']: scene['name'] for scene in nusc.scene}
    
    # Split based on official split
    train_labels = {}
    val_labels = {}
    
    print(f"\n3. Splitting based on official split...")
    for scene_token, labels in all_labels.items():
        scene_name = scene_token_to_name.get(scene_token, 'unknown')
        
        if scene_name in official_splits['train']:
            train_labels[scene_token] = labels
        elif scene_name in official_splits['val']:
            val_labels[scene_token] = labels
        else:
            # Default to train if not found
            print(f"   Warning: {scene_name} not in official split, adding to train")
            train_labels[scene_token] = labels
    
    train_samples = sum(len(labels) for labels in train_labels.values())
    val_samples = sum(len(labels) for labels in val_labels.values())
    
    print(f"\n   Results:")
    print(f"     Train: {len(train_labels)} scenes, {train_samples} samples ({train_samples/(train_samples+val_samples)*100:.1f}%)")
    print(f"     Val:   {len(val_labels)} scenes, {val_samples} samples ({val_samples/(train_samples+val_samples)*100:.1f}%)")
    
    # Show which scenes went where
    print(f"\n   Train scenes:")
    for token in train_labels.keys():
        name = scene_token_to_name[token]
        samples = len(train_labels[token])
        print(f"     - {name} ({samples} samples)")
    
    print(f"\n   Val scenes:")
    for token in val_labels.keys():
        name = scene_token_to_name[token]
        samples = len(val_labels[token])
        print(f"     - {name} ({samples} samples)")
    
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
    print("✅ SPLIT COMPLETE (Using Official Split)")
    print("="*80)

if __name__ == '__main__':
    main()
