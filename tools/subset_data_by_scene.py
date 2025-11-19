"""
Scene 단위로 dataset을 나누는 스크립트
연속성을 유지하므로 temporal 모델에 적합
"""
import pickle
import argparse
import random
from collections import defaultdict

def create_subset_by_scene(input_pkl, output_pkl, ratio=0.1, seed=42):
    """
    Scene 단위로 샘플링 (연속성 유지)
    
    Args:
        input_pkl: 원본 pkl 파일 경로
        output_pkl: 출력 pkl 파일 경로
        ratio: 사용할 데이터 비율 (0.1 = 10%)
        seed: random seed
    """
    print(f'Loading {input_pkl}...')
    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)
    
    original_infos = data['infos']
    print(f'Original samples: {len(original_infos)}')
    
    # Scene별로 샘플 그룹화
    scene_to_samples = defaultdict(list)
    for idx, info in enumerate(original_infos):
        scene_token = info['scene_token']
        scene_to_samples[scene_token].append(idx)
    
    print(f'Total scenes: {len(scene_to_samples)}')
    
    # Scene 단위로 샘플링
    random.seed(seed)
    all_scenes = list(scene_to_samples.keys())
    num_scenes = int(len(all_scenes) * ratio)
    selected_scenes = random.sample(all_scenes, num_scenes)
    
    print(f'Selected scenes: {num_scenes} ({ratio*100:.1f}%)')
    
    # 선택된 scene의 모든 샘플 수집
    selected_indices = []
    for scene_token in selected_scenes:
        selected_indices.extend(scene_to_samples[scene_token])
    
    # 시간 순서 유지
    selected_indices.sort()
    subset_infos = [original_infos[i] for i in selected_indices]
    
    print(f'Subset samples: {len(subset_infos)} ({len(subset_infos)/len(original_infos)*100:.1f}%)')
    
    # 새로운 데이터 생성
    data['infos'] = subset_infos
    
    # 저장
    print(f'Saving to {output_pkl}...')
    with open(output_pkl, 'wb') as f:
        pickle.dump(data, f)
    
    # 선택된 scene 정보를 JSON으로 저장
    import json
    from pathlib import Path
    
    info_path = Path(output_pkl).with_suffix('.json')
    scene_info_to_save = {
        'selection_method': 'random',
        'ratio': ratio,
        'seed': seed,
        'num_scenes': num_scenes,
        'num_samples': len(subset_infos),
        'selected_scene_tokens': selected_scenes
    }
    
    with open(info_path, 'w') as f:
        json.dump(scene_info_to_save, f, indent=2)
    
    print(f'Scene info saved to: {info_path}')
    
    # Scene token 목록도 텍스트 파일로 저장
    token_list_path = Path(output_pkl).with_suffix('.txt')
    with open(token_list_path, 'w') as f:
        f.write(f"Selected {num_scenes} scenes (random strategy, ratio={ratio}, seed={seed})\n")
        f.write("=" * 80 + "\n\n")
        for i, scene_token in enumerate(selected_scenes, 1):
            num_samples = len(scene_to_samples[scene_token])
            f.write(f"{i:3d}. Scene: {scene_token}  (Samples: {num_samples})\n")
    
    print(f'Scene list saved to: {token_list_path}')
    
    print('\nDone!')
    
    # 통계
    print('\n=== Statistics ===')
    samples_per_scene = [len(scene_to_samples[s]) for s in selected_scenes]
    print(f'Samples per scene: min={min(samples_per_scene)}, '
          f'max={max(samples_per_scene)}, '
          f'avg={sum(samples_per_scene)/len(samples_per_scene):.1f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scene 단위로 dataset subset 생성')
    parser.add_argument('--input', required=True, help='Input pkl file')
    parser.add_argument('--output', required=True, help='Output pkl file')
    parser.add_argument('--ratio', type=float, default=0.1, 
                       help='Scene 비율 (default: 0.1 = 10%)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    create_subset_by_scene(args.input, args.output, args.ratio, args.seed)

