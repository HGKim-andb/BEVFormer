"""
객체 밀도가 높은 Scene 위주로 dataset을 나누는 스크립트
더 많은 학습 정보를 제공하는 scene 우선 선택
"""
import pickle
import argparse
import random
from collections import defaultdict

def create_subset_by_density(input_pkl, output_pkl, ratio=0.1, seed=42, 
                             selection='top', category_filter=None):
    """
    객체 밀도 기준으로 Scene 선택
    
    Args:
        input_pkl: 원본 pkl 파일 경로
        output_pkl: 출력 pkl 파일 경로
        ratio: 사용할 Scene 비율 (0.1 = 10%)
        seed: random seed (selection='random' 시 사용)
        selection: 'top' (밀도 높은 것), 'bottom' (낮은 것), 'random' (랜덤)
        category_filter: 특정 카테고리만 카운트 (예: ['car', 'truck', 'bus'])
    """
    print(f'Loading {input_pkl}...')
    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)
    
    original_infos = data['infos']
    print(f'Original samples: {len(original_infos)}')
    
    # Scene별로 샘플과 객체 수 수집
    scene_info = defaultdict(lambda: {'samples': [], 'obj_count': 0, 'category_counts': defaultdict(int)})
    
    for idx, info in enumerate(original_infos):
        scene_token = info['scene_token']
        scene_info[scene_token]['samples'].append(idx)
        
        # 객체 수 카운트
        gt_names = info.get('gt_names', [])
        for name in gt_names:
            if category_filter is None or name in category_filter:
                scene_info[scene_token]['obj_count'] += 1
                scene_info[scene_token]['category_counts'][name] += 1
    
    print(f'Total scenes: {len(scene_info)}')
    
    # Scene별 통계 계산
    scene_stats = []
    for scene_token, info in scene_info.items():
        num_samples = len(info['samples'])
        obj_count = info['obj_count']
        density = obj_count / num_samples if num_samples > 0 else 0
        
        scene_stats.append({
            'scene_token': scene_token,
            'num_samples': num_samples,
            'obj_count': obj_count,
            'density': density,
            'category_counts': dict(info['category_counts'])
        })
    
    # 밀도 기준 정렬
    scene_stats.sort(key=lambda x: x['density'], reverse=True)
    
    # 통계 출력
    print('\n=== Scene Statistics ===')
    print(f"Density range: {scene_stats[-1]['density']:.2f} ~ {scene_stats[0]['density']:.2f} objects/sample")
    print(f"Top 5 densest scenes:")
    for i in range(min(5, len(scene_stats))):
        s = scene_stats[i]
        print(f"  Scene {i+1}: {s['obj_count']} objects in {s['num_samples']} samples "
              f"(density={s['density']:.2f})")
    
    # Scene 선택
    num_scenes = int(len(scene_stats) * ratio)
    
    if selection == 'top':
        selected_scenes = scene_stats[:num_scenes]
        print(f'\n✅ Selecting TOP {num_scenes} densest scenes ({ratio*100:.1f}%)')
    elif selection == 'bottom':
        selected_scenes = scene_stats[-num_scenes:]
        print(f'\n✅ Selecting BOTTOM {num_scenes} scenes ({ratio*100:.1f}%)')
    else:  # random
        random.seed(seed)
        selected_scenes = random.sample(scene_stats, num_scenes)
        print(f'\n✅ Selecting RANDOM {num_scenes} scenes ({ratio*100:.1f}%)')
    
    # 선택된 scene의 모든 샘플 수집
    selected_scene_tokens = {s['scene_token'] for s in selected_scenes}
    selected_indices = []
    
    for scene_token in selected_scene_tokens:
        selected_indices.extend(scene_info[scene_token]['samples'])
    
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
        'selection_method': selection,
        'ratio': ratio,
        'num_scenes': num_scenes,
        'num_samples': len(subset_infos),
        'category_filter': category_filter,
        'selected_scenes': [
            {
                'scene_token': s['scene_token'],
                'num_samples': s['num_samples'],
                'obj_count': s['obj_count'],
                'density': round(s['density'], 2),
                'category_counts': s['category_counts']
            }
            for s in selected_scenes
        ]
    }
    
    with open(info_path, 'w') as f:
        json.dump(scene_info_to_save, f, indent=2)
    
    print(f'Scene info saved to: {info_path}')
    
    # Scene token 목록도 텍스트 파일로 저장 (간단히 확인용)
    token_list_path = Path(output_pkl).with_suffix('.txt')
    with open(token_list_path, 'w') as f:
        f.write(f"Selected {num_scenes} scenes ({selection} strategy, ratio={ratio})\n")
        f.write("=" * 80 + "\n\n")
        for i, s in enumerate(selected_scenes, 1):
            f.write(f"{i:3d}. Scene: {s['scene_token']}\n")
            f.write(f"     Samples: {s['num_samples']:4d}  |  Objects: {s['obj_count']:5d}  |  "
                   f"Density: {s['density']:6.2f} obj/sample\n")
            if s['category_counts']:
                top_3 = sorted(s['category_counts'].items(), key=lambda x: x[1], reverse=True)[:3]
                f.write(f"     Top categories: {', '.join(f'{k}({v})' for k, v in top_3)}\n")
            f.write("\n")
    
    print(f'Scene list saved to: {token_list_path}')
    
    # 선택된 scene 통계
    total_objects = sum(s['obj_count'] for s in selected_scenes)
    avg_density = sum(s['density'] for s in selected_scenes) / len(selected_scenes)
    
    print('\n=== Selected Scenes Statistics ===')
    print(f'Total objects: {total_objects}')
    print(f'Average density: {avg_density:.2f} objects/sample')
    print(f'Density range: {min(s["density"] for s in selected_scenes):.2f} ~ '
          f'{max(s["density"] for s in selected_scenes):.2f}')
    
    # Category 분포
    if category_filter:
        print(f'\nFiltered by categories: {category_filter}')
    else:
        category_totals = defaultdict(int)
        for s in selected_scenes:
            for cat, count in s['category_counts'].items():
                category_totals[cat] += count
        
        print('\nTop 10 categories:')
        sorted_cats = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
        for cat, count in sorted_cats[:10]:
            print(f'  {cat}: {count}')
    
    print('\nDone!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='객체 밀도 기준으로 Scene 선택하여 dataset subset 생성'
    )
    parser.add_argument('--input', required=True, help='Input pkl file')
    parser.add_argument('--output', required=True, help='Output pkl file')
    parser.add_argument('--ratio', type=float, default=0.1, 
                       help='Scene 비율 (default: 0.1 = 10%%)')
    parser.add_argument('--selection', choices=['top', 'bottom', 'random'], 
                       default='top',
                       help='Selection strategy: top (densest), bottom (sparsest), random')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed (for random selection)')
    parser.add_argument('--category-filter', nargs='+', 
                       help='특정 카테고리만 카운트 (예: car truck bus)')
    
    args = parser.parse_args()
    
    create_subset_by_density(
        args.input, args.output, args.ratio, args.seed,
        args.selection, args.category_filter
    )

