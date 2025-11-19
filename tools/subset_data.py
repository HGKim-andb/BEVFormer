"""
Full dataset의 일부만 추출해서 새로운 pkl 파일 생성
"""
import pickle
import argparse
import random

def create_subset_pkl(input_pkl, output_pkl, ratio=0.1, seed=42):
    """
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
    
    # 샘플 수 계산
    num_samples = int(len(original_infos) * ratio)
    print(f'Subset samples: {num_samples} ({ratio*100:.1f}%)')
    
    # 랜덤 샘플링
    random.seed(seed)
    subset_indices = sorted(random.sample(range(len(original_infos)), num_samples))
    subset_infos = [original_infos[i] for i in subset_indices]
    
    # 새로운 데이터 생성
    data['infos'] = subset_infos
    
    # 저장
    print(f'Saving to {output_pkl}...')
    with open(output_pkl, 'wb') as f:
        pickle.dump(data, f)
    
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input pkl file')
    parser.add_argument('--output', required=True, help='Output pkl file')
    parser.add_argument('--ratio', type=float, default=0.1, help='Ratio of data to use (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    create_subset_pkl(args.input, args.output, args.ratio, args.seed)

