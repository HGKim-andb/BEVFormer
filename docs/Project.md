# Emergence Prediction for Autonomous Driving

## 배경
자율주행에서 가려진 영역(occluded regions)에서 갑자기 나타나는 객체(pedestrian, vehicle 등)를 
사전에 예측하는 연구를 하고 있습니다.

## 목표
nuScenes 데이터셋의 3D annotation을 사용해서, 
"과거에는 없었다가 미래에 나타나는 객체"를 찾아서 
BEV grid (200x200) label로 변환하는 것입니다.

## Emergence의 정의
- 과거 5 프레임 (t-5 ~ t-1)에는 없었던 객체
- 미래 3 프레임 (t+1, t+2, t+3) 중 하나에 나타남
- 센서 범위 내 (2m < distance < 40m)
- 유효한 카테고리 (vehicle, pedestrian, bicycle 등)