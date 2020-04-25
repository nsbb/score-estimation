── 지도학습
  ├── data.py            : 정리되지 않은 Raw 데이터를 점수별로 이미지 갯수가 골고루 분포되도록 Shuffle 및 Arrange 하여 데이터 전처리.
	├── score.py           : 전달받은 이미지들을 Data Augmentation하여 4배 혹은 20배로 늘려준 후, Resnet18모델을 사용하여 이미지의 Score를 예측.
  └── sharpness.py       : 여러 sharpness function 수식들을 function으로 구현. 입력받은 이미지의 sharpness 값을 계산하여 return.
