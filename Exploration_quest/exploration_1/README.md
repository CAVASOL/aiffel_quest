## 2-1. 프로젝트: 새로운 데이터셋으로 나만의 이미지 분류기 만들어보기

| 평가문항  | 상세기준 | 
| :--- | :--- | 
| 1. base 모델을 활용한 Transfer learning이 성공적으로 진행되었는가? | VGG16 등이 적절히 활용되었음 | 
| 2. 학습과정 및 결과에 대한 설명이 시각화를 포함하여 체계적으로 진행되었는가? | loss, accuracy 그래프 및 임의 사진 추론 결과가 제시됨 |   
| 3. 분류모델의 test accuracy가 기준 이상 높게 나왔는가? | test accuracy가 85% 이상 도달하였음 | 

### Details

`tensorflow_datasets`의 `tf_flowers` 데이터를 사용하여 `ResNet18`, `ResNet34`, `ResNet50`, `VGG16` 모듈 별로 모델을 디자인하고, 결과를 시각화하여 확인한 후 모델 성능을 올리기 위해 다각도로 실험을 진행했습니다. 데이터 전처리 과정에서 `Autotune` 기법을 적용했고, 모델의 성능을 최적화하기 위해 Dropout()이나 Batch Normalization()과 같은 `Regularization` 기법을 적용하여 모델을 디자인하였습니다. 데이터를 학습한 후, 임의의 새로운 사진 `daisy.jpg`를 활용하여 테스트 하였습니다.

### Result

| **VGG16** | **ResNet18** | **ResNet34** | **ResNet34-recap** | **ResNet50** |
| :---: | :---: | :---: | :---: | :---: | 
| 87.5 | 87.5 | 75.0 | 75.0 | 87.5 |
| 100% daisy | 100% daisy | 100% sunflowers | 100% daisy | 100% tulips |

### Retrospect

>tensorflow_datasets의 tf_flowers 데이터를 각각의 모델에 활용하여 성능을 평가했을 때, 결과는 각각 다르게 나타났습니다. 옵티마이저에 따른 변화가 눈의 띄었고, 에폭 조정에 따라서도 결과가 다르게 나타났습니다. 데이터의 유형과 크기를 고려한 모델 디자인과 성능 최적화에 노력했습니다. 모델 디자인과 데이터 학습 시 발생한 오류를 해결하는 과정에서 다양한 해결 방안들을 시도하였고, 결과를 나타내는 지표들의 변화들을 관찰하는 것에 집중했습니다. 각 모듈 별 실험에 관한 자세한 세부적인 내용은 해당 모듈 이름의 파일에서 확인하실 수 있으며, 실험 과정과 결과에 대한 비교 및 궁금했던 부분에 대해 xp1_retrospect.ipynb 파일에 정리하였습니다.

### Reference

* [Load and preprocess images](https://www.tensorflow.org/tutorials/load_data/images)
* [Image classification](https://www.tensorflow.org/tutorials/images/classification)
* [TensorFlow model optimization](https://www.tensorflow.org/model_optimization/guide)
* [Weight clustering in Keras example](https://www.tensorflow.org/model_optimization/guide/clustering/clustering_example)
* [In-depth EDA and K-Means Clustering](https://www.kaggle.com/code/thebrownviking20/in-depth-eda-and-k-means-clustering)
* [Clustering and Analysis using EDA and K Means](https://www.kaggle.com/code/ham9615/clustering-and-analysis-using-eda-and-k-means)
* [Python for Data Science: Implementing Exploratory Data Analysis (EDA) and K-Means Clustering](https://medium.com/@aziszamcalvin/python-for-data-science-implementing-exploratory-data-analysis-eda-and-k-means-clustering-bcf1d24adc12)
