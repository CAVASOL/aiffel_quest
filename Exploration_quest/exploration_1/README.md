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
---
## AIFFEL Campus Online Code Peer Review Templete
- 코더 : 김연
- 리뷰어 : 전다빈

   
## PRT(Peer Review Template)
- [O]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부
    base_model = tf.keras.applications.VGG16(input_shape = IMG_SHAPE,
                                         include_top = False,
                                         weights = 'imagenet')

EPOCHS = 15

history = model.fit(train_batches, epochs = EPOCHS,
                    validation_data = validation_batches)
- VGG16 모델이 적용되었고 85%이상의 성능을 보였습니다.
- [O]  **2. 전체 코드에서 가장 핵심적이거나 저가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
    - 
- [O]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
    - 굉장히 다양한 모델을 시도하면서 각각의 모델에 어떤 부분을 적용하였고 그 결과 어떠한 결과가 나왔는지 이해가 쉽도록  비교를 해주었다 생각합니다.       
- [O]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
    -tensorflow_datasets의 tf_flowers 데이터를 각각의 모델에 활용하여 성능을 평가했을 때, 결과는 각각 다르게 나타났습니다. 옵티마이저에 따른 변화가 눈의 띄었고, 에폭 조정에 따라서도 결과가 다르게 나타났습니다. 데이터의 유형과 크기를 고려한 모델 디자인과 성능 최적화에 노력했습니다. 모델 디자인과 데이터 학습 시 발생한 오류를 해결하는 과정에서 다양한 해결 방안들을 시도하였고, 결과를 나타내는 지표들의 변화들을 관찰하는 것에 집중했습니다. 각 모듈 별 실험에 관한 자세한 세부적인 내용은 해당 모듈 이름의 파일에서 확인하실 수 있으며, 실험 과정과 결과에 대한 비교 및 궁금했던 부분에 대해 xp1_retrospect.ipynb 파일에 정리하였습니다.    
-      
- [O]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
	- 
- 연님 프로젝트를 보며 많이 반성이 되었습니다. 앞으로도 더 많은 배움 부탁드릴게요!
## 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```

