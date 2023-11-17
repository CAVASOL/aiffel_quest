## Exploration Quest

### Composition

```
Exploration_quest
    ├── exploration_1
    │       ├── xp1_cats_and_dogs.ipynb
    │       ├── xp1_flowers_resnet18.ipynb
    │       ├── xp1_flowers_resnet34_recap.ipynb
    │       ├── xp1_flowers_resnet34.ipynb
    │       ├── xp1_flowers_resnet50.ipynb
    │       ├── xp1_flowers_vgg16.ipynb
    │       ├── xp1_retrospect.ipynb 
    │       └── README.md
    │     
    ├── exploration_2
    │       ├── xp2_amazon_review.ipynb
    │       ├── xp2_news_summary.ipynb 
    │       ├── xp2_retrospect.ipynb 
    │       └── README.md   
    │   
    ├── exploration_3
    │       ├── xp3_semantic_segmentation.ipynb
    │       ├── xp3_project.ipynb 
    │       └── README.md 
    │ 
    └── README.md
```
### Exploration 1

`tensorflow_datasets`의 `tf_flowers` 데이터를 사용하여 `ResNet18`, `ResNet34`, `ResNet50`, `VGG16` 모듈 별로 모델을 디자인하고, 결과를 시각화하여 확인한 후 모델 성능을 올리기 위해 다각도로 실험을 진행했습니다. 데이터 전처리 과정에서 `Autotune` 기법을 적용했고, 모델의 성능을 최적화하기 위해 Dropout()이나 Batch Normalization()과 같은 `Regularization` 기법을 적용하여 모델을 디자인하였습니다. 데이터를 학습한 후, 임의의 새로운 사진 `daisy.jpg`를 활용하여 테스트 하였습니다.

### Exploration 2

뉴스 기사 데이터인 `news_summary_more.csv`를 활용하여 `Extractive/Abstractive summarization`을 이해하기 위해 텍스트 정규화 기법을 적용했습니다. 이를 위해 다양한 방법으로 단어장의 크기를 줄였고, `seq2seq` 모델을 디자인했습니다. 또한 `Bahdanau 스타일의 어텐션 메커니즘`을 적용하여 모델의 성능을 향상시켰습니다. 이후, `Summa`의 `summarize` 기능을 활용하여 추출적 요약을 수행했습니다. 실험의 과정과 결과와 관련한 회고는 `xp2_retrospect.ipynb` 파일에 정리했습니다.

### Exploration 3

인물, 강아지와 고양이, 그리고 자동차 사진에 `OpenCV` 라는 이미지 처리 라이브러리를 이용하여 아웃포커싱 효과를 적용했습니다. 또한 `Semantic Segmentation` 기술을 사용하여 물체를 감지하고 분할한 후, 두 장의 이미지를 합성하여 배경을 제거하고 크로마키 사진을 생성하였습니다. 