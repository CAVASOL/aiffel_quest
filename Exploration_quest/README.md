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
    ├── exploration_4
    │       ├── cifar10_init.gif
    │       ├── cifar10_recap.gif
    │       ├── fashion_mnist_dcgan.gif
    │       ├── README.md
    │       ├── xp4_generative_modeling.ipynb
    │       ├── xp4_initial.png
    │       ├── xp4_project.ipynb 
    │       └── xp4_recap.png
    │ 
    ├── exploration_5
    │       ├── xp5_project.ipynb
    │       ├── xp5_recap_pytorch.ipynb   
    │       ├── xp5_transformer_chatbot.ipynb
    │       └── README.md
    │
    ├── exploration_6
    │       ├── xp6_controlnet.ipynb
    │       ├── xp6_deploy_project.ipynb  
    │       ├── xp6_mlops.ipynb
    │       ├── xp6_stable_diffusion.ipynb
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

### Exploration 4

FASHION-MNIST 데이터 생성용 `DCGAN 모델구조`를 이용해서 `CIFAR-10` 데이터를 생성하는 모델을 제작했습니다. 이미지 데이터의 형태를 (28, 28, 1)에서 (32, 32, 3)으로 조정했고, 이를 반영하여 생성자와 판별자 모델의 입력 및 출력 형태와 모델 구조를 변경했습니다. 또한, RGB 3채널의 컬러 이미지 시각화 과정과 관련하여 고려해야 할 부분들을 추가하여 작업하였고, 결과를 확인했습니다. 모델 성능을 최적화하고, 선명한 이미지 데이터 결과를 추출하기 위해 가설을 세우고 모델에 적용했습니다. 모델을 재구성한 후 다시 학습을 진행하였으며, 결과를 시각화하여 확인했습니다.

### Exploration 5

한국어 챗봇을 위해 영어 챗봇 모델을 한국어 데이터로 변환하는 실험의 과정은 다음과 같습니다. 한국어 챗봇 데이터를 수집하여 전처리하고, `SubwordTextEncoder`를 활용하여 토큰화합니다. 이후, 트랜스포머 모델을 구현하고 학습시킨 후, 적절한 예측 함수를 통해 입력된 문장에 대한 응답을 생성하고 모델을 평가합니다. 이를 통해 한국어에 적합한 챗봇 모델을 구축하고, 테스트하여 결과를 확인했습니다.