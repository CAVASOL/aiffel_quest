## Main Quest 04. 폐렴아 기다려라!

| 평가문항  | 상세기준 | 
| :--- | :--- | 
| 1. 의료영상을 처리하는 CNN 기반 딥러닝 모델이 잘 구현되었다. | 모델 학습이 안정적으로 수렴하는 것을 시각화를 통해 확인하였다. | 
| 2. 데이터 준비, 모델 구성 등 전체 과정의 다양한 실험이 체계적으로 수행되었다. | Regularization, Augmentation 등의 기법 사용 여부에 따른 모델 성능 측정이 Ablation study 형태로 체계적으로 수행되었다. |   
| 3. ResNet 모델을 구현하였다. | Resnet의 Residual block을 구현하고, 학습의 경우 1에폭만 진행하여도 인정. | 

### 목차

```
5. 폐렴아 기다려라 (mq4_holdup.ipynb)

    5-1. 들어가며
    5-2. 의료영상에 대해
    5-3. X-RAY 이미지
    5-4. 폐렴을 진단해보자 (1)
        1) Set up
        2) 데이터 가져오기
        3) 데이터 시각화

    5-5. 폐렴을 진단해보자 (2)
        4) CNN 모델링
        5) 데이터 imbalance 처리
        6) 모델 훈련
        7) 결과 확인


6. 폐렴아 기다려라 - 프로젝트 (mq4_holdup_project.ipynb)

    6-1. 프로젝트 : 폐렴 진단기 성능개선
        Step 1. 실험환경 Set-up
        Step 2. 데이터 준비하기
        Step 3. 데이터 시각화

        Step 4. ResNet-18 구현
        Step 5. 데이터 imbalance 처리
        Step 6. 모델 훈련
        Step 7. 결과 확인과 시각화

        ** ResNet50 구현
        ** 데이터 imbalance 처리
        ** 모델 훈련
        ** 결과 확인과 시각화

    6-2. 프로젝트 제출
```

### 학습 내용

```
**5. 폐렴아 기다려라**

* 의료영상 데이터의 특징과 중요성에 대해 설명하여, 왜 이러한 기술이 의료 분야에서 중요한지 이해합니다.
* X-RAY 이미지의 구조와 폐렴 진단에 어떻게 활용되는지에 대해 살펴봅니다.
* CNN 모델 설계, 데이터 불균형 처리 방법, 모델 훈련, 모델의 성능을 확인하여 폐렴 진단 결과 평가 및 확인하는 과정을 살펴봅니다.

**6. 폐렴아 기다려라 - 프로젝트**

* 노드 5의 실습 과정을 되짚어보면서 어떤 점을 더 개선해볼 수 있을지 살펴보고, 모델의 성능을 향상시켜 봅시다.
* ResNet18 모듈과 ResNet50 모듈을 사용하여 모델을 구현 해보고, 결과를 살펴봅니다.
```



### CNN / ResNet18 / ResNet50 모델 구성 및 결과 확인

**CNN 모델 구성 및 결과**
>이미지의 크기는 180x180, BATCH_SIZE는 16, EPOCHS는 10으로 설정하였습니다. optimizer는 Adam, 손실 함수는 binary_crossentropy를 사용했습니다. LMS와 Jupiter 노트에서의 테스트 결과가 미미하지만 다르게 나타났습니다.

```
IMAGE_SIZE = [180, 180]
BATCH_SIZE = 16
EPOCHS = 10

def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ])

    return block

def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])

    return block

def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),

        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(),

        conv_block(32),
        conv_block(64),

        conv_block(128),
        tf.keras.layers.Dropout(0.2),

        conv_block(256),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten(),
        dense_block(512, 0.7),
        dense_block(128, 0.5),
        dense_block(64, 0.3),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model

with tf.device('/GPU:0'):
    model = build_model()

    METRICS = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=METRICS
    )
```

>테스트 결과는 다음과 같습니다.

```
LMS: Epoch 10/10:

    loss: 0.1350 - accuracy: 0.9519 - precision: 0.9876 - recall: 0.9472 
    val_loss: 0.1251 - val_accuracy: 0.9548 - val_precision: 0.9972 - val_recall: 0.9410

    LMS: Evaulation: 
    Loss: 0.4681052565574646 
    Accuracy: 0.8541666865348816, 
    Precision: 0.832962155342102, 
    Recall: 0.9589743614196777

Jupiter: Epoch 10/10:

    loss: 0.1188 - accuracy: 0.9579 - precision: 0.9871 - recall: 0.9560 
    val_loss: 0.0744 - val_accuracy: 0.9721 - val_precision: 0.9880 - val_recall: 0.9738

    Jupiter: Evaulation: 
    Loss: 0.616283655166626, 
    Accuracy: 0.8092948794364929, 
    Precision: 0.773737370967865, 
    Recall: 0.9820512533187866
```


---


**ResNet18 모델 구성 및 결과**
>ResNet 모델이 기본적으로 224x224 크기의 이미지를 사용하므로 이미지의 크기는 224x224, BATCH_SIZE는 32(16에서 32로 조정), EPOCHS는 20(10에서 20으로 조정)으로 설정하였습니다. 데이터 전처리 과정에서 이미지 데이터를 랜덤으로 좌우 반전하는 Augmentation 기법을 적용했습니다. 

```
def augment(image, label):
    image = tf.image.random_flip_left_right(image)  # 랜덤하게 좌우를 반전합니다.
    return image,label
```

>모델 구현에서 BatchNormalization()과 Dropout()을 같이 적용했습니다. Dropout()을 삽입하지 않았을 때의 테스트 결과 정확도는 79% 였으며, Dropout()을 삽입한 후의 테스트 결과 정확도는 87% 정도(82에서 87사이)로 나타났습니다. 컴파일 설정은 CNN의 경우와 동일합니다. 테스트 결과를 시각화하여 그래프로 확인하였을 때, 에폭에 따른 정확도 지표가 잦은 변동폭을 보였습니다. 에폭을 5 -> 10 -> 15 -> 20의 순으로 테스트 하였으며, 에폭이 적을 수록 학습의 결과가 매우 불안정했습니다. 결과적으로 에폭 20으로 다시 재설정하여 테스트를 진행했으며, 손실값의 경우 에폭 10에 진입하기 전까지는 불안정한 변동을 보였습니다.

```
from tensorflow.keras import layers

def residual_block(x, filters, s=1):
    shortcut = x
    x = layers.Conv2D(filters, (3, 3), strides=s, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    if s != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=s, padding='valid')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet18(input_shape=(224, 224, 3)):
    input_tensor = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, s=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, s=2)
    x = residual_block(x, 256)
    x = residual_block(x, 512, s=2)
    x = residual_block(x, 512)

    x = layers.Dropout(0.5)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=x)
    
    return model
```

>테스트 결과는 다음과 같습니다.

```
Loss: 0.3426632583141327,
Accuracy: 0.8766025900840759,
Precision: 0.9311294555664062,
Recall: 0.8666666746139526
```

---


**ResNet50 모델 구성 및 결과**
>ResNet18의 셋업과 동일하게 이미지의 크기는 224x224, BATCH_SIZE는 32, EPOCHS는 20으로 설정하였습니다. 모델 구현에서 BatchNormalization()과 Dropout()을 같이 적용했습니다. Dropout()을 삽입하지 않았을 때의 테스트 결과 정확도는 81% 였으며, Dropout()을 삽입한 후의 테스트 결과 정확도는 82% 정도로 나타났습니다. 차이는 미미합니다. 컴파일 설정은 CNN과 ResNet18의 경우와 동일합니다. ResNet18 모듈을 사용한 경우가 마찬가지로 테스트 결과를 시각화하여 그래프로 확인하면 정확도 지표의 변동이 요란하게 나타나는 것을 확인할 수 있었습니다. 

```
from tensorflow.keras.applications import ResNet50

def build_resnet50():
    base_model = ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3))

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    return model
```

>테스트 결과는 다음과 같습니다.

```
Loss: 0.5211844444274902,
Accuracy: 0.8285256624221802,
Precision: 0.8179775476455688,
Recall: 0.9333333373069763
```

---


### Retrospect

>의료영상 데이터의 특징과 중요성에 대해 알 수 있었으며, X-RAY 이미지의 구조와 폐렴 진단에 어떻게 활용되는지 살펴보기 위해 CNN, ResNet18, ResNet50 모듈을 사용하여 각각 모델을 구현하고, 결과를 시각화하여 살펴봤습니다. 각각의 모듈을 사용하여 구현한 모델의 학습 결과는 비슷했습니다. 데이터에 최적화 된 모델이나 셋업이었다고 확신할 수 없기에 테스트 결과만을 보고 성능을 검토하기에는 무리가 있습니다. 각각의 결과를 시각화하여 그래프로 확인하였을 때 ResNet18 모듈과 ResNet50 모듈의 경우 정확도 지표의 변동폭이 요란하게 나타났고, 전체 과정 중 어떤 부분이 이런 현상을 초래한걸까 궁금했습니다. 다양한 모듈을 적용하여 모델을 구현하고, 데이터 학습을 진행한 후 결과를 살펴볼 수 있었던 유익한 프로젝트였습니다. 노드를 진행하며 궁금했던 부분들에 대해 아래와 같이 정리하였습니다.


**Q. ResNet18 모듈에서의 weight layer는 어떤 구성으로 이루어져 있나요?**

```
ResNet-18와 같은 ResNet의 기본 버전을 구현하려면 Residual Block을 만들어야 합니다.
Residual Block은 ResNet 아키텍처에서 매우 중요한 구성 요소 중 하나이며, 여러 레이어로 구성됩니다.
Residual Block의 각 레이어에 대한 설명은 다음과 같습니다:

- 3x3 CNN (합성곱 레이어):

    이 레이어는 3x3 크기의 커널(필터)을 사용하여 입력 데이터에서 특징을 추출하는 합성곱 레이어입니다.
    이 합성곱 레이어는 일반적으로 ReLU 활성화 함수를 포함하고, 필터 수를 조절하여 특징 맵의 차원을 변경합니다.
    ResNet-18의 경우, 이 레이어는 64개의 필터를 사용할 수 있습니다.

- BatchNormalization (배치 정규화):

    BatchNormalization은 미니배치의 각 데이터에 대해 정규화를 수행하는 레이어입니다. (네트워크의 안정성 향상, 학습 속도 가속화)
    BatchNormalization은 각 미니배치의 평균 및 표준 편차를 계산하고, 이를 사용하여 입력 데이터를 정규화합니다.
    정규화된 데이터는 스케일링 및 이동 파라미터에 따라 조절됩니다.

- Residual Connection (잔여 연결):

    Residual Block의 핵심 요소로, 입력 데이터와 레이어 출력 사이에 추가된 스킵 연결을 의미합니다.
    잔여 연결은 레이어의 입력 데이터를 직접 레이어의 출력에 더함으로써, 레이어가 학습해야 하는 잔여 정보를 전달합니다.
    잔여 연결은 그라디언트 소실 문제를 해결하고, 더 깊은 네트워크를 학습하는데 도움을 줍니다.
    ResNet의 핵심 아이디어 중 하나로, 매우 깊은 네트워크에서도 효과적인 학습을 가능하게 합니다.
```

**Q. ResNet-18와 ResNet-50은 어떤 공통점과 차이점이 있나요?**

```
공통점:

    * Residual Network 구조: 두 모델은 모두 잔여 연결 (Residual Connection)을 사용하여 깊은 신경망을 학습하는 데 중점을 둡니다.  
        잔여 연결은 입력과 출력을 더하는 방식으로 학습 중에 잔여 정보를 전달합니다.  
    * 컨볼루션 레이어: ResNet-18과 ResNet-50 모두 합성곱 레이어를 사용하여 이미지 특징을 추출합니다.  
    * 사전 훈련된 모델: 두 모델은 대규모 데이터셋(예: ImageNet)에서 사전 훈련된 가중치를 가질 수 있으며,  
        전이 학습을 수행할 때 이러한 가중치를 사용할 수 있습니다.  

차이점:

    * 네트워크 깊이: 가장 큰 차이점은 네트워크의 깊이입니다. ResNet-18은 18개의 레이어로 구성되며,  
        ResNet-50은 50개의 레이어로 구성됩니다. 따라서 ResNet-50은 ResNet-18보다 더 깊은 네트워크입니다.  

    * 블록 구성: ResNet-18은 기본 Residual Block을 사용하며, 각 블록에 2개의 합성곱 레이어를 가지고 있습니다.  
        ResNet-50은 보편적으로 "Bottleneck" 구조를 사용하며, 각 블록에 3개의 합성곱 레이어를 가집니다.  
        이 Bottleneck 구조는 계산 효율성을 높이기 위해 설계되었습니다.  

    * 매개변수 양: ResNet-50은 ResNet-18에 비해 훨씬 더 많은 매개변수를 가지고 있습니다.  
        따라서 ResNet-50은 더 많은 학습 가능한 가중치를 가지며, 더 복잡한 특징을 학습할 수 있습니다.  

    * 계산 복잡성: ResNet-50은 ResNet-18에 비해 계산 복잡성이 높습니다. 따라서 더 많은 컴퓨팅 리소스와 메모리를 필요로 합니다.  

일반적으로 ResNet-50은 ResNet-18에 비해 더 깊고 성능이 우수한 모델로 간주됩니다.  
그러나 모델 선택은 작업에 따라 다를 수 있으며, 모델의 깊이 및 계산 복잡성을 고려하여 선택해야 합니다.
```

### Reference

* [CNN, Convolutional Neural Network](http://taewan.kim/post/cnn/)
* [Rethinking the Usage of Batch Normalization and Dropout in the Training of Deep Neural Networks](https://arxiv.org/pdf/1905.05928.pdf)
* [딥러닝에서 클래스 불균형을 다루는 방법](https://3months.tistory.com/414)
* [Image Classification using CNN](https://www.kaggle.com/code/arbazkhan971/image-classification-using-cnn-94-accuracy)
* [Resnet18: Detection of Pneumonia in X-ray](https://www.kaggle.com/code/nischallal/resnet18-detection-of-pneumonia-in-x-ray)
* [Choosing the right backbone size resnet18, resnet34, resnet50](https://forums.fast.ai/t/choosing-the-right-backbone-size-resnet18-resnet34-resnet50/103750)
* [Detailed Guide to Understand and Implement ResNets](https://cv-tricks.com/keras/understand-implement-resnets/)
* [ResNet-50: The Basics and a Quick Tutorial](https://datagen.tech/guides/computer-vision/resnet-50/)

