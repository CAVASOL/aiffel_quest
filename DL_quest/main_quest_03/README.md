## Main Quest 03. 인공지능과 가위바위보 하기

| 평가문항  | 상세기준 | 
| :--- | :--- | 
| 1. 1. 이미지 분류기 모델이 성공적으로 만들어졌는가? | 학습과정이 정상적으로 수행되었으며, 학습 결과에 대한 그래프를 시각화(ex. train acc / train loss / val acc / val loss 등) 해 보았음 | 
| 2. 오버피팅을 극복하기 위한 적절한 시도가 있었는가? | 오버피팅 극복을 위하여 데이터셋의 다양성, 정규화 등을 2가지 이상 시도해보았음 |   
| 3. 분류모델의 test accuracy가 기준 이상 높게 나왔는가? | 85% 이상 도달하였음 | 

### 목차

```
2. 인공지능과 가위바위보 하기

    2-1) 인공지능과 가위바위보 하기  

    2-2) 데이터 준비  
        1) MNIST 숫자 손글씨 Dataset 불러들이기
        2) 학습용 데이터와 시험용 데이터
        3) 데이터 전처리 하기

    2-3) 딥러닝 네트워크 설계하기  
        1) Sequential Model을 사용해 보자

    2-4) 딥러닝 네트워크 학습시키기  

    2-5) 얼마나 잘 만들었는지 확인하기
        1) 테스트 데이터로 성능을 확인해 보자
        2) 어떤 데이터를 잘못 추론했을까?

    2-6) 더 좋은 네트워크 만들어보기  

3. 미니 프로젝트: 가위바위보 분류기를 만들자

    3-1) 미니 프로젝트: 가위바위보 분류기를 만들자
        1) 라이브러리 버전 확인하기
        2) 데이터 준비하기
        3) 데이터 불러오기 & 이미지 데이터 크기 조정하기
        4) 딥러닝 네트워크 설계하기
        5) 딥러닝 네트워크 학습시키기
        6) 얼마나 잘 만들었는지 확인하기: 테스트
        7) 더 좋은 네트워크 만들어보기

    3-2) test_accuracy를 높이기 위해 설정한 몇 가지 시도
```

### 학습 내용

* 이미 잘 정제된 10개 클래스의 숫자 손글씨 데이터를 분류하는 classifier 만들기
* 정제되지 않은 웹캠 사진으로부터 데이터 만들어보기
* 흑백 사진이 아닌 `컬러 사진`을 학습하는 classifier 만들기
* 분류하고자 하는 클래스의 개수를 마음대로 조절하기 (`10개에서 3개로`)

### 학습 목표

* `MNIST Dataset`와 텐서플로우 `Sequential API`를 이용하여 숫자 손글씨 인식기를 만들어 볼 수 있다.
* `Sequential Model`을 활용하여 딥러닝 네트워크를 설계하고 학습 시킬 수 있다.
* 테스트 데이터로 성능을 확인하고 하이퍼파라미터 조정을 시도하여 성능을 높여볼 수 있다.
* 가위바위보 분류기를 만들기 위한 기본 데이터를 웹캠으로 제작 할 수 있다.

### 3-2. test_accuracy를 높이기 위해 설정한 몇 가지 시도

```
3-2. test_accuracy를 높이기 위해 설정한 몇 가지 시도

    1. 채널 조정 및 더 많은 횟수의 에포크 시도

        n_channel_1=16
        n_channel_2=24 # 32 -> 24
        n_channel_3=32
        n_dense=32
        n_train_epoch=20 # 10 -> 20

    2. BatchNormalization() 추가

        model.add(keras.layers.BatchNormalization())

    3. layer 추가

        model.add(keras.layers.Conv2D(n_channel_3, (2,2), activation='relu'))

    4. Dropout 추가

        model.add(keras.layers.Dropout(0.4))

    5. learning_rate=0.01 설정

        optimizer = keras.optimizers.Adam(learning_rate=0.01)

        model.compile(optimizer=optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])

==============================================

# 더 좋은 네트워크 만들어보기

import tensorflow as tf
from tensorflow import keras

n_channel_1=16
n_channel_2=24 # 32 -> 24
n_channel_3=32
n_dense=32
n_train_epoch=20 # 10 -> 20

model=keras.models.Sequential()
model.add(keras.layers.Conv2D(n_channel_1, (2,2), activation='relu', input_shape=(28, 28, 3)))

# BatchNormalization()) 추가
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Conv2D(n_channel_2, (2,2), activation='relu'))

# BatchNormalization()) 추가
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(2, 2))

# layer 추가
model.add(keras.layers.Conv2D(n_channel_3, (2,2), activation='relu'))

# BatchNormalization()) 추가
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Flatten())

# Dropout 추가
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(n_dense, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

print('Model에 추가된 Layer 개수: ', len(model.layers))

model.summary()

optimizer = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=optimizer,
             loss="sparse_categorical_crossentropy",
             metrics=['accuracy'])

# 모델 학습
history = model.fit(x_train, y_train, epochs=n_train_epoch)

# 모델 시험
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"test_loss: {test_loss} ")
print(f"test_accuracy: {test_accuracy}")
```


### Retrospect

* **금요일(27/10)**  
    600개(나, 강임구님)의 훈련 데이터와 300개(김영진님)의 테스트 데이터를 활용하여 딥러닝 학습을 구현했습니다.  
    600개의 훈련 데이터엣 20%를 검증 데이터로 분리하였고, 데이터 섞기(Shuffle)를 진행한 후 딥러닝 학습을 실행했습니다.  
    채널 조정과 에포크 설정에 초점을 맞춰서 test_accuracy를 올리기 위해 노력했으나, 최대 56% 이상으로 올라가지 않았습니다.  
    optimizer에 initial_learning_rate=1e-2을 설정했습니다.  
    Dropout을 추가하였고, 0.5 설정을 유지했습니다.  

* **토요일(28/10)**  
    2829(그루님들의 이미지 데이터)개의 훈련데이터와 300(LMS Test data)개의 테스트 데이터를 활용했습니다.  
    2829개의 훈련 데이터에서 20%를 검증 데이터로 분리하였고, 데이터 섞기(Shuffle)를 진행한 후 딥러닝 학습을 실행했습니다.  
    채널 조정과 에포크 설정에 초점을 맞춰서 test_accuracy를 올리기 위해 노력했으나, 최대 54% 이상으로 올라가지 않았습니다.  
    optimizer에 initial_learning_rate=1e-2을 설정했습니다.  
    Dropout을 추가하였고, 0.5 설정을 유지했습니다.  
    데이터의 양을 대폭 늘렸으나, 오히려 성능이 저조하게 나타나는 것을 확인할 수 있었습니다.  
    데이터의 양도 중요하지만 데이터 자체의 질(quality)도 중요하다고 생각했습니다.  

* **일요일(29/10)**  
    훈련 데이터와 테스트 데이터의 수(2829 -> 300, 300 -> 30)를 조정했고, 이미지 데이터를 교체했습니다.  
    검증 데이터 분리나 셔플 코드를 진행하지 않았습니다.  
    채널과 에포크를 조정했고, layer를 하나 더 추가했습니다.  
    BatchNormalization()을 추가했습니다.  
    Dropout 추가했고, learning_rate를 0.01로 설정했습니다.  
    테스트 데이터를 매우 여러번 교체하였음에도  test_accuracy의 성능은 최대 66% 정도였습니다.  
    노이즈가 적은 데이터로 양을 대폭 늘리거나 줄이는 시도를 했고, 레이어의 채널 조정을 다양하게 설정하여 결과를 확인했습니다.  

>이미지 데이터 전처리 과정과 하이퍼파라미터를 활용한 성능 올리기에 대해 더 많은 연습이 필요하다고 느꼈습니다. 노이즈가 적은 대량의 데이터를 활용했다면 어떤 결과가 나타났을까 궁금했고, 한편 컬러 이미지를 활용하여 딥러닝 학습을 구현하는 것은 사실 매우 까다로운 작업이구나 라고 느꼈습니다. 제가 직접 만든 이미지 데이터를 활용하여 딥러닝 학습을 구현하고 다양하게 실험할 수 있어서 유익했습니다. 비록 test_accuracy 85% 이상이라는 결과를 얻지 못하였지만, 딥러닝 학습 구현을 위해 어떤 초기 설정이 필요한지, 어떤 데이터가 요구되는지, 데이터 전처리를 위해 어떤 과정이 필요한지 학습할 수 있었습니다. 컬러 이미지를 활용하여 딥러닝 학습을 구현하는 과정이 굉장히 흥미로웠습니다. 이번 프로젝트를 진행하며 개인적으로 궁금하고 학습이 필요했던 부분에 대해 아래와 같이 정리하였습니다.


**Q. 딥러닝 모델의 성능을 높일 수 있는 몇 가지 사항**

```
1. 데이터 전처리: 입력 데이터를 정규화하세요. 현재 코드에서는 입력 이미지의 픽셀 값이 0에서 255 범위에 있을 것으로 가정하고 있지만,  
    이미지 픽셀 값을 0에서 1로 스케일링하면 성능이 향상될 수 있습니다.  

2. 데이터 증강 (Data Augmentation): 이미지 데이터가 부족할 경우, 데이터 증강 기술을 사용하여 데이터셋을 확장하세요.  
    데이터 증강은 회전, 확대/축소, 가로 뒤집기 등을 포함합니다. 이는 모델의 일반화 능력을 향상시키는 데 도움이 됩니다.  

3. 학습률 스케줄링 (Learning Rate Scheduling): 학습률을 고정값으로 두지 말고 학습률을 동적으로 조절할 수 있도록 학습률 스케줄링을 적용하세요.  
    이것은 학습 과정을 최적화하고 빠른 수렴을 돕는 데 도움이 됩니다.  

4. Batch Normalization: 각 레이어 뒤에 Batch Normalization 레이어를 추가하면 학습이 안정화되고, 더 빠른 수렴을 이룰 수 있습니다.  

5. 더 깊은 네트워크 구조: 모델의 레이어 수를 늘려보거나, 더 복잡한 신경망 아키텍처를 시도하여 모델의 표현 능력을 높일 수 있습니다.  

6. 더 많은 데이터 수집: 가능하다면 더 많은 데이터를 수집하여 모델을 더 잘 교육시키세요.  
    더 다양한 조건에서 다양한 손 모양을 포함하는 데이터가 모델의 성능 향상에 도움이 됩니다.  

7. 검증 데이터: 학습 데이터를 학습과 검증 세트로 나누고, 모델의 과적합을 모니터링하십시오.  
    필요한 경우, 조기 종료 (early stopping)와 같은 정규화 기술을 사용하여 과적합을 줄이세요.  

8. 다양한 하이퍼파라미터 튜닝: 다양한 하이퍼파라미터 (예: 레이어 크기, 드롭아웃 비율, 배치 크기, 학습률)를 시도하여 최상의 모델 구성을 찾아보세요.  

9. Transfer Learning: 사전 훈련된 모델을 사용하여 전이 학습을 시도해 보세요.  
    대규모 데이터셋에서 사전 훈련된 모델을 사용하면 모델의 초기 가중치가 향상될 수 있습니다.  
```

**Q. 컬러 이미지를 활용한 딥러닝 학습 구현 과정에서 이미지 데이터 전처리는 어떻게 해야 하나?**

```
1. 이미지 크기 조정 (Resizing): 모든 입력 이미지를 일정한 크기로 조정해야 합니다.  
    대부분의 딥러닝 모델은 고정된 입력 크기를 요구하며, 일반적으로는 정사각형 형태로 조정합니다.    
    일반적으로 사용되는 크기는 224x224, 128x128, 또는 32x32 등이 있습니다.  

2. 데이터 정규화 (Normalization): 이미지 데이터의 각 픽셀 값을 0에서 1 사이의 값으로 정규화합니다.  
    이는 모델의 수렴 속도를 높이고 안정성을 향상시키는 데 도움이 됩니다.  
    정규화는 RGB 이미지의 경우 각 채널에 대해 0에서 255로 나누는 방식으로 수행합니다.  

3. 데이터 증강 (Data Augmentation, 선택적): 데이터 증강은 모델의 일반화 성능을 향상시키기 위해 사용될 수 있습니다.  
    이는 이미지 회전, 반전, 확대/축소, 밝기 조절 등을 포함할 수 있습니다. 데이터 증강은 모델이 더 다양한 상황에서 학습하도록 돕는데 사용됩니다.  

4. 이미지 포맷 변환 (Channel Ordering, 선택적): 일부 딥러닝 라이브러리 및 모델은 이미지의 채널 순서를 다르게 요구할 수 있습니다.  
    일반적으로 사용되는 두 가지 순서는 "RGB" (Red-Green-Blue)와 "BGR" (Blue-Green-Red)입니다.  
    모델의 요구사항에 따라 적절한 포맷을 선택합니다.  

5. 레이블 인코딩: 클래스 레이블을 숫자로 인코딩해야 합니다. 일반적으로 원-핫 인코딩 또는 정수 인코딩을 사용합니다.  

6. 데이터 분할 (Data Split): 데이터를 학습, 검증 및 테스트 세트로 나누는 것이 중요합니다.  
    학습 데이터로 모델을 훈련하고, 검증 데이터로 하이퍼파라미터 튜닝 및 모델 성능 평가, 테스트 데이터로 모델의 최종 성능 평가를 수행합니다.  
```