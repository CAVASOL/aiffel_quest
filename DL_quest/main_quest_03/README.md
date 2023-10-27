## Main Quest 03. 인공지능과 가위바위보 하기

| 평가문항  | 상세기준 | 
| :--- | :--- | 
| 1. 1. 이미지 분류기 모델이 성공적으로 만들어졌는가? | 학습과정이 정상적으로 수행되었으며, 학습 결과에 대한 그래프를 시각화(ex. train acc / train loss / val acc / val loss 등) 해 보았음 | 
| 2. 오버피팅을 극복하기 위한 적절한 시도가 있었는가? | 오버피팅 극복을 위하여 데이터셋의 다양성, 정규화 등을 2가지 이상 시도해보았음 |   
| 3. 분류모델의 test accuracy가 기준 이상 높게 나왔는가? | 85% 이상 도달하였음 | 

### 학습 목표

* `MNIST Dataset`와 텐서플로우 `Sequential API`를 이용하여 숫자 손글씨 인식기를 만들어 볼 수 있다.
* Sequential Model을 활용하여 딥러닝 네트워크를 설계하고 학습 시킬 수 있다.
* 테스트 데이터로 성능을 확인하고 하이퍼파라미터 조정을 시도하여 성능을 높여볼 수 있다.
* 가위바위보 분류기를 만들기 위한 기본 데이터를 웹캠으로 제작 할 수 있다.

### Retrospect

> 훈련 데이터로서 나와 강임구 그루님의 사진 600개를 업로드 한 후, 훈련 데이터에서 검증 데이터의 비율으 20% 정도로 분리했다. 테스트 데이터는 김영진 그루님의 사진 300개로 활용하였다. text_accuracy 성능을 85% 이상 높이기 위해 다양한 하이퍼파라미터를 시도하였으나 최대 0.57 정도의 값이 나왔다. 직접 제작한 이미지 데이터로 딥러닝 학습 모델을 설계하고 성능을 분석할 수 있어서 흥미로웠다. 하지만 개인적으로 정확도 성능을 높이는 것과 관련하여 이미지 데이터에 대한 전처리 과정에 대한 정보가 더 필요하고, 하이퍼파라미터 조정에 대해 더 연습을 해야겠다고 느꼈다.

```
3-2. test_accuracy를 높이기 위해 설정한 몇 가지 시도

    1. 채널 조정 및 더 많은 횟수의 에포크 시도

    n_channel_1 = 256   -->   n_channel_1 = 32
    n_channel_2 = 512   -->   n_channel_2 = 64
    n_dense = 512       -->   n_dense = 64
    n_train_epoch = 8   -->   n_train_epoch = 20
    2. Dropout 레이어 추가

    model.add(keras.layers.Dropout(0.5))
    3. 학습률 스케줄링 설정

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-7, 
                                                            decay_steps=10000, 
                                                            decay_rate=0.9)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

==============================================

# 더 좋은 네트워크 만들어보기

n_channel_1 = 32
n_channel_2 = 64
n_dense = 64
n_train_epoch = 20

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(n_channel_1, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Conv2D(n_channel_2, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(n_dense, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, 
                                                          decay_steps=10000, 
                                                          decay_rate=0.9)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer,
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# 모델 훈련
model.fit(x_train, y_train, epochs=n_train_epoch)

# 모델 시험
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"test_loss: {test_loss} ")
print(f"test_accuracy: {test_accuracy}")
```