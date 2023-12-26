## 3-1. 프로젝트: 베트남 음식 분류하기

| 평가문항  | 상세기준 | 
| :--- | :--- | 
| 1. 주어진 함수들과 TensorFlow Data API를 이용해서 dataloader를 구현했다 | dataloader를 만드는 과정이 에러 없이 수행되었다 | 
| 2. 베트남 음식 사진 분류를 위한 모델을 구현했다 | EfficientNetB0 backbone과 Dense 레이어를 결합하여 모델을 성공적으로 구현했다 |   
| 3. TensorFlow GradientTape을 이용해서 custom trainer 클래스를 구현했다 | 학습이 진행되면서 training accuracy가 점차 증가하였다 | 

**Details**

Design a `Dataloader`, `Model`, and `Custom Trainer` using the `30VNFoods` dataset, then train and test the model.

**Result** 

`30vnfoods.ipynb` 파일 참고.  

**Retrospect**

> 딥러닝 이미지 분류 작업과 관련한 그 동안의 과제들을 자세한 안내와 함께 복습할 수 있어서 유익했습니다. 또한 트레이너를 커스터마이징하여 디자인해볼 수 있었고, .py파일에서 모델을 설계하고 훈련하는 방법을 알게되어 기쁩니다. 