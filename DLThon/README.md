## DLThon

| 평가문항  | 상세기준 | 
| :--- | :--- | 
| 1. 적합한 로스와 메트릭을 사용하여 훈련이 이루어졌는가? | 데이터셋 구성, 모델 훈련, 결과물 시각화의 한 사이클이 정상적으로 수행되어 테스트 결과를 출력하였다. | 
| 2. 두 가지 이상의 차이점을 두어 비교가 이루어졌는가? | 선택한 모델의 훈련에 필요한 하이퍼 파라메터들의 수치별 성능과 비용의 비교분석을 진행하였다. |   
| 3. 훈련 결과 및 제품화 가능성에 대한 탐색이 이루어졌는가? | 데이터과 모델에 대한 구성 및 훈련 비용과 성능, 모델을 제품화한다면 응용분야와 강점 및 개선사항 대해 정량적, 정성적 분석을 진행하였다. | 

### Details  

`해파리의 종을 구분`하기 위한 `이미지 데이터셋`은 여러 종류의 해파리 이미지로 구성되어 있습니다. 해파리는 다른 동물들과 마찬가지로 동일한 종에 속해 있어도 크기나 형태가 매우 다양합니다. 이러한 이유로 전문적인 지식 없이는 종을 구분하기가 어렵습니다. 이 데이터셋은 동일한 해파리 강에 속하지만 서로 다른 종들이 혼합되어 있습니다. 이러한 작업을 발전시킨다면, 예를 들어 은행나무의 암수 구분과 같은 전문적인 분류 작업을 더욱 효율적으로 수행할 수 있을 것입니다. 딥러닝 모델을 활용하여 해파리 이미지를 입력받아 각각의 클래스로 `분류`해보겠습니다.

### Result  

`dlthon_jellyfish.ipynb` 파일 참고.  

### Retrospect

>6개의 클래스로 구성된 해파리 이미지 데이터셋을 사용하여 학습 데이터, 검증 데이터, 테스트 데이터로 분리한 후, 이미지 데이터의 크기를 균일하게 조정한 다음 학습을 진행했습니다. 모델 최적화를 위해 다양한 모듈을 바탕으로 디자인한 모델을 적용하며 정확도 지표나 손실 등의 성능을 관찰했습니다. 이미지 데이터 전처리 과정이 예상보다 어려웠고, 이 과정에서 시간을 많이 할애해야 했습니다. 데이터 유형에 따라 적절하고 필요한 전처리 과정이 무엇인지 신중하게 생각하고 적용해 볼 필요가 있었습니다. 또한 이미지 데이터의 특성을 고려하여 어떤 방식의 분석 방법이 적절한가에 대해서도 신중하게 접근하는 것이 중요했습니다. 이미지 데이터를 활용한 분석 작업은 데이터에 대한 충분한 이해가 중요함을 상기할 수 있었던 프로젝트였습니다. 짧은 시간 동안 그루님들과 다양한 모델을 적용하여 실험하고, 지표와 결과를를 관찰하며 성능 최적화에 대한 고민을 함께 풀어보고, 프로젝트를 참여하며 느꼈던 각자의 고민과 생각을 공유할 수 있었습니다. 개인적으로 노드 학습을 진행하며 느꼈던 고민들을 그루님들과 이야기하며 해소할 수 있어서 유익했습니다.

### Reference

* [Jellyfish Image - Classification](https://www.kaggle.com/code/xokent/jellyfish-image-classification)
* [jellyfish-types using TensorFlow](https://www.kaggle.com/code/mansi0123/jellyfish-types-using-tensorflow)
* [Hugging Face - google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)
* [Metatext - google vit base patch16-224-in21k model](https://metatext.io/models/google-vit-base-patch16-224-in21k)
* [DenseNet-121 - Jellyfish Image Classification](https://www.kaggle.com/code/marquis03/densenet-121-jellyfish-image-classification)
* [Object Detection for JellyFish using small dataset and RetinaNet](https://medium.com/@yhoso/detecting-jellyfish-using-openimagedata-and-keras-retinanet-77afca4e7b4f)
* [Image Classification with Hugging Face Transformers and `Keras`](https://www.philschmid.de/image-classification-huggingface-transformers-keras)
* [A complete Hugging Face tutorial: how to build and train a vision transformer](https://theaisummer.com/hugging-face-vit/)