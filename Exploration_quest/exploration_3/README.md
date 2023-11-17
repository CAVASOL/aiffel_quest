## 6-1. 프로젝트: 인물 모드 문제점 찾기

| 평가문항  | 상세기준 | 
| :--- | :--- | 
| 1. 인물모드 사진을 성공적으로 제작하였다. | 아웃포커싱 효과가 적용된 인물모드 사진과 동물 사진, 배경전환 크로마키사진을 각각 1장 이상 성공적으로 제작하였다. | 
| 2. 제작한 인물모드 사진들에서 나타나는 문제점을 정확히 지적하였다. | 인물사진에서 발생한 문제점을 정확히 지적한 사진을 제출하였다. |   
| 3. 인물모드 사진의 문제점을 개선할 수 있는 솔루션을 적절히 제시하였다. | semantic segmentation mask의 오류를 보완할 수 있는 좋은 솔루션을 이유와 함께 제시하였다. | 

### Details  

인물, 강아지와 고양이, 그리고 자동차 사진에 `OpenCV` 라는 이미지 처리 라이브러리를 이용하여 아웃포커싱 효과를 적용했습니다. 또한 `Semantic Segmentation` 기술을 사용하여 물체를 감지하고 분할한 후, 두 장의 이미지를 합성하여 배경을 제거하고 크로마키 사진을 생성하였습니다. 

### Result  

`xp3_project.ipynb` 파일 하단 참고.  

생성된 사진들을 살펴본 결과 몇 가지 문제점을 발견할 수 있었습니다. 먼저, 물체의 경계 부분이 정확하게 분리되지 않은 경우가 있었고, 이로 인해 합성된 이미지가 자연스럽지 않게 보이는 경우도 있었습니다. 또한, 아웃포커싱 효과가 부자연스럽게 적용된 경우가 있었습니다. 이러한 문제를 해결하기 위해 몇 가지 대안적인 방법을 고려하였습니다. 물체의 경계가 뚜렷하지 않은 문제에 대해서는 더 정교한 Semantic Segmentation 기술을 적용하거나, 추가적인 후처리 과정을 통해 물체의 경계를 더 정확하게 만들어 나갈 수 있을 것입니다. 또한, 아웃포커싱 효과의 자연스러움을 높이기 위해 적절한 블러링 기법을 사용하여 부드러운 전환 효과를 연출할 수 있을 것으로 예상됩니다. 이러한 솔루션들을 통해 실험에서 발생한 문제를 개선하여, 보다 자연스러운 결과물을 얻을 수 있을 것으로 기대됩니다.  

### Retrospect

>이미지 처리 작업에서는 먼저 객체의 종류를 식별하고, 그 객체를 배경으로부터 분리한 후, 배경에 흐릿한 효과를 주고 객체를 합성하는 과정을 거쳤습니다. 이 프로젝트는 흥미로운 동시에 유익한 작업이었습니다.  
>그러나 두 개의 다른 이미지 파일에서 추출한 두 객체를 하나의 배경에 합성하는 과정에서 문제가 발생했습니다. 객체 사이에 거리를 만들기 위해 중간 공간을 넣거나 객체의 위치를 조정하는 등의 여러 시도를 했지만, 결과물이 예상과 다르게 반씩 잘린 객체가 이어붙거나, 검은색 배경이 생성되는 어색한 결과물이 나타났습니다.  
>이를 해결하기 위해 이미지의 크기를 조정하고 위치를 조절하려 했지만, 생각보다 문제가 복잡해졌습니다. 그러나 좀 더 신중하게 접근하여 문제를 해결해보도록 하겠습니다. 이런 문제들을 해결하기 위해서는 객체 간 거리 조절, 위치 조정 등에 대한 정확한 조작이 필요하며, 예상치 못한 결과물이 나타날 수 있는데 이를 보완하기 위해 추가적인 시행착오와 조정이 필요해요.  

### Reference

* [한땀한땀 딥러닝 컴퓨터 비전 백과사전 - SegNet](https://wikidocs.net/148875)
* [Introduction to Semantic Segmentation](https://encord.com/blog/guide-to-semantic-segmentation/)
* [TensorFlow Image segmentation](https://www.tensorflow.org/tutorials/images/segmentation)
* [Open Source Computer Vision - Semantic Segmentation Example](https://docs.opencv.org/4.x/dc/d20/tutorial_js_semantic_segmentation.html)
* [The Beginner’s Guide to Semantic Segmentation](https://www.v7labs.com/blog/semantic-segmentation-guide#h4)
* [Data Cleaning Checklist: How to Prepare Your Machine Learning Data](https://www.v7labs.com/blog/data-cleaning-guide)
* [Top 8 Image-Processing Python Libraries Used in Machine Learning](https://neptune.ai/blog/image-processing-python-libraries-for-machine-learning)
* [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561)
* [Guide to Image Segmentation in Computer Vision: Best Practices](https://encord.com/blog/image-segmentation-for-computer-vision-best-practice-guide/)

---