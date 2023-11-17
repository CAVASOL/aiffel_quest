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
>두 개의 서로 다른 이미지를 해당 이미지의 배경과 객체를 분리한 후, 새로운 배경 이미지에 합성하는 과정에서 굉장히 다양한 혼선이 있었어요. 두 개의 이미지 사이에 임의의 공간을 삽입하거나, 각 객체에 위치를 설정하여 간격을 넓히려고 계획했으나 그저 간단하게 생각했던 작업이 의외로 어려운 과제가 되었습니다. 각 객체가 반 씩 잘린 모습으로 양 옆에 위치하거나, 각 객체 사이에 매우 어색한 검은색 미지의 공간이 생겨서 이상한 결과물이 발생했거든요. 하지만! 두 개의 이미지를 hconcat()과 vconcat() 을 사용하여 이어붙인 후 semantic segmentation을 실행했고, 배경과의 경계가 명확하지 않았지만 배경과의 합성이 성공적으로 이뤄졌습니다. 두 객체가 서로를 마주 보는 방향으로 다시 작업할 계획입니다. 실험 결과와 관련한 자세한 내용은 xp3_project.ipynb 파일의 중간 부분의 크로마키 합성 과정 내용을 참고해 주세요.

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