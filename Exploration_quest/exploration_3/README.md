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

# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 김연 님
- 리뷰어 : 최현우


# PRT(Peer Review Template)
- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부
    * 데이터 이미지들을 전부 Segmentation & Blur 처리 확인했습니다!
      
- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
    * 문제점을 해결하기 위해 제시한 방안 중 'Gradient boosting' , 'Instance Seg' 부분을 이미지와 서술로 이해하기 쉽게 작성 해주셨습니다 (코드는 길어서 생략!)
        
- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
    * 위에서 처럼 Semanticc Seg에서 놓친 부분을 'Gradient boosting'으로 경계선을 잘 구분할 수 있는 방법에 대해 자세하게 기록 해주셨습니다.
        
- [X]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
   * 더할 나위 없이 교과서처럼 회고를 작성해 주셨습니다.
        
- [X]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
   * 노드에서 길게 배웠던 과정들을 함수로 묶어 훨씬 간결하고 효율적으로 구현 하셨습니다
     ```
      def segment_person_image(img_path):
      img = cv2.imread(img_path)
  
      model_dir = os.getenv('HOME') + '/aiffel/human_segmentation/models'
      model_file = os.path.join(model_dir, 'deeplabv3_xception_tf_dim_ordering_tf_kernels.h5')
      model_url = 'https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5'
      urllib.request.urlretrieve(model_url, model_file)
  
      model = semantic_segmentation()
      model.load_pascalvoc_model(model_file)
      segvalues, output = model.segmentAsPascalvoc(img_path)
  
      LABEL_NAMES = [
          'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
          'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
          'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv']
  
      # colormap[15]
      # array([192, 128, 128])
  
      seg_color = (128, 128, 192)
      seg_map = np.all(output == seg_color, axis=-1)
      img_mask = seg_map.astype(np.uint8) * 255
  
      img_blur = cv2.blur(img, (40, 40))
  
      img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
      img_bg_mask = cv2.bitwise_not(img_mask_color)
      img_bg_blur = cv2.bitwise_and(img_blur, img_bg_mask)
  
      img_concat = np.where(img_mask_color == 255, img, img_bg_blur)
  
      plt.imshow(cv2.cvtColor(img_concat, cv2.COLOR_BGR2RGB))
      plt.show()
     ```


# 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
