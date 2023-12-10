## Exploration 6

### Index

1. Create Image with ControlNet - `xp6_cotrolnet.ipynb`
2. Deploy Model with TKserving and Docker - `xp6_deploy_project.ipynb`

### 1. Create Image with ControlNet

| 평가문항  | 상세기준 | 
| :--- | :--- | 
| 1. 프롬프트와 하이퍼파라미터를 변경하여 윤곽선 검출 조건을 준 이미지를 생성하였는가? | ChatGPT를 사용하여 프롬프트를 작성하고, 하이퍼파라미터를 조절하여 윤곽선 검출 조건을 준 이미지를 생성하였다. | 
| 2. 인체 자세 검출 전처리기를 이용하여 이미지를 생성하였는가? | 인체 자세 검출 전처리기를 사용하여 이미지를 생성하였다. |   
| 3. 윤곽선 검출과 인체 자세 검출 전처리기를 사용하여 이미지를 생성하였는가? | Canny 알고리즘을 사용한 윤곽선 검출 전처리기와 Openpose 전처리기를 같이 사용하여 이미지를 생성하였다. | 

**Details**

`Utilize OpenPose & Canny` to create distinct individual images. Merge them for a new visual experience, blending human actions with edge-detected forms to produce unique, intriguing visuals.

**Result** 

`xp6_controlnet.ipynb` 파일의 `Conclusion` 참고.  

**Retrospect**

> OpenPose와 Canny를 활용하여 이미지를 생성하는 작업은 흥미로웠습니다. 입력된 프롬프트의 범위에서 벗어하는 기형의 형상이 도출되긴 했지만 결과적으로는 재미있는 작업이었습니다. 블로그나 르포에 업데이트된 다른 엔지니어들의 작업 내용을 알 수 있어서 좋았고, 새로운 기술을 배울 수 있어서 유익했습니다. 

**Reference**

* [Mikubill / sd-webui-controlnet / OpenPose + Canny to create image #1853](https://github.com/Mikubill/sd-webui-controlnet/discussions/1853)
* [Reddit / Controlnet 1.1 (Openpose fullbody+Canny): Can be a lot of fun](https://www.reddit.com/r/StableDiffusion/comments/12nnwo7/controlnet_11_openpose_fullbodycanny_can_be_a_lot/)
* [멀티 컨트롤넷으로 캐릭터와 배경을 합성해 보기 StableDiffusion WebUi](https://blog.naver.com/PostView.naver?blogId=dk3dcg&logNo=223039145360)
* [How to Control Generated Images by Diffusion Models via ControlNet in Python](https://thepythoncode.com/article/control-generated-images-with-controlnet-with-huggingface)

### 2. Deploy Model with TKserving and Docker

| 평가문항  | 상세기준 | 
| :--- | :--- | 
| 1. KerasTuner로 하이퍼파라미터 튜닝을 진행하였는가? | KerasTuner를 정상작동하였고 하이퍼파라미터 튜닝을 제대로 수행했다. 또한 최적화된 하이퍼파라미터로 모델 학습을 진행했다. | 
| 2. TensorFlow Serving을 사용해서 API를 만들어냈는가? | TFServing을 활용해 모델을 배포했으며 결과값을 캡쳐해 주피터노트북으로 제출했다. |   
| 3. TFLite로 변환했고 파일을 저장하였는가? | TFLite 파일을 만들었으며 해당 파일을 깃헙 레포에 넣어놓았다. | 

**Details**

`The goal involves choosing a project, adjusting its parameters precisely, and deploying it using TFServing and Docker`. The focus is on optimizing model performance by finetuning parameters and ensuring seamless deployment for efficient functionality within a Docker container.

**Result** 

`xp6_deploy_project.ipynb` 파일의 `Conclusion` 참고.  

**Retrospect**

> 개인적으로 Heroku나 Vercel을 활용하는 편이라 전체적인 작업이 조금 번거롭게 느껴졌습니다. Docker와 TFServing을 활용해본 것이 유의미했고, 로컬 환경에서 다양한 방법으로 배포를 시도해본 것이 유익했습니다. 작업 과정에서 ML 모델을 배포함에 있어 TKServing과 Docker를 사용하는 것이 일반적으로 사용하는 방법인지 and/or 엔지니어들이 선호하는 방식인지 궁금했습니다. container의 개념이 흥미로웠고 한편, container나 deploy의 원리가 확립되지 않은 상태에서 단순히 플로우를 경험해보는 것은 무의미하다는 생각을 했습니다. 과정에서 다양한 참고 자료가 필요했고, 참고한 자료는 아래와 같습니다.

**Reference**

* [TensorFlow Serving with Docke](https://www.tensorflow.org/tfx/serving/docker)
* [Hosting Models with TF Serving on Docker](https://towardsdatascience.com/hosting-models-with-tf-serving-on-docker-aceff9fbf533)
* [Serving ML Quickly with TensorFlow Serving and Docker](https://medium.com/tensorflow/serving-ml-quickly-with-tensorflow-serving-and-docker-7df7094aa008)
* [How to Serve Machine Learning Models With TensorFlow Serving and Docker](https://neptune.ai/blog/how-to-serve-machine-learning-models-with-tensorflow-serving-and-docker)

---

# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 김연 님
- 리뷰어 : 김대선
전체적으로 코드가 보기가 쉽고 간결하게 작성이 잘 되었습니다. 프롬프트와 코드 부분에서 보고 많이 배움이 되었습니다.


# PRT(Peer Review Template)
- [V]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부
      <img width="907" alt="스크린샷 2023-12-06 오후 6 13 34" src="https://github.com/CAVASOL/aiffel_quest/assets/144193737/1a083fe1-5cbd-4010-accf-311b91ca4d90">

    

- [V]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
        - <img width="907" alt="스크린샷 2023-12-06 오후 6 13 34" src="https://github.com/CAVASOL/aiffel_quest/assets/144193737/001136c0-c364-44cf-8002-d6c42cd42cd6">

    

- [V]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
          <img width="905" alt="스크린샷 2023-12-06 오후 6 09 01" src="https://github.com/CAVASOL/aiffel_quest/assets/144193737/65164701-8ae0-4e72-b4de-58493a10f486">

    
    
- [V]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
        - 상세한 회고로 배울점이 많았습니다.
        <img width="1025" alt="스크린샷 2023-12-06 오후 6 09 37" src="https://github.com/CAVASOL/aiffel_quest/assets/144193737/ad481eec-e7ae-4430-900a-db3b8a1ce811">


- [V]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
	<img width="911" alt="스크린샷 2023-12-06 오후 6 23 29" src="https://github.com/CAVASOL/aiffel_quest/assets/144193737/1f3ebe5f-fc31-4928-a090-8a16cd3cb927">

# 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
