## Exploration 6

### Index

1. Create Image with ControlNet
2. Deploy Model with TKserving and Docker

### 1. Create Image with ControlNet

| 평가문항  | 상세기준 | 
| :--- | :--- | 
| 1. 프롬프트와 하이퍼파라미터를 변경하여 윤곽선 검출 조건을 준 이미지를 생성하였는가? | ChatGPT를 사용하여 프롬프트를 작성하고, 하이퍼파라미터를 조절하여 윤곽선 검출 조건을 준 이미지를 생성하였다. | 
| 2. 인체 자세 검출 전처리기를 이용하여 이미지를 생성하였는가? | 인체 자세 검출 전처리기를 사용하여 이미지를 생성하였다. |   
| 3. 윤곽선 검출과 인체 자세 검출 전처리기를 사용하여 이미지를 생성하였는가? | Canny 알고리즘을 사용한 윤곽선 검출 전처리기와 Openpose 전처리기를 같이 사용하여 이미지를 생성하였다. | 

**Details**

Create individual images using `ControlNet`'s OpenPose and Canny, and then `create an image` by `mixing OpenPose and Canny` again.

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

Mission to `select one of the projects`, `finetune hyperparameters`, and deploy it with `TFServing & Docker`.

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