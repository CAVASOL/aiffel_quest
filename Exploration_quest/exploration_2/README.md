## 4-1. 프로젝트: 뉴스기사 요약해보기

| 평가문항  | 상세기준 | 
| :--- | :--- | 
| 1. Abstractive 모델 구성을 위한 텍스트 전처리 단계가 체계적으로 진행되었다. | 분석단계, 정제단계, 정규화와 불용어 제거, 데이터셋 분리, 인코딩 과정이 빠짐없이 체계적으로 진행되었다. | 
| 2. 텍스트 요약모델이 성공적으로 학습되었음을 확인하였다. | 모델 학습이 진행되면서 train loss와 validation loss가 감소하는 경향을 그래프를 통해 확인했으며, 실제 요약문에 있는 핵심 단어들이 요약 문장 안에 포함되었다. |   
| 3. Extractive 요약을 시도해 보고 Abstractive 요약 결과과 함께 비교해 보았다. | 두 요약 결과를 문법완성도 측면과 핵심단어 포함 측면으로 나누어 비교하고 분석 결과를 표로 정리하여 제시하였다. | 

### Details

뉴스 기사 데이터인 [news_summary_more.csv](https://github.com/sunnysai12345/News_Summary)를 활용하여 `Extractive/Abstractive summarization`을 이해하기 위해 텍스트 정규화 기법을 적용했습니다. 이를 위해 다양한 방법으로 단어장의 크기를 줄였고, `seq2seq` 모델을 디자인했습니다. 또한 `Bahdanau 스타일의 어텐션 메커니즘`을 적용하여 모델의 성능을 향상시켰습니다. 이후, `Summa`의 `summarize` 기능을 활용하여 추출적 요약을 수행했습니다. 실험의 과정과 결과와 관련한 회고는 `xp2_retrospect.ipynb` 파일에 정리했습니다.

### Result

`xp2_retrospect.ipynb` 파일 참고.

### Retrospect

>영어 원문으로 작성된 뉴스 기사를 데이터로 활용하여 추상적 요약 및 추출적 요약을 실험했습니다. 개인적으로 평소 영어로 작성된 고객 리뷰와 같은 자연어는 어떻게 분석하면 좋을지 실험 과정과 퍼포먼스에 대해 궁금했는데 이번 실험을 통해 자연어 처리 과정을 익힐 수 있어서 유익했습니다. 모델을 디자인하는 과정이 어렵게 느껴졌지만 자연어 처리 과정의 전체적인 순서를 알아볼 수 있었고, 각 과정에서 어떤 퍼포먼스가 기대되는지 그리고 무엇이 중요한지 알 수 있었습니다. 실험한 모델의 학습 결과를 살펴보면 스펠링 오타나 난해한 단어가 등장하는 것을 확인할 수 있었고, 키워드 추출을 통해 요약으로 나타난 결과가 원본 텍스트에 얼마가 근사한지 퍼센트와 같은 숫자로 확인할 수 있다면 좋겠다고 생각했습니다. 동일한 데이터를 바탕으로 Transformer나 Hugging Face Transformers를 활용하여 추출적 요약을 진행한다면 어떤 결과가 나타날지 궁금합니다. 일상 생활에서 편리하게 사용하는 다양한 쳇봇 서비스나 번역 서비스 등은 고도화된 기술로서 우리 삶의 정교한 존재로 한 부분을 차지하고 있습니다. 비록 매우 다듬어진 상태의 깨끗한 데이터를 활용하였지만, 실시간 데이터를 활용한 분석을 얼마나 복잡하고 정교할지 감히 상상해봅니다.

### Reference

* [Load text](https://www.tensorflow.org/tutorials/load_data/text)
* [Attention layer](https://keras.io/api/layers/attention_layers/attention/)
* [KerasNLP Workflows](https://keras.io/keras_nlp/)
* [Getting Started with KerasNLP](https://keras.io/guides/keras_nlp/getting_started/)
* [Text classification with Transformer](https://keras.io/examples/nlp/text_classification_with_transformer/)
* [Abstractive Summarization with Hugging Face Transformers](https://keras.io/examples/nlp/t5_hf_summarization/)


---


