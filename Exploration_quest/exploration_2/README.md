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

## AIFFEL Campus Online Code Peer Review Templete
- 코더 : 김연
- 리뷰어 : 강임구


## PRT(Peer Review Template)
- [v]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부
<br/>
-Abstractive 모델 구성을 위한 텍스트 전처리 단계가 체계적으로 진행되었다.<br/>

```
텍스트 정규화
불용어 처리(stopwords)
null 제거
데이터셋 분리(train/test)
등등 진행되었음
```
<br/>
- 텍스트 요약모델이 성공적으로 학습되었음을 확인하였다.
```
train / val loss 그래프가 수렴하는 모습을 첨부하였음
xp2_retrospect.ipynb 파일에 원문, 핵심 키워드, 실제 요약, 예측 요약을 정리하였음
```
<br/>
- Extractive 요약을 시도해보고 Abstractive 요약 결과와 함께 비교해 보았다.
```
xp2_retrospect.ipynb 파일에 원문, 핵심 키워드, 실제 요약, 예측 요약을 정리하였음
```
<br/>
- [v]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

```
많아서 첨부는 생략하겠습니다.
xp2_news_summary.ipynb 파일에 각 진행 단계마다 설명을 달아놓아
이해가 따라올 수 있게 하였습니다
```
        
- [v]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

```
summa의 keywords 사용하여 추출적요약 시도
xp2_retrospect.ipynb 참조
```


        
- [v]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

```
많은 결과가 출력이 되는데, 기존의 요약과는 다른 요약을 출력하면서도 원문의 내용을 담고 있는 의미 있는 요약들이 보입니다. 심지어 일부 요약의 경우에는 원문에 없던 단어를 사용해서 요약을 하기도 하고 있습니다. 워드 임베딩과 RNN의 콜라보로 이뤄낸 신기한 성과네요!

물론 좋지 않은 요약의 예도 꽤나 보입니다. 성능을 개선하기 위해서는 seq2seq와 어텐션의 자체의 조합을 좀 더 좋게 수정하는 방법도 있고, 빔 서치(beam search), 사전 훈련된 워드 임베딩(pre-trained word embedding), 또는 인코더 - 디코더 자체의 구조를 새로이 변경한 하는 트랜스포머(Transformer)와 같은 여러 개선 방안들이 존재합니다. 이런 방안들에 대해서도 향후 살펴보겠습니다.
```
```
xp2_retrospect.ipynb 참조
```
        
- [v]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

```

data = data[data.apply(lambda x: len(x['text'].split()) <= text_max_len and len(x['headlines'].split()) <= headlines_max_len, axis = 1)]

```





## 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```

