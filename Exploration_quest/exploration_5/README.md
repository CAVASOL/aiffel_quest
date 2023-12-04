## 10-1. 프로젝트: 한국어 데이터로 챗봇 만들기

| 평가문항  | 상세기준 | 
| :--- | :--- | 
| 1. 한국어 전처리를 통해 학습 데이터셋을 구축하였다. | 공백과 특수문자 처리, 토크나이징, 병렬데이터 구축의 과정이 적절히 진행되었다. | 
| 2. 트랜스포머 모델을 구현하여 한국어 챗봇 모델 학습을 정상적으로 진행하였다. | 구현한 트랜스포머 모델이 한국어 병렬 데이터 학습 시 안정적으로 수렴하였다. |   
| 3. 한국어 입력문장에 대해 한국어로 답변하는 함수를 구현하였다. | 한국어 입력문장에 맥락에 맞는 한국어로 답변을 리턴하였다. | 

### Details  

한국어 챗봇을 위해 영어 챗봇 모델을 한국어 데이터로 변환하는 실험의 과정은 다음과 같습니다. 한국어 챗봇 데이터를 수집하여 전처리하고, `SubwordTextEncoder`를 활용하여 토큰화합니다. 이후, 트랜스포머 모델을 구현하고 학습시킨 후, 적절한 예측 함수를 통해 입력된 문장에 대한 응답을 생성하고 모델을 평가합니다. 이를 통해 한국어에 적합한 챗봇 모델을 구축하고, 테스트하여 결과를 확인했습니다.

### Result  

`xp5_project.ipynb` 파일의 `Conclusion` 참고.  

### Retrospect

> 한국어 쳇봇 데이터셋을 활용하여 전처리하고, 토큰화한 후 트랜스포머 모델을 구현하는 작업이 흥미로웠습니다. 직접 모델을 구현하지는 못 했고, 설계된 모델을 이해하는 것도 어려웠습니다. 멀티 헤드 어텐션의 개념이 새로웠고, 퍼포먼스를 이해하기 위해 다양한 자료를 참고했습니다. 영어 텍스트와 한글 텍스트의 전처리 과정이 어떻게 다른지 실험을 통해 복습할 수 있어서 유익했고, 트랜스포머의 퍼포먼스를 관찰하며 배울 수 있었습니다. 모델 학습 시 손실을 최대한 낮추도록 에폭을 조정한 후 시각화하여 확인했습니다. 손실이 0에 가까울 수록 임의의 질문에 대한 대답이 좀 더 적절하게 도출되는 것을 관찰할 수 있었어요. 한국어 데이터셋을 활용하여 쳇봇을 구현한 다양한 작품들이 PyTorch를 활용했다는 사실을 알게 되었고, PyTorch를 활용한 모델 구성과 검증에 대해 살펴봤습니다. 자연스러운 대화가 가능한 쳇봇을 구현하기 위해서는 정말 세심한 작업이 필요하겠구나 생각해 볼 수 있었어요.

### Reference

**Transformer Chatbot**

* [What is a Deep Learning Chatbot?](https://shanebarker.com/blog/deep-learning-chatbot/)  
* [A Transformer Chatbot Tutorial with TensorFlow 2.0](https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html)  
* [Small Transformer for Korean chatbot (pytorch)](https://www.kaggle.com/code/fleek12/small-transformer-for-korean-chatbot-pytorch)  
* [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)  
* [What are Encoder in Transformers](https://www.scaler.com/topics/nlp/transformer-encoder-decoder/)  
* [Transformers-based Encoder-Decoder Models](https://huggingface.co/blog/encoder-decoder)
* [The Transformer Attention Mechanism](https://machinelearningmastery.com/the-transformer-attention-mechanism/)  
* [Transformer: A Novel Neural Network Architecture for Language Understanding](https://blog.research.google/2017/08/transformer-novel-neural-network.html)
* [How to Implement Scaled Dot-Product Attention from Scratch in TensorFlow and Keras](https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras/)
* [Hugging Face - T5, num_heads](https://huggingface.co/transformers/v3.0.2/model_doc/t5.html)  
* [Transformers Explained Visually (Part 3): Multi-head Attention, deep dive](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853)  
* [TensorFlow - tf.keras.layers.MultiHeadAttention](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention)  
* [TensorFlow - Understanding masking & padding](https://www.tensorflow.org/guide/keras/understanding_masking_and_padding)  
* [TensorFlow - Module: tf.keras.preprocessing.sequence](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence)  
* [TensorFlow - tf.keras.utils.pad_sequences](https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences)  
* [TensorFlow - Neural machine translation with a Transformer and Keras](https://www.tensorflow.org/text/tutorials/transformer)  
* [Cornell Univ. - Cornell Movie-Dialogs Corpus Dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)  
* [Cornell Movie-Dialogs Corpus Dataset Details](https://convokit.cornell.edu/documentation/movie.html)  

**Korean Chatbot**

* [hyunwoongko / kochat](https://github.com/hyunwoongko/kochat)  
* [jyujin39 / Korean_Chatbot_Model](https://github.com/jyujin39/Korean_Chatbot_Model)  
* [kooyunmo / diya-chat](https://github.com/kooyunmo/diya-chat) 

---

## AIFFEL Campus Online Code Peer Review Templete
- 코더 : 김연 그루님
- 리뷰어 : 강임구 그루


## PRT(Peer Review Template)
- [v]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부

1. 한국어 전처리를 통해 학습 데이터셋을 구축하였다.
```
질문과 답변 나누기
Q, A = list(data_df['Q']), list(data_df['A'])

```

```
특수문자, 공백 처리
def preprocess_sentence(sentence):
  sentence = sentence.strip()

  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)

  sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
  sentence = sentence.strip()
  return sentence
```

2. 트랜스포머 모델을 구현하여 한국어 챗봇 모델 학습을 정상적으로 진행하였다.

```
def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

  # Mask for padding in the encoder
  enc_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)

  # Used by the decoder to mask future tokens.
  # A padded mask is also included internally.
  look_ahead_mask = tf.keras.layers.Lambda(
      create_look_ahead_mask,
      output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)

  # Masking the vectors of the encoder in the second attention block
  # Mask for padding in the decoder
  dec_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)

  # Encoder
  enc_outputs = encoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[inputs, enc_padding_mask])

  # Decoder
  dec_outputs = decoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

  # Fully connected layer
  outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

  return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)
```

3. 한국어 입력 문장에 대해 한국어로 답변하는 함수를 구현하였다.

```
'넌 누구야?' 라는 질문을 입력하여
'사람마다 다르겠지만 사귀고 난 후가 좋겠어요.' 라는 답변을 얻었다.
```

```
'오늘 너무 춥다.' 라는 질문을 입력하여
'최근에 이별을 하셨나보군요.' 라는 답변을 얻었다.
```


    
- [v]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.


```
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(Q+A, target_vocab_size=2**13)

print('Question sample after integer encoding: {}'.format(tokenizer.encode(Q[87])))
print('Answer sample after integer encoding: {}'.format(tokenizer.encode(A[87])))
```

```
각 단어에 고유한 정수가 부여된 Vocabulary를 기준으로 단어 시퀀스가 정수 시퀀스로 인코딩된 결과를 확인할 수 있습니다.
위의 결과와 마찬가지로 Q와 A 페어에 대해서 전부 정수 인코딩을 수행합니다.
이와 동시에 문장의 최대 길이를 정하고, 해당 길이로 padding 합니다.
```

        
- [v]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

```
새로운 시도
xp5_recap_pytorch : 파이토치로 새로 시도하였음
```

```
새로운 시도 - 기존 40에서 30으로 조정하였음
MAX_LENGTH = 30

print("Maximum length of sentence:", MAX_LENGTH)

epochs - 노드에서 10 시행한것을 chatbot-> 30으로 시행하여
loss가 줄어든 것을 그래프로 확인하였음

```

        
- [v]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

```
한국어 쳇봇 데이터셋을 활용하여 전처리하고, 토큰화한 후 트랜스포머 모델을 구현하는 작업이 흥미로웠습니다. 직접 모델을 구현하지는 못 했고, 설계된 모델을 이해하는 것도 어려웠습니다. 멀티 헤드 어텐션의 개념이 새로웠고, 퍼포먼스를 이해하기 위해 다양한 자료를 참고했습니다. 영어 텍스트와 한글 텍스트의 전처리 과정이 어떻게 다른지 실험을 통해 복습할 수 있어서 유익했고, 트랜스포머의 퍼포먼스를 관찰하며 배울 수 있었습니다. 모델 학습 시 손실을 최대한 낮추도록 에폭을 조정한 후 시각화하여 확인했습니다. 손실이 0에 가까울 수록 임의의 질문에 대한 대답이 좀 더 적절하게 도출되는 것을 관찰할 수 있었어요. 한국어 데이터셋을 활용하여 쳇봇을 구현한 다양한 작품들이 PyTorch를 활용했다는 사실을 알게 되었고, PyTorch를 활용한 모델 구성과 검증에 대해 살펴봤습니다. 자연스러운 대화가 가능한 쳇봇을 구현하기 위해서는 정말 세심한 작업이 필요하겠구나 생각해 볼 수 있었어요.
```


        
- [v]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

```
def preprocess_sentence(sentence):
  sentence = sentence.strip()

  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)

  sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
  sentence = sentence.strip()
  return sentence

전처리 부분을 함수화 하여 사용
```





## 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```


