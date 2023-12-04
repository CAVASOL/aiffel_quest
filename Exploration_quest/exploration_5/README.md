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