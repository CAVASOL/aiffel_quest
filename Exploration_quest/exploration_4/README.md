## 8-1.프로젝트 : CIFAR-10 이미지 생성하기

| 평가문항  | 상세기준 | 
| :--- | :--- | 
| 1. GAN의 두 모델 구조를 통해 이미지를 성공적으로 생성하였다. | 오브젝트 종류를 육안으로 구별할 수 있을 만한 이미지를 생성하였다. | 
| 2. 생성 이미지 시각화 및 학습 그래프를 통해 GAN 학습이 바르게 진행되었음을 입증하였다. | gif를 통해 생성이미지 품질이 서서히 향상되는 것과, fake accuracy가 추세적으로 0.5를 향해 하향하고 있음을 확인하였다. |   
| 3. 추가적인 GAN 모델구조 혹은 학습과정 개선 아이디어를 제안하고 이를 적용하였다. | 모델구조 변경 혹은 학습과정 개선 아이디어를 추가적으로 적용해보았고, 적용 전과 비교하였을 때의 내용을 표와 같은 시각적 자료로 작성하였다. | 

### Details  

FASHION-MNIST 데이터 생성용 `DCGAN 모델구조`를 이용해서 `CIFAR-10` 데이터를 생성하는 모델을 제작했습니다. 이미지 데이터의 형태를 (28, 28, 1)에서 (32, 32, 3)으로 조정했고, 이를 반영하여 생성자와 판별자 모델의 입력 및 출력 형태와 모델 구조를 변경했습니다. 또한, RGB 3채널의 컬러 이미지 시각화 과정과 관련하여 고려해야 할 부분들을 추가하여 작업하였고, 결과를 확인했습니다. 모델 성능을 최적화하고, 선명한 이미지 데이터 결과를 추출하기 위해 가설을 세우고 모델에 적용했습니다. 모델을 재구성한 후 다시 학습을 진행하였으며, 결과를 시각화하여 확인했습니다. 본 실험의 결과와 회고는 다음와 같습니다.

### Result  

`xp4_project.ipynb` 파일의 `Conclusion` 참고. 

| **Initial GAN** | **Recap GAN** | 
| :---: | :---: | 
| <img align="center" alt="result image data for initial generative modeling" src="https://github.com/CAVASOL/aiffel_quest/blob/main/Exploration_quest/exploration_4/xp4_initial.png" width="100%"> | <img align="center" alt="result image data for reconstructed generative modeling" src="https://github.com/CAVASOL/aiffel_quest/blob/main/Exploration_quest/exploration_4/xp4_recap.png" width="100%"> | 
| <img align="center" alt="gif of resulting image data for initial generative modeling" src="https://github.com/CAVASOL/aiffel_quest/blob/main/Exploration_quest/exploration_4/cifar10_init.gif?raw=true" width="100%"> | <img align="center" alt="gif of resulting image data for reconstructed generative modeling" src="https://github.com/CAVASOL/aiffel_quest/blob/main/Exploration_quest/exploration_4/cifar10_recap.gif?raw=true" width="100%"> |  

### Retrospect

>gif 형태로 결과를 생성하는 것은 흥미로운 작업이었고, 결과물을 확인하는 것이 재밌었습니다. 결과 이미지 데이터의 선명도를 높이기 위해 기존의 모델보다 조금 복잡한 형태로 모델을 재구성하는 과정에서 다양한 참고 자료가 필요했어요. 생성자와 판별자 모델에 대한 저의 가설을 적용함에 있어 적절성 여부를 판단할 수 있는 근거가 필요했습니다. dropout() 함수를 적용하므로서 이미지 데이터 추출에 리스크가 발생할 수 있다는 것을 배웠고, 모델 구현에 있어 세심한 접근이 필요하다는 것을 익힐 수 있었습니다. 또한 다양한 컬러 이미지 데이터를 사용하여 실험을 할 때와 흑백 이미지 데이터를 활용할 때 어떤 부분이 달라지는지, 달라져야 하는지 복습할 수 있어서 유익했습니다.

### Reference

* [Kaggle - GAN CIFAR10](https://www.kaggle.com/code/avk256/gan-cifar10)
* [GAN-CIFAR-10.ipynb](https://colab.research.google.com/drive/1r3InSYsSN6BgZdnyCu3vCnpZ1cniKRTJ?usp=sharing)
* [DCGAN on CIFAR-10](https://wandb.ai/sairam6087/dcgan/reports/DCGAN-on-CIFAR-10--Vmlldzo5NjMyOQ)
* [Developing a DCGAN for CIFAR-10 Dataset](https://datahacker.rs/013-developing-a-dcgan-for-cifar-10-dataset/)
* [GANs — Deep Convolutional GANs with CIFAR10 (Part 8)](https://mafda.medium.com/gans-deep-convolutional-gans-with-cifar10-part-8-be881a77e55b)
* [How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)  
* [10 Lessons I Learned Training GANs for one Year](https://towardsdatascience.com/10-lessons-i-learned-training-generative-adversarial-networks-gans-for-a-year-c9071159628)  
* [Tips for Training Stable Generative Adversarial Networks](https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/)  
* [Improved Techniques for Training GANs](https://proceedings.neurips.cc/paper_files/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf)  