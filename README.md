# capstone
------
## 파일 개수
> train: 576, test: 70

1. 특징값 추출  
  > train set과 test set을 구분하여 특징값 추출.(mfcc)  
  > sr = 16000, n_mfcc= 100, n_fft = 400, hop_length = 160  
  > output list은 (100,40)을 갖게 되는데, 조금 짧은 음원 파일의 경우 후위에 0의 값을 갖는  
  > padding을 주었다.  
  > **1당 0.01125초라고 함. 400을 넣어서라던지(4.5초) 해서 초를 늘리자.**
  > MixMaxScale
  > ![image](https://user-images.githubusercontent.com/65759076/113121745-e8c08400-924d-11eb-9220-2413e9659a7e.png)
  > 
  > label을 one hot encoding 벡터로 정리  
  > ![image](https://user-images.githubusercontent.com/65759076/113120169-43f17700-924c-11eb-965a-954b12bb834d.png)
  > modeling을 하기 위해 4D Array로 변환.  
  > 
  > ![image](https://user-images.githubusercontent.com/65759076/113011396-b9aa0400-91b4-11eb-8c48-1eab30a9a39f.png)  
 
2. 모델 구성  
  > - 모델 구성  
  > ![image](https://user-images.githubusercontent.com/65759076/113010845-207aed80-91b4-11eb-83b8-8c38c84b23d9.png)  
  > ![image](https://user-images.githubusercontent.com/65759076/113118032-02f86300-924a-11eb-9dfc-141753f24d7d.png)
  > 
  > - 진행 상황  
  > ![image](https://user-images.githubusercontent.com/65759076/113011272-98491800-91b4-11eb-813f-30896924f6b9.png)
  > 
  > - EarlyStop이 일어났을 때의 학습, 오차 그래프.  
  > ![image](https://user-images.githubusercontent.com/65759076/113011165-7c457680-91b4-11eb-8e91-c7bcfb9b0a3c.png)  

3. 결과
  > - 경적 소리 input(파일 이름에 -1-이 포함.)  
  > ![image](https://user-images.githubusercontent.com/65759076/113093731-6a52ea80-922b-11eb-8bc5-7e8917a7747b.png)
  > 
  > - 사이렌 소리 input(파일 이름에 -8-이 포함.)  
  > ![image](https://user-images.githubusercontent.com/65759076/113086859-79cb3700-921d-11eb-8d05-dc7156c1ff92.png)

### 0331 현황
> 모델 테스트 진행. pkl파일 불러오기 오류 미해결. 라즈베리파이에서 녹음한 rawdata에 대한  
> 결과를 아두이노에 송신하는 코드 작성해야. CNN 모델에 대한 이론 학습 필요.

### 0411 현황
> 화재경보기 소리 60개 확보함. 이를 함께 전처리하여 모델링 진행.  
> Sample 소리들로는 구분 가능. 마이크 이용하여 테스트 해보아야 함.  
> 
> **0412**
> 마이크 테스트 완료.
> 
> 문제점: 마이크로 측정된 소리에 앞뒤 padding이 없으면 다른 소리랑 헷갈려한다. 즉 padding 추가한 데이터도 학습시켜야.  
> 마이크로 녹음 및 소리 구분 가능. 하지만 분류하고자 한 소리 이외의 소리들의 처리는 ?  
> 현재는 소리 크기의 TH를 잡고 조용하면 모델 분류 x  
