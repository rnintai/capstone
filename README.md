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
  > 
  > label을 one hot encoding 벡터로 정리  
  > ![image](https://user-images.githubusercontent.com/65759076/113110706-58307680-9242-11eb-9f00-bb2abb4fdf59.png)  
  > 
  > modeling을 하기 위해 4D Array로 변환.  
  > 
  > ![image](https://user-images.githubusercontent.com/65759076/113011396-b9aa0400-91b4-11eb-8c48-1eab30a9a39f.png)  
 
2. 모델 구성  
  > - 모델 구성  
  > ![image](https://user-images.githubusercontent.com/65759076/113010845-207aed80-91b4-11eb-83b8-8c38c84b23d9.png)  
  > ![image](https://user-images.githubusercontent.com/65759076/113115676-a3995380-9247-11eb-9969-c116f9b964ad.png)  
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


