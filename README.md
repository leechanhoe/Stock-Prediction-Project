# CNN & 픽셀 차트 이미지를 활용한 주가 예측 프로젝트

"비슷한 차트 검색기 - 주가 예측" 프로젝트의 초기 주제인 딥러닝 주가예측 프로젝트입니다.

<br>
<br>
<br>

1. [머신러닝/딥러닝을 활용한 주가 예측시 한계](#1-머신러닝딥러닝을-활용한-주가-예측시-한계)
2. [목표 변경](#2-목표-변경)
3. [모델에 사용한 이미지](#3-모델에-사용한-이미지)
4. [학습 데이터 선별](#4-학습-데이터-선별)
5. [CNN 구조](#5-cnn-구조)
6. [독특한 결과값 조정](#6-독특한-결과값-조정)
7. [앙상블](#7-앙상블)
8. [성능](#성능)
9. [결론](#결론)

<br>
<br>
<br>
<br>
<br>
<br>

## 1. 머신러닝/딥러닝을 활용한 주가 예측시 한계

![](https://velog.velcdn.com/images/dodo4723/post/4eb19d1c-dc05-4015-b33b-657b4645fbdc/image.jpeg)


평범한 머신러닝/딥러닝 모델은 다음날의 주가 예측시 바로 전날의 종가로 예측하는 경향이 있습니다. LSTM과 LGBM을 활용한 주가 예측에서도 위 문제를 피하기 힘들었습니다.

이 문제를 해결하기 위해선 창의적인 접근법이 필요하다고 생각하였습니다.

<br>
<br>
<br>

## 2. 목표 변경

기존 주가 예측이라는 목표에서 `한국주식 전 종목(약 2600개)중 향후 상승(하락) 확률이 높은 종목을 추출하여 상승(하락) 여부를 맞추는 것`을 목표로 설정하였습니다.

<br>
<br>

## 3. 모델에 사용한 이미지

차트 96x96, 148x96 픽셀 이미지로 표현

![](https://velog.velcdn.com/images/dodo4723/post/a8d9d5ec-70cc-4556-8839-aaa726fb60b2/image.png)

- 하나의 캔들스틱은 가로3px
- 캔들의 몸통은 3px, 꼬리는 1px
- 하단에 반투명색 거래량 표기
- 이동평균선(10, 20, 60) 표기

<br>
<br>

## 4. 학습 데이터 선별


1. 5거래일 후 종가가 현재 종가보다 n%(5~20) 상승(하락)한 차트들로 학습

2. 10거래일 후 종가가 현재 종가보다 n%(5~20) 상승(하락)한 차트들로 학습

3. 5거래일 후 종가가 현재 종가보다 n%(5~20) 상승(하락)하고, 10거래일 후 종가가 5거래일 후 종가보다 n%(1~10) 상승(하락)한 차트들로 학습

<br>
<br>

## 5. CNN 구조

![](https://velog.velcdn.com/images/dodo4723/post/7ad22de4-37fc-414d-94a7-9e7091b93050/image.png)

<br>
<br>

## 6. 독특한 결과값 조정

기존 CNN 모델은 각 데이터에 대한 하락/상승을 0 또는 1의 정수로 반환하지만, 여기서는 독특한 방법을 사용하였습니다.

각 데이터에 대한 하락/상승 확률을 아래와 같이 0과 1사이의 실수로 반환받습니다.

![](https://velog.velcdn.com/images/dodo4723/post/7a3f69f7-9d5b-45e0-98fe-e9bf99d6ef6e/image.png)

구간의 범위를 조절해가며 정확도를 조사하다보면 정확도가 높은 특정 범위 (예시 : 0.7~0.9)가 있습니다.

이 구간 안에 들어온 종목만 향후 상승할 가능성이 높은 종목이라고 판단합니다.

<br>
<br>

## 7. 앙상블

각각 다른 파라미터, 다른 학습데이터로 학습한 5개의 CNN 모델을 앙상블하여 5개 모델이 모두 위와 같은 방식으로 상승이라고 판단한 종목을 추출합니다.

```python
# 테스트 데이터셋에 대한 예측 생성
pred_probs1 = model1.predict(test_images)
pred_probs2 = model2.predict(test_images)
pred_probs3 = model3.predict(test_images)
pred_probs4 = model4.predict(test_images)
pred_probs5 = model5.predict(test_images)
```

```python
# 5개의 모델이 모두 상승으로 정답이라고 했을때

selected_images = []
selected_labels = []

for i in range(len(test_images)):
    pred_probs1_max_idx = np.argmax(pred_probs1[i])
    pred_probs2_max_idx = np.argmax(pred_probs2[i])
    pred_probs3_max_idx = np.argmax(pred_probs3[i])
    pred_probs4_max_idx = np.argmax(pred_probs4[i])
    pred_probs5_max_idx = np.argmax(pred_probs5[i])

    if (0.64 <= pred_probs1[i][pred_probs6_max_idx] <= 0.9
    and 0.67 <= pred_probs2[i][pred_probs7_max_idx] <= 1 
    and 0.6 <= pred_probs3[i][pred_probs8_max_idx] <= 1 
    and 0.52 <= pred_probs4[i][pred_probs9_max_idx] <= 0.8
    and 0.59 <= pred_probs5[i][pred_probs10_max_idx] <= 1):
    
        selected_images.append(test_images[i])
        selected_labels.append(test_labels_one_hot[i])

selected_images = np.array(selected_images)
selected_labels = np.array(selected_labels)
```

하루에 2600개의 종목 중 평균 2~3개의 종목이 추출됩니다.

<br>
<br>
<br>

## 8. 성능

![](https://velog.velcdn.com/images/dodo4723/post/9a08912b-b33e-45d1-8b59-c520e2682a8c/image.png)

![](https://velog.velcdn.com/images/dodo4723/post/1fc039e8-4477-4cc6-bdeb-8d7cc1702526/image.png)

평균적으로 55%이상의 정확도를 얻을 수 있었습니다.

<br>
<br>
<br>
<br>

## 9. 결론

다양한 방식으로 연구해보았지만, 실전 매매에 사용하기에는 무리가 있었습니다. 하지만 여기서 또 다른 생각이 들었습니다.

딥러닝도 결국 과거 비슷한 것을 찾아서 학습합니다. 그러면 애초에 딥러닝 없이 가장 비슷한 차트들을 찾아버리면 좋지 않을까요? 

결국 프로젝트 주제를 '비슷한 차트 검색기'로 변경하였습니다. 이에 관련한 자세한 내용은 [블로그](https://blog.similarchart.com/113)에 기재하였습니다.
