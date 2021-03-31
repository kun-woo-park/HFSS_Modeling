# HFSS-train

## Train 목적
본 repo은 HFSS라는 High frequency simulator의 resonance frequency의 계산 값을 기반으로 이를 하나의 수식으로 수렴시키는 것이 목적이다. 주어진 table.xlsx의 데이터는 # of turns, permitivity(surface), LS/LW, SEP, permitivity(subsidence), OD, HFSS(resonance frequency) 로 이루어져있다. 이에 HFSS 이외의 6 개의 feature로 HFSS를 표현하는 모델을 구축하고자 한다. 모델 구축 후 작성된 내용을 [LCpressure_PRA.pdf](LCpressure_PRA.pdf) 에서 확인 할 수 있다.

## 모델 구현
총 세가지 모델을 규정했으며, 각 모델은 다음을 가정한다.

### (log) model
<img src="./res_img/mod0.JPG" width="35%">

### (log + first order) model
<img src="./res_img/mod1.JPG" width="40%">

### (log + first order + second order) model
<img src="./res_img/mod2.JPG" width="50%">

각 모델의 변수는 다음을 의미한다.

### Notation
<img src="./res_img/variables.JPG" width="40%">

여기서 우린 target value와 prediction value의 오차를 줄이기 위해 다음과 같은 Objective function을 최소화 해야 한다.
### Objective function
<img src="./res_img/OF.JPG" width="25%">

## 구현 결과
각각의 모델에 대한 수렴 결과는 다음과 같다.

### (log) model result
<img src="./res_img/logerr_new.JPG" width="40%">

### (log + first order) model result
<img src="./res_img/x1err_new.JPG" width="40%">

### (log + first order + second order) model result
<img src="./res_img/x2err_new.JPG" width="40%">

더 복잡한 구조의 모델이 MSE가 감소함을 확인하였으나, 그 크기가 크지 않아, 이후 log만을 사용한 모델로 축소시켜 실험하였다.

## Outlier의 제거
여기서  한 점이 outlier로 발생하였다. 해당 값을 찾아보니 다음과 같았고, 이 값을 제외하고 다시 수렴시킨 결과 MSE가 감소하였다.

### Value of outlier
<img src="./res_img/outlier.JPG" width="40%">

### Result after reject outlier
<img src="./res_img/rejol_new.JPG" width="40%">

## Normalize 방법 수정
이전까지는 normalize 방법을 standardize 방법을 채택했었는데, 이 방법은 변수 표본마다의 표준편차와 평균값이 달라질 수 있어 기존값을 복원하는데에 신뢰도가 떨어진다. 따라서 이를 대체하기 위해 단지 각 표본들을 Maximum 값으로 나누어 scaling을 진행하였다. Scaling 후의 결과는 기존의 standardize보다는 높은 MSE를 보였다. Scaling 이후에는 복잡한 과정 없이 기존값을 복원할 수 있어, percentage error를 추가적으로 표기하였다 

### Result after scaling input features
<img src="./res_img/scaled_err_re.JPG" width="40%">

## Regularizer 추가
여기서 필요없는 feature을 제외하여 모델을 간소화 하고, overfit을 줄이기 위해 Regularizer을 추가하였다. Regularizer은 l1 norm을 사용하였다(Lasso regression).

### Objective function of Lasso regression
<img src="./res_img/sparse_OF.JPG" width="40%">

## 결과
결과를 보면,  feature 갯수 자체를 작게 사용해서 그런지, MSE가 낮게 나오려면, 6개의 feature을 모두 사용해야 했다. 따라서 Regularizer을 이용한 결과는 직접 gamma 값을 조정하며 수렴시켰고, 결과적으로 0.001 값을 주었을때 이상적으로 수렴함을 확인하였다.

### Cardinality for gamma (Lasso_GD)
<img src="./res_img/lasso_new.JPG" width="40%">

### Result after regularization(gamma = 0.001)
<img src="./res_img/lasso_err_new.JPG" width="40%">

### Decision variables
<img src="./res_img/Dev.JPG" width="40%">

## Analytic과 비교
최종적으로 설계한 모델로 하여금 Analytic으로 다시 학습시켜 HFSS와의 정확도를 비교하였다. 여기서 Analytic이란 실제 실험으로 얻어진 resonance frequency이다.

먼저 초기 Analytic 학습 결과 3개의 outliers가 발견되었다.

### Result of training with Analytic data
<img src="./res_img/annonrej_res.JPG" width="40%">

### Outliers of Analytic data
<img src="./res_img/an_OL.JPG" width="40%">

이를 제거하고 또다시 학습시킨 결과는 다음과 같다.

### Result after reject outliers
<img src="./res_img/anrej_res.JPG" width="40%">

최종 결과를 HFSS와 비교하였다.

### Result compared with HFSS
<img src="./res_img/comp_res_hotfix_fin.JPG" width="40%">

## Evaluation
주어진 evaluation 데이터를 활용하여 evaluate 해본 결과는 다음과 같다.

### Result of evaluating model
<img src="./res_img/test_res_hotfix.JPG" width="40%">
이때 각 Decision variables은 다음과 같고, 이를 수식에 그대로 적용하여 다시 실험해 보았다.

### Decision variables
<img src="./res_img/fin_res.JPG" width="40%">

### Result of model
<img src="./res_img/fin_res_hotfix.JPG" width="40%">
