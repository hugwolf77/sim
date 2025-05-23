---
categories: algorithm
title: The Negative Log-Likelihood Loss
created: 2025-01-05
tags:
  - Deep-learning
  - Loss_Function
  - algorithm
---
---
#### *The Negative Log-Likelihood Loss : NLL*
---

- NLL은 주로 분류 문제, 특히 다중 클래스 분류 문제에서 확률 분포를 예측하는 모델을 훈련할 때 사용

- 선행 이해 :

##### `Likelihood` 
: 주어진 데이터가 특정 확률 분포에서 나왔을 가능성을 나타내는 척도
>[!Note] 확률과 우도의 관계 : 활률밀도함수에서
>
> **(1) 중점 사항 (Focus)**
> - 확률 (Probability): 알려진 확률에 기반하여 미래 이벤트의 가능성을 예측하는 데 중점.
> - 우도 (Likelihood): 관찰된 데이터와 특정 가설 또는 모델 간의 일치를 평가하는 데 중점.
> 
> **(2) 척도 (Scale)**
 >- 확률 (Probability): 0에서 1까지의 범위를 가지며 이벤트 발생의 가능성을 나타냄.
> - 우도 (Likelihood) : 특정 가설을 지원하는 정도를 나타내며 특정 범위에 제한되지 않음.
>
> **(3) 해석 (Interpretation)**
>- 확률 (Probability): 미래 이벤트의 확실성 또는 가능성을 측정.
>- 우도 (Likelihood) : 주어진 가설이 관찰된 데이터를 얼마나 잘 설명하는지를 나타냄.
>
> **(4) 응용 (Applications)**
>- 확률 (Probability): 예측 모델링, 의사 결정, 확률 게임 등에서 널리 사용.
>- 우도 (Likelihood) : 통계적 추론, 가설 검정, 모델 적합 등에서 널리 사용.

##### `Log Likelihood`
: Likelihood는 여러 확률 값의 곱으로 계산되기 때문에, 데이터가 많아질수록 값이 매우 작아지는 경우가 많음. 계산의 편의성과 수치적 안정성을 위해 Likelihood에 로그를 취한 Log-Likelihood를 사용.
Likelihood가 최대가 되는 지점과 Log-Likelihood가 최대가 되는 지점은 동일.
곱셈을 덧셈으로 바꿔주기 때문에 계산이 간편해짐.

##### `Negative Log-Likelihood (음의 로그 가능도)`
: 딥러닝에서는 손실 함수를 최소화하는 방식으로 모델을 훈련. Log-Likelihood는 최대화해야 하는 값이므로, 여기에 음수 부호를 붙여 Negative Log-Likelihood (NLL)로 만들어 최소화 문제로 즉, convex 형태로 바꿈.

##### `다중 클래스 문제에서의 손실함수로 사용 수식`

$$
\begin{align}
	L\ =\ -\sum_{i=1}^{n}[\ y_{i}\times log(p_{i})] \\
	&n : class\ number \\
	&i : class\ index \\
	&y_{i} : class\ one\ hot\ coding \\
	&p_{i} : model\ infer\ class\ probability\ vector
\end{align}
$$
#### `NLL의 특징`
- 주로 다중 클래스 분류 문제에서 사용.
- 모델의 출력은 각 클래스에 대한 확률 분포여야 함. (softmax 함수 등을 통해 얻음).
- 범주형 교차 엔트로피 손실 (Categorical Cross-Entropy Loss)와 동일한 결과를 나타냄. (수식적으로 동일함)
- PyTorch: `torch.nn.NLLLoss`, 입력으로 확률 분포가 아닌 log-probabilities를 넣어주어야 하기 때문에 `torch.nn.LogSoftmax` 를 이용하여 log 형태의 확률로 출력된 값을 입력으로 사용함.
