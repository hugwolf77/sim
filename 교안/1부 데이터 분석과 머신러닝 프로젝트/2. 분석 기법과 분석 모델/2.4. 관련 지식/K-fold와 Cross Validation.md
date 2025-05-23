---
categories: 
title: K-fold와 Cross Validation
created: 2025-04-07
tags:
---
---
#### *K-fold와 Cross Validation*
---

#### *Cross Validation* : 교차 타당성 검증
1) 사용하는 이유
		a. 과적합의 방지.
		b. 데이터의 효율적인 사용
		c. 모델성능평가의 안정성
2) 교차 검증의 종류
		a. Hold Out : 미리 train과 test를 나눠두고 학습하고 교대 
		b. K-fold :  수개의 구분(폴드:fold)으로 나누고 한개구분을 test로 선정하고 나머지 구분으로 학습하는 작업을 반복
		c. nested : 일정 구획으로 넓혀 가면서 이후 데이터 일저어 크기를 test로 사용 => (시계열 교차검증 등에 쓰임.)
3) 검증 기준
		a. 평균 정확도(Mean Accuracy): 모든 폴드에 대한 평균 정확도
		c. 표준 편차(Standard Deviation): 모든 폴드에 대한 정확도의 표준 편차
	$$
		\begin{align}
			&CVE(Cross\ \ Validation \ \ Error) = \quad \frac{1}{n} \sum^{K}_{k=1}n_{k}MSE_{k} \\ \\
			&MSE_{k}\ \ = \quad \frac{1}{n_{k}} \sum_{i\in C_{k}}(y_{i}-\hat y_{i}^{[-k]})^{2}
		\end{align}
	$$
4) 주의
		a. 폴드(fold)가 너무 작으면 모델의 성능이나 학습에 영향이 갈 수 있음.
		b. 폴드(fold)의 구성이 편중되어 있거나 대표성이 없으면 문제가 있을 수 있음.



---

**1. scikit-learn (사이킷런)**

- **제공하는 기법:**
    
    - **GridSearchCV:** 지정된 하이퍼 파라미터 값들의 모든 가능한 조합을 체계적으로 탐색합니다.
    - **RandomizedSearchCV:** 하이퍼 파라미터 값들을 지정된 분포에서 무작위로 샘플링하여 탐색합니다. Grid Search보다 탐색 공간이 넓을 때 효율적일 수 있습니다.
    - **HalvingGridSearchCV, HalvingRandomSearchCV:** 자원(예: 학습 데이터 수)을 점진적으로 늘려가면서 유망한 하이퍼 파라미터 조합을 효율적으로 탐색하는 기법입니다.
- **특징:**
    
    - 머신러닝의 기본 라이브러리로서 다양한 모델과 함께 편리하게 사용할 수 있습니다.
    - 사용법이 비교적 간단하고 문서화가 잘 되어 있습니다.

**2. Optuna (옵튜나)**

- **제공하는 기법:**
    
    - **Bayesian Optimization:** 이전까지의 하이퍼 파라미터 평가 결과를 바탕으로 다음 탐색할 파라미터 값을 지능적으로 선택하여 효율적으로 최적의 값을 찾습니다.
    - **Tree-structured Parzen Estimator (TPE):** Bayesian Optimization의 한 종류로, 하이퍼 파라미터 공간을 확률적으로 모델링하여 탐색합니다.
    - **Grid Search, Random Search:** 기본적인 탐색 방법도 제공합니다.
- **특징:**
    
    - Bayesian Optimization과 TPE 알고리즘을 효과적으로 구현하여 더 적은 시도로 좋은 성능을 보이는 하이퍼 파라미터를 찾을 수 있습니다.
    - 탐색 과정을 시각화하고 모니터링하는 기능을 제공하여 편리합니다.
    - 다양한 머신러닝 프레임워크(PyTorch, TensorFlow, scikit-learn 등)와 연동이 용이합니다.

**3. Hyperopt (하이퍼옵트)**

- **제공하는 기법:**
    
    - **Tree-structured Parzen Estimator (TPE):** Optuna와 마찬가지로 Bayesian Optimization의 한 종류를 제공합니다.
    - **Random Search:** 무작위 탐색도 지원합니다.
    - **Simulated Annealing:** 최적화 알고리즘의 일종으로, 전역 최적해를 찾을 가능성을 높입니다.
- **특징:**
    
    - 유연한 탐색 공간 정의가 가능하며, 조건부 하이퍼 파라미터와 같은 복잡한 구조를 다룰 수 있습니다.
    - 분산 컴퓨팅을 지원하여 대규모 튜닝 작업을 효율적으로 수행할 수 있습니다.

**4. scikit-optimize (skopt)**

- **제공하는 기법:**
    
    - **Bayesian Optimization:** Gaussian Process를 기반으로 한 Bayesian Optimization을 제공합니다.
    - **Random Forest Embedding:** Random Forest 모델을 활용하여 탐색 공간을 효율적으로 탐색하는 방법입니다.
- **특징:**
    
    - scikit-learn과 유사한 인터페이스를 제공하여 익숙하게 사용할 수 있습니다.
    - 수치형뿐만 아니라 범주형 하이퍼 파라미터 튜닝도 지원합니다.

**어떤 라이브러리를 선택해야 할까요?**

- **scikit-learn:** 머신러닝 초보자이거나 기본적인 모델과 함께 간단한 Grid Search나 Random Search를 수행하고 싶을 때 좋습니다.
- **Optuna:** Bayesian Optimization을 쉽고 효과적으로 사용하고 싶거나, 튜닝 과정을 시각적으로 확인하고 싶을 때 추천합니다. 다양한 프레임워크와의 호환성도 뛰어납니다.
- **Hyperopt:** 복잡한 하이퍼 파라미터 공간을 정의해야 하거나, 분산 환경에서 튜닝을 수행해야 할 때 유용합니다.
- **scikit-optimize:** scikit-learn에 익숙하고 Bayesian Optimization을 사용하고 싶을 때 고려해볼 수 있습니다.

---



