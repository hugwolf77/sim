---
categories: statistics
title: Bartlett's-test, KMO-test
created: 2024-08-11
tags:
  - statistics
  - factor_analysis
  - Machine_learning
---
---
#### *Bartlett's-test, KMO-test*
---

#### Bartlett's test
1. 개념: 
- 다중 집단의 등분산 검정. 
- 집단간 분산이 같은지 다른지 여부를 알아볼 때 사용.
- 독립 2표본 T-test 또는 one-way ANOVA 실시 전 등분산 가정 확인 시 사용
- 세 집단 이상에서도 사용할 수 있으나, 표본이 정규성을 보일 때만 가능하다.

2. 검정 통계량 계산:
- $s_{i}^{2}$ 는 i 번째 집단의 분산, N은 총 표본의 크기, $n_{i}$는 i 번째 집단의 표본 크기, k는 집단의 수, $s_{i}^{p}$는 합동 분산 (pooled variance) 일때

$$
	\begin{align}
		&T = \frac{(N-k)\ ln\ s_{p}^{2}-\sum_{i=1}^{k}(N_{i}-1)\ ln\ s_{i}^{2}}{1+(1/ (3(k-1)))((\sum_{i=1}^{k} 1/(N_{i}-1))-1/(N-k))} \\
		&s_{p}{2} = \sum_{i=1}^{k}\frac{(N_{i}-1)}{(N-k)}\ s_{i}^{2}
	\end{align}
$$

3. 가설
	- $H_{0}$ : 집단간 분산이 같다.
	- $H_{1}$ : 적어도 두 집단간 분산이 다르다.
	- p-value : $\alpha$ 5% (0.05) 보다 작을 때 귀무가설 기각 


- 일반적으로 ANOVA 전에 표본 집단들의 등분산 검정에 사용된다.
- Lenven, Bartlett's 검정을 통해서 등분산이 확인되어야 
-  일원분산분석의 경우 등분산을 만족하지 못할 때 **Welch test 나 Brown-Forsythe test**  사용하여야 한다.


ANOVA 사후검정

-  등분산 : Tukey, Duncan, Scheffe
- 비등분산: Tamhane' T2, Dunnett's T3, Games-Howell, Dunnett's C 등의 방법


#### Kaiser-Meyer-Olkin(KMO) 검정
KMO 값은 변수들 간의 상관관계가 요인 분석에 적합한지를 평가하는 지표로, 즉 **요인 분석을 수행하기에 적절한 데이터인지 판단하는 데 사용**

**KMO 값이 0에 가까울수록:** 변수들 간의 상관관계가 낮아 요인 분석에 적합하지 않다는 것을 의미
**KMO 값이 1에 가까울수록:** 변수들 간의 상관관계가 높아 요인 분석에 적합하다는 것을 의미
일반적으로 KMO 값이 **0.6 이상**이면 요인 분석을 수행하는 것이 적절하다고 판단