---
categories: algorithm
title: Boltzmann Machine
created: 2025-02-05
tags:
  - Neural_Network
  - Boltzmann_Machine
---
---
#### *Boltzmann Machine*
---
- DNN의 두 종류:
	- 1) Inference, Classifier
	- 2) Distribution, probability -> 조건부 확률 분포의 사후 분포를 예측하는 것과 관계가 있나? 다시말해서 베이지안적 관계가 있나?
	- 생성형모델의 경우 베이지안적 접근이 의미가 있을지도?
	- 모델의 feature들의 분포가 조건에 따라서 가장 생성대상의 일반적 또는 조건에 맞느 행태를 취하는 지를 확률 또는 특징 feature의 분포로 생성해 내는 것으로 본다면 말이다. 

- 볼츠만 머신은 통계물리학의 에너지 개념을 도입한 마르코프 모델의 일종
- 에너지와 확률의 은유 관계를 활용하여 노드들이 가질 수 있는 값들의 집합에 대한 확률분포를 정의한 신경 모델.
- 은닉층과 가시층으로 나누어진 노드와 노드 사이를 잇는 간선으로 구성되어있으며, 노드들이 갖는 값들의 각 조합을 형상(미시적 상태)이라 하며, 형상 S에 대한 확률 P(S)를 에너지에 대해 다음과 같이 정의한 모델.
$$P(S_{i})=\frac{exp(-\beta\in_{i})}{z}, Z=\sum_{j} exp=(-\beta\in_{j})$$

Memory는 어떻게 형성되는가? 경험의 기록과 출력은 어떻게 이뤄지나?
John Hopfield 1982 => 기억의 원리로 네트워크를 제안. 패턴의 에너지를 다음과 같이 설계함. $x{1}, x{2}$ 로 이뤄진 형상의 경우

$$E(x)\ = -Wx{1}x{2}-b_{1}x{1}-b{2}x{2}$$
$$P(x)=\frac{exp[-E(x)]}{z}$$
홉필드 네트워크에서 좋은 WW값이 정해지는 원리는 뇌과학에서 밝혀진 헤비안hebbian 규칙과 닮음?
“Fire together, wire together”


1985년 데이비드 애클리David Ackley, 제프리 힌튼Geoffrey Hinton, 테렌스 세즈노프스키Terrence Sejnowski에 의해서 제안. 1986년 오차역전파 알고리즘을 발표하는 제프리 힌튼.

- 데이터 분포간의 거리 
$$D_{KL}(P_{0}||P)=\sum_{x}P_{0}(x)log\frac{P_{0}(x)}{P(x)}$$
모형의 분포 P(x)P(x)와 데이터의 분포 P0(x)P0(x) 사이의 거리


1986년 전산 언어학자인 폴 스몰렌스키Paul Smolensky는 데이터 xx를 표현하는 다른 그래프Graph 구조를 제안
$$E(x,h)=-\sum_{i,j}W_{ij}x_{i}h{j}-\sum_{i}a_{i}x_{i}-\sum_{j}b_{j}h{j}$$


1. John J. Hopfield, "Neural networks and physical systems with emergent collective computational abilities", Proc. Natl. Acad. Sci. USA, 79(8): 2554–2558 (1982).
2. Donald O. Hebb, "The organization of behavior", New York: Wiley & Sons (1949).
3. David H. Ackley, Geoffrey E. Hinton, and Terrence J. Sejnowski, "A learning algorithm for Boltzmann machines", Cognitive Science, 9(1): 147-169 (1985).
4. Paul Smolensky, "Information processing in dynamical systems: Foundations of harmony theory", In D. E. Rummelhart and J. L. MacClelland, editors, _Parallel distributed computing: Explorations in the microstructure of cognition. Vol. 1: Foundations_, chapter 6. MIT press (1986).
5. Solomon Kullback, "Information theory and statistics", Courier Corporation (1997).
6. Geoffrey E. Hinton, "Training products of experts by minimizing contrastive divergence", Neural Computation, 14(8): 1771-1800 (2002).
7. Miguel Á. Carreira-Perpiñán and Geoffrey E. Hinton, "On Contrastive Divergence Learning", International Conference on Artificial Intelligence and Statistics, 10:33-40 (2005).
8. Juno Hwang, Wonseok Hwang, and Junghyo Jo, "Tractable loss function and color image generation of multinary restricted Boltzmann machine", NeurIPS 2020 DiffCVGP workshop paper (2020).
9. Diederik P. Kingma and Max Welling, "An introduction to variational autoencoders", Foundations and Trends in Machine Learning 12(4): 307-392 (2019).
10. Ian Goodfellow, "NIPS 2016 tutorial: Generative adversarial networks", _arXiv:1701.00160_ (2016).