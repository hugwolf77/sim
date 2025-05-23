---
categories: 
title: statsmodels_01 - AutoRegression
created: 2025-05-21
tags:
---
---
#### *statsmodels_01 - AutoRegression*
---

#### [statsmodel.ts.AutoReg](https://www.statsmodels.org/dev/tsa.html#univariate-autoregressive-processes-ar)

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

# AR(1) 모델 데이터 생성
np.random.seed(0)
n = 100  # 데이터 포인트 개수
phi = 0.8  # AR(1) 계수
noise = np.random.normal(0, 1, n)  # 백색 잡음
X = np.zeros(n)
X[0] = noise[0]  # 초기값

for t in range(1, n):
    X[t] = phi * X[t-1] + noise[t]

# 데이터 시각화
plt.plot(X, label="AR(1) Process")
plt.title("Simulated AR(1) Process")
plt.legend()
plt.show()

```

```python
from statsmodels.tsa.ar_model import AutoReg

# AR(1) 모델 피팅
model = AutoReg(X, lags=1)  # 1 시차를 사용하는 AR(1) 모델
model_fitted = model.fit()

# 추정된 계수 출력
print("AR(1) 계수:", model_fitted.params)

# 예측 수행
pred = model_fitted.predict(start=len(X), end=len(X) + 10)
print("예측 값:", pred)

# 시뮬레이션된 데이터와 예측 결과 시각화
plt.plot(np.arange(len(X)), X, label="Original AR(1) Data")
plt.plot(np.arange(len(X), len(X) + len(pred)), pred, label="Predicted", color='red')
plt.legend()
plt.show()

```

