---
layout: single
title: (인공지능과빅데이터) 연습문제_03
categories: 수업
tags:
  - 연습문제
  - AI
  - DL
toc: "true"
toc_sticky: "true"
toc_label: 목차
author_profile: "false"
nav: '"docs"'
search: "true"
use_math: "true"
created: "{{date}} {{time}}"
---
---
## *CH-3*
---

- 다음은 pytorch를 이용하여 간단한 MLP 모델을 만드는 code 의 일부분으로 학습을 정의한 부분이다. 
```python

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"set device = {device}")
model = MLP().to(device) 
criterion = torch.nn.CrossEntropyLoss().to(device) 
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4)

epochs = 20
torch.cuda.empty_cache()    
train_losses = []
test_losses = []
train_acc = []
test_acc = []

for e in range(epochs):
    # training loop
    running_loss = 0       
    running_accuracy = 0 
    model.train()
    for _, data in enumerate(tqdm(train_loader)):
        # training phase            
        inputs, labels = data
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()
        optimizer.zero_grad() 
        
        # forward        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)            

        # backward
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_accuracy += torch.sum(preds == labels.data).detach().cpu().numpy()/inputs.size(0)
```

1.  다음은 위 코드에 대한 설명이다. 틀린 설명을 고르시오.
	❤️ 1) epochs = 20, 으로 설정하여 한번에 입력되는 학습 batch 크기가 20임을 정하고 있다.
	2) optimizer.zero_grad() 는 새로운 batch 입력 데이터에 대해서 학습을 진행하기 위하여  기존 가중치의 변화량을 계산한 gradient를 초기화하는 명령이다.
	3) criterion = torch.nn.CrossEntropyLoss().to(gpu) 는 손실함수를 정의한 부분이다.
	4) optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4) 는 최적하기법을 정의한 부분으로 learning rage 은 1e-4 로 하였다.
	5) loss.backward() 는 Loss 를 역전파하는 과정이고, optimizer.step() 학습을 위한 Step을 진행시키는 과정이다.

2. 다음은 위 코드에 대한 설명이다. 맞는 설명을 고르시오
	1) model = MLP().to(device)  은 MLP 클라스를 무조건 cpu에서 계산하도록 하는 코드이다.
	❤️2) device = torch.device("cuda" if torch.cuda.is_available() else "cpu")는 torch가 GPU를 사용할 수 있는지 아니면 CPU를 사용할 수 있는지 확인하여 deive 변수에 대입하는 코드이다.
	3) for e in range(epochs): 부분은 epochs 파라메터를 20으로 정하였기 때문에 1부터 20까지 e 에 배당하고 반복한다.
	4) running_loss = 0  은  한 step 동안 계산할 loss를 초기화한 것이다.
	5) model.train() 은 모델이 validation 중임을 나타낸다.

- 다음은 pytorch를 이용하여 간단한 MLP 모델을 만드는 code 의 일부분으로 test 과정을 정의한 부분이다. 
```python
### Test 
model.eval() 
y_pred = [] 
y_true = [] 
with torch.no_grad(): 
	for _, data in enumerate(tqdm(test_loader)): 
		inputs, labels = data 
		inputs = inputs.to(device).float() 
		outputs = model(inputs) 
		_, preds = torch.max(outputs, 1) 
		y_pred += list(preds.detach().cpu().numpy()) 
		y_true += list(labels.detach().numpy())
```

3.  다음은 with torch.no_grad():  코드에 대한 설명이다. 맞는 설명을 고르시오.
	1) test 과정을 진행 할 수 있도록 가중치가 학습할 수 있게 준비하는 코드이다.
	2) epochs 동안 학습된 가중치를 초기화 하는 코드이다.
	❤️ 3)  test 과정을 위하여 with 의 범위 안에서는 가중치가 변화지 않다록 선언하는 것이다. 
	4) epochs 만큼 반복을 위하여 iterator를 정의하는 것이다.
	5) test의 loss를 계산하기 위하여 손실함수를 지정하는 코드이다.

4. Confusion Matrix에서 Precision은 다음 중 어떤 개념을 나타내는가? 
	1) 모델이 실제 Negative 인 것을 Negative 로 정확히 분류한 비율 
	❤️2) 모델이 Positive로 예측한 것 중에서 실제 Positive인 것의 비율 
	3) 모델이 Positive로 잘못 예측한 것의 비율 
	4) 실제 Negative 인 것 중에서 모델이 Positive로 잘못 분류한 비율 
	5) 모델이 실제 Positive 인 것을 Positive로 정확히 분류한 비율

5. Confusion Matrix에서 Recall은 다음 중 어떤 개념을 나타내는가? 
	1) 모델이 실제 Negative 인 것을 Negative 로 정확히 분류한 비율 
	2) 모델이 Positive로 예측한 것 중에서 실제 Positive인 것의 비율 
	3) 모델이 Positive로 잘못 예측한 것의 비율 
	4) 실제 Negative 인 것 중에서 모델이 Positive로 잘못 분류한 비율 
	❤️5) 실제 Positive 인 것 중에서 모델이 Positive로 정확히 분류한 비율
	
6. Confusion Matrix에서 F1 score는 다음 중 어떤 개념을 나타내는가? 
	 ❤️1) Precision과 Recall의 조화평균 
	 2) Precision과 Recall의 산술 평균 
	 3) Precision과 Recall의 가중 평균 
	 4) Precision과 Recall의 곱 
	 5) Precision과 Recall의 차

7. ETL (Extract, Transform, Load) 프로세스에서 각 단계의 역할에 대한 설명으로 올바르지 않은 것은 무엇인가?
	1) Extract 단계는 데이터를 소스 시스템에서 추출하여 추출된 데이터를 원본 형식으로 가져온다.
	2) Transform 단계는 추출된 데이터를 비즈니스 규칙 및 요구 사항에 따라 변환하고 정제하여 목적지 시스템으로 올바른 형식으로 데이터를 준비한다.
	3) Load 단계는 변환된 데이터를 목적지 시스템으로 적재하여 저장하고, 필요한 경우 데이터를 인덱싱하여 쉽게 검색할 수 있도록 한다.
	4) Extract 단계에서는 데이터를 추출하는 데 필요한 쿼리 및 필터링 작업을 수행한다.
	❤️5) Transform 단계에서는 주로 데이터 웨어하우스에 데이터를 적재하는 작업을 수행한다.

8. EDA (Exploratory Data Analysis)에 대한 다음 설명 중 올바르지 않은 것은 무엇지 고르시오?
	1) EDA는 데이터셋의 기본적인 통계적 특성을 요약하고 탐색하여 데이터의 패턴 및 구조를 파악하는 과정이다.
	2) EDA는 데이터 분석 전에 반드시 수행되어야 하며, 데이터의 품질을 검증하고 잠재적인 문제점을 파악하는 데 도움이 된다.
	3) EDA는 시각화 기법 등을 사용하여 데이터의 분포, 상관 관계, 이상치 등을 탐색한다.
	❤️4)  EDA는 데이터를 변환하고 모델링하는 데 사용되는 과정으로, 데이터셋의 최종 결과물을 생성한다.
	5) EDA를 통해 도출된 인사이트는 데이터를 해석하고 향후 분석 방향을 결정하는 데 사용될 수 있다.
9.  MLOps (Machine Learning Operations)에 대한 다음 설명 중 올바르지 않은 것은 무엇지 고르시오?
	1) MLOps는 머신러닝 모델의 개발, 배포, 운영 및 유지보수를 자동화하고 효율화하기 위한 개념이다.
	2)  MLOps는 소프트웨어 개발의 DevOps 개념을 머신러닝 모델 개발 및 운영에 적용한 것을 말한다.
	❤️3)  MLOps는 모델을 개발한 후에만 관련될 뿐, 모델 개발 과정 자체에는 영향을 미치지 않는다.
	4)  MLOps는 모델의 생명 주기 전반에 걸쳐 통합된 프로세스를 제공하여 모델의 안정성, 확장성 및 성능을 향상시킨다.
	5) MLOps는 CI/CD (Continuous Integration/Continuous Deployment) 및 모델 모니터링과 같은 개념을 포함하여 머신러닝 시스템의 자동화된 배포 및 관리를 지원한다.

10. Cross validation에 대한 다음 설명 중 올바른 것은 무엇인가?
	1) Cross validation은 데이터를 훈련 세트와 테스트 세트로 한 번만 나누어 모델을 평가하는 것이다.
	❤️2) Cross validation은 모델의 성능을 평가하기 위해 훈련 데이터를 여러 부분으로 나눈 후, 각 부분을 순서대로 테스트 데이터로 사용하여 모델을 평가하는 것이다.
	3) Cross validation은 오버피팅을 방지하기 위해 사용되며, 모델의 복잡성을 증가시키는 데 도움이 된다.
	4) Cross validation은 모델의 하이퍼파라미터를 조정하는 데 사용되며, 최적의 하이퍼파라미터를 찾는 데 도움이 된다.
	5) Cross validation은 모델의 학습 속도를 높이기 위해 사용되며, 훈련 데이터의 크기를 줄이는 데 도움이 된다.

11. 가중치를 네트워크의 입력과 출력 연결 수에 따라 조정된 값으로 초기화하는 방법으로 활성화 함수로 하이퍼볼릭 탄젠트(tanh) 또는 시그모이드(sigmoid)를 사용할 때 효과적 것은 무엇인지 고르시오.
	❤️1) Xavier 초기화 (Xavier Initialization 또는 Glorot Initialization)
	2) 정규화 초기화 (Normalization Initialization)
	3) 레이어마다 다른 초기화 (Layer-wise Initialization)
	4) 무작위 초기화 (Random Initialization)
	5) 균등 초기화 (Uniform Initialization)

12. Drop out에 대한 설명으로 올바른 것을 고르시오.
	1) Drop out은 모델의 학습 과정에서 가중치 값을 무작위로 조정하여 오버피팅을 방지하는 기법이다.
	❤️2) Drop out은 학습 과정에서 일부 뉴런을 무작위로 비활성화하여 오버피팅을 방지하는 기법이다.
	3) Drop out은 모델의 손실 함수에 패널티를 부여하여 오버피팅을 방지하는 기법이다.
	4) Drop out은 학습률을 동적으로 조절하여 오버피팅을 방지하는 기법이다.
	5) Drop out은 모델의 가중치를 정규화하여 오버피팅을 방지하는 기법이다. 