---
layout: single
title: 
categories: 
tags: 
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
### 1. Anomaly Detection - Deep Learning 기법을 중심으로

- ML or statistic 기초 방법
  - IQR (Interquartile Range) : 특정 차순으로 데이터 정렬, 데이터의 25% 지점 (Q1)과 75% 지점 (Q3)에 일정 배율 범위 밖의 데이터를 이상치로 판단.
  - 확률적 접근
  - 분류모델 사용
  - isolation forest

- Traditional anomaly detection

|Reconstruction-Based|Clustering-Based|one-class Learning|
|--------------------|-------------------|------------------|
|PCA|Gaussian Mixture Models|One-Class Support Vector Machine|
|Kernel PCA| K-means|Support Vector Data Description(SVDD)|
|Robust PCA| Kenel Desity Estimator||

- Deep Learning 방법을 통한 Anomaly Detection을 다룰 때, 기법적인 것 만큼 데이터에 대한 이해와 분석 목적에 대한 논리적 근거에 대한 고찰을 놓쳐서는 안된다.

(1) Anomaly 란?
 >“An anomaly is an observation that deviates considerably from some concept of normality”
 - 위 문장에 따르면 정상으로 부터 구분되어 진다고 하였다. 따라서 정상에 대한 개념적 범위를 정의하는 것에서 부터 어려운 문제이다. 즉, 분석하고자 하는 또는 분석모델이 추구하고자 하는 기능과 가치에 따라서 정상의 범위가 정해진다.
 - 이렇게 정해진 범위 밖에 있는 모든 상태에 대해서 비정상 "Anomaly" 라고 보아야 한다.

(2) Anomaly Detection 
 -  Anomaly Detection / Outlier Detection / OOD Detection 세 가지 용어로 분류
    - Anomaly Detection : 앞에서 설명한 정상 범위를 벗어난 경우 감지
    - Outlier Detection : 클라스 분류에서 학습에 없었더너 새로운 클라스에 대해 감지
    - OOD Detection     : 데이터 셋에 대해서 학습 시와 테스트 시에 비정상 차이를 감지
 - 혼재되어 사용되지만 결국 모두 Anomaly Detection 임. 


(3) Anomaly Detection의 어려움

  - Normal 에 대한 설정의 어려움 (분야와 목적에 따라서 그 범위 설정이 넓고 다양함), 실제 상황에서의 분석 대상의 다양한 특성의 범위에서 어떻게 정상의 범위 기준을 설정할 것인가
  - Anomal Data 자체가 희귀하여 거의 대부분 Class Imbalance 문제가 있다.
  - Anomal Data 자체도 여러가지 variation의 특성을 가지고 있다. 예상되는 또는 사례가 발견되는 비정상 특성도 있지만 그렇지 않은 경우도 있다.

(4)  Anomaly Data 
  - Novelty Detection : 현재는 없더라도 발생할 가능성이 있는 정보에 간섭이 없는 비정상 감지
  - Outlier Detection : 발생할 가능성이 없는 정보에 간섭이 발생한 비정상 감지


(5) Time-series Anomaly Data
  - Point Anomalies
  - contextual Anomalies
  - Cellective Anomalies

  <center>
  <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbR7uEd%2FbtrCYvvvAdd%2Fm5wqvJ1Pv8wkdHAHliufy1%2Fimg.png">
  </center>

(6) 비정상 감지 학습 기법의 분류

- 학습 Label 사용 여부 - 비성상 data sample 사용여부에 따른 분류

  - supervised Anomaly Detection
    - 장점: 양/불 판정 정확도 높음.
    - 단점: 비정상 sample을 취득하고 분류하는 시간과 비용이 높음. Class-Imbalance 문제 방안 필요.

  - semi-supervised (One-Class) Anomaly Detection : 정상데이터의 범위(discriminative boundary)를 알고 있을 경우(또는 설정하여) 

    >- One-Class Classification 방법 " One-Class SVMs for Document ClassificationOne-Class SVMs for Document Classification"  
    >- Energy-based 방법론 “Deep structured energy based models for anomaly detection, 2016 ICML” 
    >- Deep Autoencoding Gaussian Mixture Model 방법론 “Deep autoencoding gaussian mixture model for unsupervised anomaly detection, 2018 ICLR” 
    >- Generative Adversarial Network 기반 방법론 “Anomaly detection with generative adversarial networks, 2018 arXiv” 
    >- Self-Supervised Learning 기반 “Deep Anomaly Detection Using Geometric Transformations, 2018 NeurIPS” 

    - 장점: 정상 sample만 있어도 학습이 가능. (주로 많이 연구되고 있음)
    - 단점: Supervised Anomaly Detection에 비해 상대적으로 양/불 판정 정확도 낮음.
  
  - Unsupervised Anomaly Detection : 필터링 기법, 재구축, 생성 비교 등의 기법을 사용하여 특정한 점수 계산 방법을 도입하거나 확률을 토대로 학습.

    - 장점: Data Labeling 불필요.
    - 단점: 분야, 데이터 특성, 하이퍼파라메러 등에 따라 결과가 민감하게 변함, 정확도가 높지 않음.

- 기법적 특징 분류

  - Reconstruction 방식
    - 정상의 범위 특성을 학습하여 Latant Space 를 통해 재구성을 통해 일반화 했을 때, 비정상은 정상의 범위 밖으로 재구성되는 차이를 특성의 차이 도는 분포의 차이 등으로 비정상을 감지하는 학습 방식
    - Autoencoder 형식이나, LSTM  기법을 사용한 기법에서 부터 아래 예시에 포함된 생성형 기법까지 다양함.
    - 예시: AnoGAN, GANomaly, CAVGA, Divide and Assemble, MetaFormer, SSPCAB, TadGAN

  - Pretrained Feature Matching 방식 
    - 주로 이미지나 비젼에 사용. 이미 학습된 모델을 통해서 학습된 정상 특성과의 거리로 비정상 판단
    - 예시: SPADE, Mahalanobis AD, PaDiM, PatchCore

  - Normalizing Flow 방식
    - 정상 데이터를 Normalizing Flow (VAE나 GAN 이 latant space factor (z)로 부터 역으로 입력 $X$의 확률 분포를 구할 수 있다는 개념) 정상 데이터를 Normalizing Flow 로 학습한되 입력 데이터의 확률 값을 바탕으로 정상과 비정상을 구분
    - 예시: Normalizing flow AD, DifferNet, CFLOW, CSFLOW, FastFlow

  - Self Supervised Learning 방식
    - 앞에서 설명한 Self Supervised Learning 을 의미, 하나의 데이터 class에 대해서 여러가지 관점에서 학습하도록하거나 Constrastive Learning 을 사용할 수도 있다.
    - 예시: GEOM, GEOD, SSL-AD, CSI, SSL-OCC, Hierarchical Transformation AD, SSD

  - Knowledge Distillation 방식
    - Knowledge Distillation은 Teacher와 Student 네트워크를 이용하는 방법이다. Teacher 네트워크는 일반적인 대량의 데이터를 학습한 모델이며, Student 네트워크는 비정상 감지에 대한 데이터만을 학습한 네트워크이다. 둘의 예측 차이를 이용하는 방식
    - 예시: Uniformed Studnets, Student Teacher AD, Multi Resolution AD, Reverse Distillation AD

  - Synthetic Anomaly 방식
    - 비정상 데이터가 부족하니 비정상 데이터를 만들어서 학습해 주자는 개념이다.
    - 예시: DRAEM, CutPaste

(8) Time-Series 관련 학습 분류

   <center>
  <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbopPZf%2FbtrD5SXo9vW%2Fy604WyJteyQFstlSUFLBM1%2Fimg.png">
  </center>

- 출처 : "A Review of Time-Series Anomaly Detection Techniques: A Step to Future Perspectives". K Shaukat et. al. (2021)
![alt text](ESP32_board_종류.png)

 - TimeVQVAE-AD, MDI, USAD, LSTM-VAE, TranAD, OC-SVM, DAGMM, 

(9) Anomaly Detection example
  - Cyber-Intrusion Detection
  - Fraud Detection
  - Malware Detection 
  - Medical Anomaly Detection 
  - Social Networks Anomaly Detection 
  - Log Anomaly Detection
  - IoT Big-Data Anomaly Detection
  - Industrial Anomaly Detection
  - Video Surveillance


(10) 기초 예제

https://www.kaggle.com/code/tikedameu/anomaly-detection-with-autoencoder-pytorch

https://blog.naver.com/joseb1234/222905561308?trackingCode=rss

---
### 2. LLM의 활용 개발

### LLM의 한계

- LLM 기술의 특성상 LLM 응답에 대한 예측이 불가능. 오류와 환상. 정적 훈련 데이터와 보유한 지식에 최종일이 존재. 새로운 것을 만들지만 창의성의 한계. 윤리적 악용. 해석불가능.

- 지도학습의 한계 :
  1) 편향성 : 학습데이터에 의식적 편향의 있으면, 출력되는 개념과 내용도 편향적임.
  2) 사실오류 : 학습한 데이터에 오류가 있으면 이 오류가 반영됨.
  3) 비논리성 : 덱스트 맥락 분석이 잘못되거나, 학습데이터의 논리 오류 ⚠️ "인과관계의 인지 여부에 대한 논란이 여전함."
  4) 독창성 부족 : 창조가 작은 특징들의 결합이라고 하지만 특징 그 자체도 학습데이터로부터 벗어나지 못한다.

- 딥러닝의 한계 :
  1) 해석불가능 : 아직도 충분하게 신경망 깊은 곳에서 일어나는 작용을 완전히 이해하거나 제어하는 것이 아니다.
  2) 데이터 의존성 : 방대한 양의 데이터를 학습하여야 하며, 지속적으로 업데이트하지 못하면 현실을 반영할 수 없다. 
  3) 비용 : 학습을 위해서 사용되는 막대한 컴퓨팅 자원과, 환경적 악영향이 발생한다.

- 기술과 사회인식 괴리:
  1) 오남용의 가능성 : 기술의 강력함만을 추구하고 의식적으로 악의적인 사용을 할 가능성이 있으나 이러한 문제 대책에 대한 연구는 거의 없다.
  2) 연구의 윤리 : 아직 기술이 초기 단계임에도 연구나 사용에 관한 사회적 연구적 기술적 사회적합의가 이뤄지지 않고 있다.

### 기술적인 한계에 대한 도전적 연구는 이어지나 윤리적 또는 인간적 문제에 대한 연구는 부족

- LoRA(Low-Rank Adaptation) : 가중치의 일부분만(가중치 매트릭스 중  소수의 low-rank만)을 학습하여 더 작은 데이터와 계산량으로 파인-튜닝을 할 수 있는 기법 

#### Prompt Engineering
- LLM을 안내하여 원하는 결과를 생성하는 프로세스 <br>
  [프롬프트 엔지니어링 관련 학습자료 사이트](https://www.promptingguide.ai/kr)

#### RAG(Retrieval-Augmented Generation)
- 대규모 언어 모델의 출력을 최적화하여 응답을 생성하기 전에 학습 데이터 소스 외부의 신뢰할 수 있는 지식 베이스를 참조하도록 하는 프로세스

---
### LangChain

[https://www.langchain.com/](https://www.langchain.com/)

[LangChain 소개 기사](https://www.ciokorea.com/column/305341#csidx973f1264e8a2e758d10e50c3f1541b5)

- 대규모 언어 모델과 애플리케이션의 통합을 간소화하는 SDK

> - 현재 지속적인 업데이트와 변화가 많아서 안정적인 학습 리소스가 없음
> - 심지어 langchain 도 수 개월에 구조 변화

[<랭체인LangChain 노트> - LangChain 한국어 튜토리얼](https://wikidocs.net/book/14314)

```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key="...")
```

##### Llama3 한글버전 개발 사례
https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B


#### meta ollama

https://ollama.com/

https://ollama.com/library 


<center>
<img src="https://velog.velcdn.com/images/kwon0koang/post/ad2c4317-c2c6-4dfc-831b-2185ac5b5cc9/image.png" width=600 >
</center>

<center>
<img src="https://velog.velcdn.com/images/kwon0koang/post/27ba04ef-cb00-473d-8675-5f8e2758e30a/image.png" width=600 >
</center>

## 사전 학습 모델 파일 사용

- GGUF 및 GGML : GPT(Generative Pre-trained Transformer)와 같은 언어 모델의 맥락에서 추론을 위한 모델을 저장하는 데 사용되는 파일 형식


1. 한글 학습 GGUF 다운로드 https://huggingface.co/teddylee777/Llama-3-Open-Ko-8B-gguf
2. Modelfile 파일 작성 Modelfile.md

```example
FROM Llama-3-Open-Ko-8B-Q8_0.gguf

TEMPLATE """{{- if .System }}
<s>{{ .System }}</s>
{{- end }}
<s>Human:
{{ .Prompt }}</s>
<s>Assistant:
"""

SYSTEM """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""

PARAMETER temperature 0
PARAMETER num_predict 3000
PARAMETER num_ctx 4096
PARAMETER stop <s>
PARAMETER stop </s>
```

3. 모델 생성, 모델 리스트 확인, 모델 실행
```cmd
ollama create {모델명}
	Ex) ollama create Llama-3-Open-Ko-8B-Q8_0 -f Modelfile


ollama list

ollama run {모델명}
	Ex) ollama run Llama-3-Open-Ko-8B-Q8_0:latest

```

4. lanchain 을 사용하여 로컬에 ollama 연결하기

```cmd
pip install langchain
```
```python
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama3:latest")
llm.invoke("What is stock?")
```

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, professional assistant named 권봇. Introduce yourself first, and answer the questions. answer me in Korean no matter what. "),
    ("user", "{input}")
])

chain = prompt | llm
chain.invoke({"input": "What is stock?"})
```

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
chain.invoke({"input": "What is stock?"})
```

```python
chain = prompt | llm | output_parser
for token in chain.stream(
    {"input": "What is stock?"}
):
    print(token, end="")
```

```python
# 첫번째 체인
prompt1 = ChatPromptTemplate.from_template("[{korean_input}] translate the question into English. Don't say anything else, just translate it.")
chain1 = (
    prompt1 
    | llm 
    | StrOutputParser()
)
message1 = chain1.invoke({"korean_input": "주식이 뭐야?"})
print(f'message1: {message1}') # What is a stock?

# 두번째 체인
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, professional assistant named 권봇. answer the question in Korean"),
    ("user", "{input}")
])
chain2 = (
    {"input": chain1}
    | prompt2
    | llm
    | StrOutputParser()
)
message2 = chain2.invoke({"korean_input": "주식이 뭐야?"})
print(f'message2: {message2}') # 주식은 한 회사의 소유권을 나타내는 증권입니다. 즉, 특정 기업에 투자하여 (중략)
```

```python
joke_chain = (
    ChatPromptTemplate.from_template("{topic}에 관련해서 짧은 농담 말해줘") 
    | llm)
poem_chain = (
    ChatPromptTemplate.from_template("{topic}에 관련해서 시 2줄 써줘") 
    | llm)

# map_chain = {"joke": joke_chain, "poem": poem_chain} # 체인에서 이처럼 사용할 때, 자동으로 RunnableParallel 사용됨
# map_chain = RunnableParallel({"joke": joke_chain, "poem": poem_chain})
map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)

map_chain.invoke({"topic": "애플"})
```
- 배포

```python
from agent import chain as chain
(FastAPI 관련 생략)

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain",
)

add_routes(
    app, 
    chain, 
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    
    # uvicorn: ASGI(Asynchronous Server Gateway Interface) 서버를 구현한 비동기 경량 웹 서버
    uvicorn.run(app, host="localhost", port=8000)
```

### huggingface key를 발급 받아 모델을 가져오기
- Huggingface : 머신러닝, 자연어 처리, 이미지 생성 모델 등 분야의 다양한 라이브러리를 제공

[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)


```python
import huggingface_hub
huggingface_hub.login()
```


<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {inlineMath: [['$', '$']]},
    messageStyle: "none",
    "HTML-CSS": { availableFonts: "TeX", preferredFont: "TeX" },
  });
</script>
