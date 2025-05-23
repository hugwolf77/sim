

- 2025.04.07 start

## CIC-IomT-MQTT dataset Analysis NN Model 

##### Description
- Make and Study NN Model for Attack Vectors in Healthcare Devices - A Multi-Protocol Dataset for Assessing IoMT Device Security data 
---
##### Author
- Kim Eui Cheol

##### License 
- MIT

---

### Now

- 2025.04.10 : DataSet, DataLoader, test_model, ModelFunc(basic train function)

### Plan

- 2025.04.13 : DataSet - Updata, ModelFunc(add Validation function & add early stopping)

- 2025.04.30 : metric class, parameter searching, save exp result

#### issues

    (1) data file list를 validation dataset을 생성하기 위해서 다시 읽어 들이는 비효율 제거
        ~~alternative 1 : 일단 trainset과 함께 로딩한다 (어짜피 훈련과정에서 사용된다.)~~
        - alternative 2 : DataSet class 에서 data file loading process를 분리하여 해결.
    (2) test 작업 시, test file loading을 별도로 진행하기 위해서는 별도의 데이터 셋을 만들어야 한다. 이때, scaling을 위해서 train dataset으로 부터 훈련된 평균과 분산값을 저장해 두는 방법은? 
        - alternative 1 : test dataset을 위한 별도의 dataset class를 작성하고 기존 train dataset class에서 property로 학십된 scaler를 가지고 test dataset class가 인자로 받을 수 있도록 하는 방법.
        - alternative 2 : DataSet class 에서 data file loading process를 분리하여 해결.

    (3) 평가를 위한 matrix class 및 결과 저장 방식
        - monitorring을 위한 tensorboard or MLflow 등
        - hyper paramter 최적화를 위한 gride-search 등
        - 

###

---

##### Data Reference 
    - S. Dadkhah, E. C. P. Neto, R. Ferreira, R. C. Molokwu, S. Sadeghi and A. A. Ghorbani. "CICIoMT2024: Attack Vectors in Healthcare devices-A Multi-Protocol Dataset for Assessing IoMT Device Security,” Internet of Things, v. 28, December 2024.

##### Model Reference

---