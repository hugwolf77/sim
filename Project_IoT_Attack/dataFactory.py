import os
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader

import torch
from torch.utils.data.dataset import Dataset

class MyDataset(Dataset):
    def __init__(self, filePath, datatype:str='train', level:int=1) -> None:
        super().__init__()
        self.df = self.__read_data__(datatype)
        self.filePath = filePath
        self.datatype = datatype
        self.class_level = ['class_1', 'class_2','class_3']
        self.label = self.df[self.class_level[level]]
        self.data = self.df.drop(self.class_level, axis=1)
        self.data = torch.tensor(self.data.values, dtype=torch.float32)
        self.label = torch.tensor(self.label.values, dtype=torch.float32)
   
    def __read_data__(self, datatype='train') -> DataFrame:
        '''
            - 저장된 디렉토리로 부터 데이터 파일을 읽어 드리고 
            
            - 각 파일의 라벨에 맞게 label column을 생성
                : 라벨 생성 시 one-hot encoding 변환하는 것이 좋을까?
            
            - concatenated 하여 하나의 DataFrame으로 반환.
        '''  
        # dataPath = filePath - 경로로 부터 파일 리스트를 생성
        # 경로는 이후 외부 입력 args 값으로 받게 함.
        dataPath = '/home/augustine77/mylab/sim/sim/Pyshark/data/CIC_2025/Wifi_and_MQTT'
        setPath = {
                    "device type" : [ "bluetooth","Wifi_and_MQTT"],
                    "data type" : {
                                    "train": "attacks/CSV/train/",
                                    "test": "attacks/CSV/test/",
                                }
                }  
        path = os.path.join(dataPath,setPath[datatype][type])
        fileList = [file for file in os.listdir(path)]
        
        df = DataFrame()
        for file in fileList:
            d = pd.read_csv(os.path.join(path,file))
            # category level에 따라서
            # file name을 이용해서 label을 작성할 때 onehot 형태로 입력
            tag1 = file.split('_')
            tag2 = tag1[1].split('-')

            # class category 의 level에 따라서
            print(f"tag1 : {tag1}")
            print(f"tag2 : {tag2} \n")
            # d['class_1'] = tag1[0]
            # d['class_2'] = tag2[0]
            # d['class_3'] = tag2[1]
            # d['class_4'] = tag2[2]
            df = pd.concat([df,d],axis=0)
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        data, label = self.data[index], self.label[index]
        return data, label

    def __inverse_transform__(self):
        pass




        
    


