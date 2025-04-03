import os
import numpy as np
import pandas as pd
from pandas import DataFrame

import torch
from torch.utils.data.dataset import Dataset

class MyDataset(Dataset):
    def __init__(self, filePath, datatype='train') -> None:
        super().__init__()
        self.df = self.__read_data__(datatype)
        self.filePath = filePath
        self.datatype = datatype
        self.len = len(self.df)
        self.label = self.df['label']
        self.data = self.df.drop('label', axis=1)
        self.data = torch.tensor(self.data.values, dtype=torch.float32)
        self.label = torch.tensor(self.label.values, dtype=torch.long)
   
    def __read_data__(self, datatype='train') -> DataFrame:
        '''
            저장된 디렉토리로 부터 데이터 파일을 읽어 드리고 
            각 파일의 라벨에 맞게 label column을 생성
            concatenated 하여 하나의 DataFrame으로 반환
        '''  
        # dataPath = filePath
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
            df = pd.concat([df,d],axis=0)
        return df


    def __len__(self):
        pass

    def __getitem__(self):
        # x = 
        # y
        pass

    def __inverse_transform__(self):
        pass




        
    


