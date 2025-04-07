import os
import pandas as pd
from pandas import DataFrame
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from torch.utils.data.dataset import Dataset


# 이후 DB를 사용할경우
# def load_data_from_db(dbTB):
#     db_con  = conn_DB()
#     with db_con.connect() as conn:
#         data = pd.read_sql_table(dbTB, conn)
#         # print(data.head(3))
#     return data


class CIC_Dataset(Dataset):
    def __init__(self, filePath, flag:str='train', val_size=0.1, level:int=3, scale=True):
        super().__init__()
        self.filePath = filePath
        self.flag = flag
        self.val_size = val_size
        self.class_level = f'class_{level}'
        self.scale = scale
        self.scaler = StandardScaler()
        self.oneHot = OneHotEncoder(sparse_output=False)
        self.data, self.label = self.__read_data__()
        print(f"self.data.shape: {self.data.shape}")
        print(f"self.label.shape: {self.label.shape}")


    def __read_data__(self):
        '''
            1. 저장된 디렉토리로부터 데이터 파일을 읽어 드리고 
            2. concatenated 하여 하나의 DataFrame으로 반환.
            3. file name을 라벨 변환 및 one-hot encoding 변환 
            4. input data 의 scaling
            5. label data 의 별도 파일화
        '''  
        # filepath
        dataPath = self.filePath
        setPath = {
                    "device type" : [ "bluetooth","Wifi_and_MQTT"],
                    "data type" : {
                                    "train": "attacks/CSV/train/",
                                    "test": "attacks/CSV/test/",
                                }
                }  
        train_path = os.path.join(dataPath,setPath["data type"]["train"])
        train_fileList = [file for file in os.listdir(train_path)]

        # train data file stack & labeling  
        df_train = self.__label_class__(train_path,train_fileList)     
        # shuffle rows
        df_train = df_train.sample(frac=1).reset_index(drop=True)
       
        # one-hot class encoding - train
        self.oneHot.fit(df_train[[self.class_level]])
        # df_train[self.class_level] = self.oneHot.transform(df_train[[self.class_level]])
        df_train_label = self.oneHot.transform(df_train[[self.class_level]])

        df_train = df_train.drop(self.class_level, axis=1)
        # split validation dataset
        split_point = int(len(df_train)*(1-self.val_size))
        # print(f"split_point: {split_point}")
        train_data = df_train.iloc[:split_point]
        val_data = df_train.iloc[split_point:]
        train_label = df_train_label[:split_point,:]
        val_label = df_train_label[split_point:,:]

        # select dataset
        if self.flag == 'train':
            data = train_data
            label = train_label

        elif self.flag == 'test':
            # test data file stack & labeling
            test_path = os.path.join(dataPath,setPath["data type"]["test"])
            test_fileList = [file for file in os.listdir(test_path)]
            df_test = self.__label_class__(test_path,test_fileList)
            # one-hot class encoding - test
            # df_test[self.class_level] = self.oneHot.transform(df_test[[self.class_level]])
            df_test_label = self.oneHot.transform(df_test[[self.class_level]])
            data = df_test.drop(self.class_level, axis=1)
            label = df_test_label

        elif self.flag == 'val':
            data = val_data
            label = val_label

        else:
            print("You must set flag argument")
            print("Not Yet Implemented :  for Predict Dataset")
            raise 
            # data = pred_data

        # data scaling
        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data.values)
        else:
            data = data.values

        return data, label
    
    # data file stack & labeling       
    def __label_class__(self, path, fileList):
        df = DataFrame()
        for i, file in enumerate(fileList):
            print(f"Now Loading........{file}...Total_file..({len(fileList)})....left..( {len(fileList)-i-1})")
            d = pd.read_csv(os.path.join(path,file))
            if self.class_level == 'class_3':
                tag = file.replace('_train.pcap.csv','')
                tag = tag.replace('_test.pcap.csv','')
                if tag[-1].isdigit():
                    tag = tag[:-1]
                d[self.class_level] = tag
            elif self.class_level == 'class_2':
                # tag = file.replace('_train.pcap.csv','')
                # tag = tag.replace('_test.pcap.csv','')
                # if tag[-1].isdigit():
                #     tag = tag[:-1]
                # d[self.class_level] = tag
                pass
            else: # 'class_1'
                # tag = file.replace('_train.pcap.csv','')
                # tag = tag.replace('_test.pcap.csv','')
                # if tag[-1].isdigit():
                #     tag = tag[:-1]
                # d[self.class_level] = tag
                pass
            df = pd.concat([df,d],axis=0)
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        data, label = self.data[index], self.label[index]
        return data, label

    def __inverse_transform__(self, data):
        return self.scaler.inverse_transform(data)
    

# not yet
class CIC_Predict_Dataset(Dataset):
    def __init__(self, filePath, level:int=3, scale=True) -> None:
        super().__init__()
        self.data = self.__read_data__()
        self.filePath = filePath
        
    def __read_data__(self) -> DataFrame:
        pass
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        data, label = self.data[index], self.label[index]
        return data, label

    def __inverse_transform__(self, data):
        return self.scaler.inverse_transform(data)
    

if __name__ == '__main__':
    pass

        
    


