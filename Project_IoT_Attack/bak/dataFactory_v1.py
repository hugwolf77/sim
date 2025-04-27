import os
import pandas as pd
from pandas import DataFrame
import warnings
warnings.filterwarnings('ignore')
# import logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
    def __init__(self, filePath, flag:str='train', val_size=0.1, level:int=1, scale=True):
        super().__init__()
        self.flag = flag
        self.val_size = val_size
        self.class_level = f'class_{level}'
        # filepath
        self.filePath = filePath
        self.setPath = {
                    "device type" : [ "bluetooth","Wifi_and_MQTT"],
                    "data type" : {
                                    "train": "attacks/CSV/train/",
                                    "test": "attacks/CSV/test/",
                                }
                }  
        self.scale = scale
        self.scaler = StandardScaler()
        self.oneHot = OneHotEncoder(sparse_output=False)

        self.data, self.label = self.__read_data__()
        print(f"\nInitialized CIC_Dataset for '{self.flag}'. Data shape: {self.data.shape}, Label shape: {self.label.shape}")
        # logging.info(f"Initialized CIC_Dataset for '{self.flag}'. Data shape: {self.data.shape}, Label shape: {self.label.shape}")


    def __read_data__(self):
        '''
            1. 지정된 데이터 저장 디렉토리로부터 데이터 파일들을 읽는다.
            2. 각 해당 file name을 라벨 변환 및 one-hot encoding 변환 한다 (level에 맞춰 label 생성).
            3. 읽어 들인 각 파일의 df를 concatenated 하여 하나의 DataFrame으로 반환 한다.
            4. input data를 standardization scaling 하는 obtion을 넣는다.
            5. label data 별도 출력값으로 만든다.
        '''  
        train_path = os.path.join(self.filePath, self.setPath["data type"]["train"])
        train_fileList = [file for file in os.listdir(train_path)]

        # train data file stack & labeling  
        df_train = self.__label_class__(train_path,train_fileList)
        # ㅋㅋ; gemini 추천처리 (왜 try except 가 아니지?;;; 경로가 잘못된 경우도 있잖아??)
        if df_train.empty:
             raise ValueError("\nNo training data loaded. Check the path and file contents.")

        # shuffle rows
        df_train = df_train.sample(frac=1, replace=False, random_state=77).reset_index(drop=True)
       
        # one-hot class encoding - train
        self.oneHot.fit(df_train[[self.class_level]])
        df_train_label = self.oneHot.transform(df_train[[self.class_level]])
        df_train = df_train.drop(self.class_level, axis=1)

        # split validation dataset
        split_point = int(len(df_train)*(1-self.val_size))
        train_data = df_train.iloc[:split_point]
        val_data = df_train.iloc[split_point:]
        train_label = df_train_label[:split_point,:]
        val_label = df_train_label[split_point:,:]
        print((f"\nSplitting training data at index {split_point} for validation."))

        # select dataset
        if self.flag == 'train':
            data = train_data
            label = train_label

        elif self.flag == 'test':
            # test data file stack & labeling
            test_path = os.path.join(self.filePath, self.setPath["data type"]["test"])
            test_fileList = [file for file in os.listdir(test_path)]
            df_test = self.__label_class__(test_path,test_fileList)
            # one-hot class encoding - test
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
            base_tag = file.replace('_train.pcap.csv','').replace('_test.pcap.csv','')
            if base_tag[-1].isdigit():
                base_tag = base_tag[:-1]

            if self.class_level == 'class_4':
                print(f"Now Loading........{file}...Check::left/Total_file..({len(fileList)-i}/{len(fileList)})")
                d = pd.read_csv(os.path.join(path,file))
                d[self.class_level] = base_tag
            elif self.class_level == 'class_3':
                parts = base_tag.split('-')
                tag = f"{parts[0]}_{parts[1]}" if len(parts) > 1 else parts[0] 
                print(f"Now Loading........{file}...Check.left/Total_file..({len(fileList)-i}/{len(fileList)})")
                d = pd.read_csv(os.path.join(path,file)) 
                d[self.class_level] = tag
            elif self.class_level == 'class_2':  
                parts = base_tag.split('-')
                tag = parts[1] if len(parts) > 1 else parts[0]
                print(f"Now Loading........{file}...Check.left/Total_file..({len(fileList)-i}/{len(fileList)})")
                d = pd.read_csv(os.path.join(path,file))  
                d[self.class_level] = tag

            elif self.class_level == 'class_1':  
                parts = base_tag.split('-')
                tag = parts[1] if len(parts) > 1 else parts[0]  
                if tag in ['ARP_Spoofing','Malformed_Data','OS_Scan','Ping_Sweep','Port_Scan', 'VulScan'] :
                    print(f"Skip:..{file}..is..in...the..List..of..skip..Category..Class...({len(fileList)-i}/{len(fileList)})")
                    continue
                else:
                    print(f"Now Loading........{file}...Check.left/Total_file..({len(fileList)-i}/{len(fileList)})")
                    d = pd.read_csv(os.path.join(path,file))
                    d[self.class_level] = tag

            df = pd.concat([df,d],axis=0)
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        data, label = self.data[index], self.label[index]
        return data, label


class CIC_Predict_Dataset(Dataset):
    def __init__(self, filePath, level:int=1, scale=True) -> None:
        super().__init__()
        self.filePath = filePath
        self.scale = scale
        self.scaler = StandardScaler()
        self.class_level = f'class_{level}'
        self.data = self.__read_data__()
        
    def __read_data__(self) -> DataFrame:
        df = DataFrame()
        pred_dataList = [file for file in os.listdir(self.filePath)]
        for i, file in enumerate(os.listdir(self.filePath)):
            d = pd.read_csv(os.path.join(self.filePath,file))
            df = pd.concat([df,d],axis=0)
        if self.scale:
            self.scaler.fit(df)
            df = self.scaler.transform(df.values)
        return df
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        data = self.data[index]
        return data
    

if __name__ == '__main__':
    pass

        
    


