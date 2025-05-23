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


class Read_DataList:
    def __init__(self, filePath, val_size=0.1, level:int=1, scale=True, add_test=True):
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
        self.scaler = StandardScaler()
        self.oneHot = OneHotEncoder(sparse_output=False)
        self.scale = scale
        self.add_test = add_test
        self.colnames:list[str] = []
        
        print(f"\n - [ Start Read Data-List And Load Data-files ] - \n")
        if self.add_test :
            self.train_data, self.train_label, self.val_Data, self.val_label, self.test_data, self.test_label = self.__read_data__()
            print(f"\nInitialized CIC_Dataset (with test) for :\n \
                  \t train data : {self.train_data.shape}, train label : {self.train_label.shape}\n \
                  \t val data : {self.val_Data.shape}, val label : {self.val_label.shape}\n \
                  \t test data : {self.test_data.shape}, test label : {self.test_label.shape}")
        else:
            self.train_data, self.train_label, self.val_Data, self.val_label = self.__read_data__()
            print(f"\nInitialized CIC_Dataset for :\n \
                  \t train data : {self.train_data.shape}, train label : {self.train_label.shape}\n \
                  \t val data : {self.val_Data.shape}, val label : {self.val_label.shape}")
        

    def __read_data__(self):

        train_path = os.path.join(self.filePath, self.setPath["data type"]["train"])
        train_fileList = [file for file in os.listdir(train_path)]

        # train data file stack & labeling  
        df_train = self.__label_class__(train_path,train_fileList)
        self.colnames:list[str] = df_train.columns.values.tolist()

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

        if self.add_test:
            # test data file stack & labeling
            test_path = os.path.join(self.filePath, self.setPath["data type"]["test"])
            test_fileList = [file for file in os.listdir(test_path)]
            df_test = self.__label_class__(test_path,test_fileList)
            # one-hot class encoding - test
            df_test_label = self.oneHot.transform(df_test[[self.class_level]])
            test_data = df_test.drop(self.class_level, axis=1)
            test_label = df_test_label

            # data scaling
            if self.scale:
                self.scaler.fit(train_data)
                train_data = self.scaler.transform(train_data.values)
                val_data = self.scaler.transform(val_data.values)
                test_data = self.scaler.transform(test_data.values)

            return train_data, train_label, val_data, val_label, test_data, test_label
        
        else: 
            # data scaling
            if self.scale:
                self.scaler.fit(train_data)
                train_data = self.scaler.transform(train_data.values)
                val_data = self.scaler.transform(val_data.values)
                
            return train_data, train_label, val_data, val_label

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

    def get_train_data(self):
        return self.train_data, self.train_label
    def get_val_data(self):
        return self.val_Data, self.val_label
    def get_test_data(self):
        return self.test_data, self.test_label
    def get_scaler(self):
        return self.scaler
    def get_oneHot(self):
        return self.oneHot
        

class CIC_Dataset(Dataset):
    def __init__(self, datazip:tuple) -> None:
        super().__init__()
        self.data, self.label = datazip

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        data, label = self.data[index], self.label[index]
        return data, label

class CIC_Infer_Dataset(Dataset):
    def __init__(self, datazip:tuple) -> None:
        super().__init__()
        self.data, self.label = datazip

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

        
    


