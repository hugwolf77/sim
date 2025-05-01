from dataFactory import Read_DataList, CIC_Dataset, CIC_Infer_Dataset, CIC_Predict_Dataset 
from torch.utils.data import DataLoader
from config import data_setting, predict_setting


class DataProvider:
    def __init__(self,*args, add_test = True, **kargs):
    
        # set data path & level 
        self.filePath = data_setting["filePath"]
        self.level = data_setting["level"]
        self.scale = data_setting["scale"]
        self.val_size = data_setting["val_size"]
        self.pred_filePath = predict_setting["pred_filePath"]
        
        # set train with dataset
        self.num_workers = data_setting["num_workers"]
        self.batch_size = data_setting["batch_size"]
        self.drop_last = data_setting["drop_last"]
        self.shuffle_flag = data_setting["shuffle_flag"]       
        self.add_test = add_test

        self.dataloader = Read_DataList(self.filePath, self.val_size, self.level, self.scale, self.add_test)

        self.trainData, self.trainLoader = self.setLoader('train')
        self.valData, self.valLoader = self.setLoader('val')
        self.testData, self.testLoader = self.setLoader('test')
        # self.predData, self.predLoader = self.setLoader('pred')


    def setLoader(self, flag):
        if flag == 'test':
            shuffle_flag = False
            batch_size = self.batch_size
            Data = CIC_Dataset(self.dataloader.get_test_data())
        # elif flag == 'pred':
        #     shuffle_flag = False
        #     batch_size = 1
            # Data = CIC_Predict_Dataset(self.pred_filePath, self.level, scale=True)
        elif flag == 'val':
            shuffle_flag = True
            batch_size = self.batch_size
            Data = CIC_Dataset(self.dataloader.get_val_data())
        else:
            shuffle_flag = self.shuffle_flag
            batch_size = self.batch_size
            Data = CIC_Dataset(self.dataloader.get_train_data())

        print(f"data_provider : Calling dataset - {flag},  Dataset size - {len(Data)}")
        data_loader = DataLoader(
            Data,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.num_workers,
            drop_last=self.drop_last)

        return Data, data_loader

    def getTrainLoader(self):
        return self.trainData, self.trainLoader

    def getValLoader(self):
        return self.valData, self.valLoader

    def getTestLoader(self):
        return self.testData, self.testLoader

    def getPredLoader(self):
        return self.predData, self.predLoader
    
    def getScaler(self):
        return self.dataloader.get_scaler()
        
    def getOneHot(self):
        return self.dataloader.get_oneHot()
        

if __name__ == '__main__':
    pass