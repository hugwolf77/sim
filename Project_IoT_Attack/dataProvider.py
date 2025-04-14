
from torch.utils.data.dataset import Dataset
# from dataFactory import CIC_Dataset, CIC_Predict_Dataset
from dataFactory_v2 import Read_DataList, CIC_Dataset, CIC_Infer_Dataset, CIC_Predict_Dataset 
from torch.utils.data import DataLoader
from config import data_setting, predict_setting


# def data_provider(flag, batch_size):
#     # set data path & level 
#     filePath = data_setting["filePath"]
#     level = data_setting["level"]
#     scale = data_setting["scale"]
#     val_size = data_setting["val_size"]
#     # set train with dataset
#     # batch_size = data_load_setting["batch_size"]
#     num_workers = data_setting["num_workers"]
#     drop_last = data_setting["drop_last"]
#     shuffle_flag = data_setting["shuffle_flag"]
    
#     if flag == 'test':
#         shuffle_flag = False
#         batch_size = batch_size
#         Data = CIC_Dataset(filePath,flag,val_size,level,scale)
#     elif flag == 'pred':
#         shuffle_flag = False
#         batch_size = 1
#         Data = CIC_Predict_Dataset(filePath,flag,level,scale)
#     else:
#         shuffle_flag = shuffle_flag
#         batch_size = batch_size
#         Data = CIC_Dataset(filePath,flag,val_size,level,scale)

#     print(f"data_provider : Calling dataset - {flag},  Dataset size - {len(Data)}")
#     data_loader = DataLoader(
#         Data,
#         batch_size=batch_size,
#         shuffle=shuffle_flag,
#         num_workers=num_workers,
#         drop_last=drop_last)
    # return Data, data_loader


def data_provider(flag, batch_size, add_test=True):
    
    # set data path & level 
    filePath = data_setting["filePath"]
    level = data_setting["level"]
    scale = data_setting["scale"]
    val_size = data_setting["val_size"]
    pred_filePath = predict_setting["pred_filePath"]
    
    # set train with dataset
    num_workers = data_setting["num_workers"]
    drop_last = data_setting["drop_last"]
    shuffle_flag = data_setting["shuffle_flag"]


    dataload = Read_DataList(filePath,val_size,level,scale,add_test)
        
    if flag == 'test':
        shuffle_flag = False
        batch_size = batch_size
        Data = CIC_Infer_Dataset(dataload.get_test_data())
    elif flag == 'pred':
        shuffle_flag = False
        batch_size = 1
        Data = CIC_Predict_Dataset(pred_filePath, level, scale=True)
    else:
        shuffle_flag = shuffle_flag
        batch_size = batch_size
        Data = CIC_Dataset(dataload.get_train_data())

    print(f"data_provider : Calling dataset - {flag},  Dataset size - {len(Data)}")
    data_loader = DataLoader(
        Data,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last)
    
    if (flag == 'train' or flag == 'val'):
        return Data, data_loader, dataload.get_scaler(), dataload.get_oneHot()

    return Data, data_loader



if __name__ == '__main__':
    pass
