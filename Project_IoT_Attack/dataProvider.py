
from torch.utils.data.dataset import Dataset
from dataFactory import CIC_Dataset, CIC_Predict_Dataset
from torch.utils.data import DataLoader
from config import data_setting, data_load_setting


def data_provider(flag):
    # set data path & level 
    filePath = data_setting["filePath"]
    level = data_setting["levle"]
    scale = data_setting["scale"]
    # set train with dataset
    batch_size = data_load_setting["batch_size"]
    num_workers = data_load_setting["num_workers"]
    drop_last = data_load_setting["drop_last"]
    shuffle_flag = data_load_setting["shuffle_flag"]
    
    if flag == 'test':
        shuffle_flag = False
        batch_size = batch_size
        Data = CIC_Dataset(filePath,flag,level,scale)
    elif flag == 'pred':
        shuffle_flag = False
        batch_size = 1
        Data = CIC_Predict_Dataset(filePath,flag,level,scale)
    else:
        shuffle_flag = shuffle_flag
        batch_size = batch_size
        Data = CIC_Dataset(filePath,flag,level,scale)

    print(f"data_provider : Calling dataset - {flag},  Dataset size - {len(Data)}")
    data_loader = DataLoader(
        Data,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last)
    return Data, data_loader


if __name__ == '__main__':
    pass
