import dataclasses

class args:
    pass


data_setting = {
    # data file path
    "filePath" : '/home/augustine77/mylab/sim/sim/03_Project_IoT_Attack/data/CIC_2024/Wifi_and_MQTT',

    # set dataset level
    "level"    : 1,
    "batch_size" : 300,
    
    "scale"    : True,
    "val_size" : 0.2,
    "num_workers" : 3,
    "drop_last" : False,
    "shuffle_flag" : True,
    "balanced_flag" : "RandomOverSampler", #"RandomUnderSampler",
}

train_setting = {
    "model" : "test_model",

    "model_load": True,
    "model_path": "",

    "epochs" : 30,
    "dropout_ratio" : 0.5,
    "learning_rate" : 0.003,
    "weight_decay"  : 0.1,
    "early_stopping" : True,
    "patience" : 10,

    # save
    "save" : True,
    "save_path" : '/home/augustine77/mylab/sim/sim/Project_IoT_Attack/save_model',
    "matrix_path": '',

    # load
    "load_model" : True,
    "load_path" : '/home/augustine77/mylab/sim/sim/Project_IoT_Attack/save_model/test_model/2025-05-06',
}

predict_setting = {
    "pred_filePath" : "",
}