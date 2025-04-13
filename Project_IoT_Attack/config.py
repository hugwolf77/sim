

data_setting = {
    # data file path
    "filePath" : '/home/augustine77/mylab/sim/sim/Project_IoT_Attack/data/CIC_2024/Wifi_and_MQTT',

    # set dataset level
    "level"    : 1,
    "batch_size" : 500,
    
    "scale"    : True,
    "val_size" : 0.2,
    "num_workers" : 3,
    "drop_last" : False,
    "shuffle_flag" : False,
}

train_setting = {
    "model" : "test_model",

    "model_load": False,
    "model_path": "",

    "epochs" : 30,
    "dropout_ratio" : 0.3,
    "learning_rate" : 0.3,
    "weight_decay"  : 0.2,
    "early_stopping" : True,
    "patience" : 7,

    # save
    "save" : True,
    "save_path" : '/home/augustine77/mylab/sim/sim/Project_IoT_Attack/save_model',
    "matrix_path": '',

    # load
    "load_model" : False,
    # "load_path" : '',
}

predict_setting = {
    "pred_filePath" : "",
}