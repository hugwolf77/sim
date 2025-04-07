

data_load_setting = {
    "batch_size" : 128,
    "num_workers" : 1,
    "drop_last" : False,
    "shuffle_flag" : True,
}

data_setting = {

    # data file path
    "filePath" : "/home/augustine77/mylab/sim/sim/Project_IoT_Attack/data/CIC_2024/Wifi_and_MQTT",

    # set dataset level
    "levle"    : 3,
    "scale"    : True,
    "val_size" : 0.1,
}

model_select = {
    "model" : "test_model"
}

model_setting = {

}

train_setting = {
    "epochs" : 10,
    'learning_rate' : 0.001
}