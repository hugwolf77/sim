from modelFunc import ModelFunction
from config import train_setting, data_setting


process  = ModelFunction(train_setting["model"], data_setting['level'], savePath=train_setting['save_path'])
process.train(
                train_setting['epochs'], 
                train_setting['learning_rate'], 
                train_setting['dropout_ratio'],
                train_setting['weight_decay'],
                data_setting['batch_size'],
                train_setting['patience'],
                
            )
