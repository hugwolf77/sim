import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder

from modelFunc import ModelFunction
import torch
from config import train_setting, data_setting


# experiment function setting
process  = ModelFunction(
                          train_setting["model"], 
                          save=train_setting['save'], 
                          level= data_setting['level'], 
                          savePath=train_setting['save_path'], 
                          load_model=train_setting['model_load'], 
                          load_path=train_setting['load_path']
                        )

# train setting
scaler, oneHot = process.train(
                train_setting['epochs'], 
                train_setting['learning_rate'], 
                train_setting['dropout_ratio'],
                train_setting['weight_decay'],
                data_setting['batch_size'],
                train_setting['patience'],               
    )

print(f"oneHot_categories : {oneHot.categories_}")


# test setting
test_predict, test_label, test_loss = process.test(
                                                    train_setting['learning_rate'],
                                                    train_setting['dropout_ratio'], 
                                                    train_setting['weight_decay'],
                                                    data_setting['batch_size'],
                                                    train_setting['load_path']
                                        )

def likelihood_to_one_hot(likelihood_array):
  predicted_classes = torch.argmax(likelihood_array, dim=-1)
  one_hot_encoded = F.one_hot(predicted_classes, num_classes=likelihood_array.size(-1))
  return one_hot_encoded.float() 

test_predict_result = likelihood_to_one_hot(test_predict).to('cpu').detach().numpy()
test_label = test_label.astype(int)
test_label_result = []
for label in test_label:
    test_label_result.append(list(label).index(1))

test_predict_result_list = []
for predict in test_predict_result:
    test_predict_result_list.append(list(predict).index(1))

# error
print(confusion_matrix(test_label_result, test_predict_result_list))
print(classification_report(test_label_result, test_predict_result_list))

   
# dataloader = DataProvider()
# print(f"dataloader - trainData : {type(dataloader.trainData)}")
# print(f"dataloader - trainLoader : {type(dataloader.trainLoader)}")