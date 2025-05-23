import os
import time

from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
# from torch.nn.functional import log_softmax, softmax
# from torch.nn.functional import cross_entropy, nll_loss
import torch.nn.functional as F
from torch import optim


from model.models import test_model
from dataprovider import DataProvider

# from torchsummary import summary as summary
from torchinfo import summary as tinfo

# train-log
from torch.utils.tensorboard import SummaryWriter

# import mlflow
# print(f"{'<--'*30:<}")
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
# raise
# Create a new MLflow Experiment
# mlflow.set_experiment("MLflow test_model")

# metrics
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from sklearn.metrics import confusion_matrix, classification_report

import os

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, save=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save = save
                
    def __call__(self, val_loss, model, path, save_name, epoch, optimizer, loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save:
                self.save_checkpoint(val_loss, model, path, save_name, epoch, optimizer, loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save:
                self.save_checkpoint(val_loss, model, path, save_name, epoch, optimizer, loss)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, save_name, epoch, optimizer, train_loss):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    }, os.path.join(path, f'{save_name}.pth'))
        # torch.save(model.state_dict(), path + '\\' + f'{save_name}.pth')
        

        self.val_loss_min = val_loss


class ModelFunction:
    def __init__(self,  model_select, level=1, save=False, savePath=None, load_model = True, load_path=None):
        self.level = level
        self.model_select = model_select
        self.device = self.set_device()
        self.save = save
        self.savePath = savePath
        self.load_model = load_model
        self.load_path = load_path
        self.DataProvider = DataProvider()
        self.accuracy = Accuracy(task="multiclass", num_classes=3).to(self.device)
        # self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=3
        # self.precision = Precision()
        # self.recall = Recall()
        # self.f1_score = F1Score()

        log_dir = './log_dir' # 임시
        self.writer = SummaryWriter(log_dir=log_dir) 
        
    def _build_model(self, dropout_ratio=0.5):
        model_dict = {
                "test_model" : test_model
        }
        model = model_dict[self.model_select](self.level, dropout_ratio).float()
        return model

    def set_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Use Device:', device)
        return device
    
    def find_latest_save_checkpoint(self, save_path):
        save_files = os.listdir(save_path)
        if not save_files:
            print("No files found in the directory.")
            return None
        file_times = [(file, os.path.getmtime(os.path.join(save_path, file))) for file in save_files]
        sorted_files = sorted(file_times, key=lambda item: item[1], reverse=True)
        return sorted_files[0][0]

    def train(self, epochs=100, learning_lr=0.5, dropout_ratio=0.5, weight_decay=0.3, batch_size=300, patience=5):

        self.model = self._build_model(dropout_ratio).to(self.device)


        self.model_name = self.model_select + f"_level_{self.level}_lr({learning_lr})_dropout({dropout_ratio})_wdecay({weight_decay})_batch({batch_size})"
        print("\n {:{}^50}".format(self.model_name,'='))
        # summary(self.model, (45,), batch_size)

        train_data, train_loader = self.DataProvider.getTrainLoader()
        scaler = self.DataProvider.getScaler()
        oneHot = self.DataProvider.getOneHot()
        val_data, val_loader = self.DataProvider.getValLoader()

        # self.writer.add_graph(self.model, train_data.data)
        
        # set save path locate at project subpath
        if self.save:
            path = os.path.join(self.savePath, self.model_select, datetime.now().strftime('%Y-%m-%d'))
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        model_optim = optim.Adam(
                                self.model.parameters(), 
                                lr=learning_lr, 
                                weight_decay=weight_decay
                                )
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.NLLLoss()
        early_stopping = EarlyStopping(
                                       patience=patience, 
                                       verbose=True, 
                                       delta=0,
                                       save=self.save
                                       )

        scheduler = optim.lr_scheduler.LambdaLR(
                                                optimizer=model_optim, 
                                                lr_lambda=lambda epoch: 0.90 ** epoch, 
                                                last_epoch=-1, 
                                                verbose=False
                                            ) 
        # load model
        if self.load_model:
            find_latest_ckp = self.find_latest_save_checkpoint(self.load_path)
            print(f"find_latest_ckp : {find_latest_ckp}, path : {self.load_path}")
            checkpoint = torch.load(os.path.join(self.load_path, find_latest_ckp), weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f"model's lastest_checkpoint is on the model")

        # add mlflow monitoring 
        # with mlflow.start_run():
        #     params = {
        #         "epochs": epochs,
        #         "learning_lr": learning_lr,
        #         "dropout_ratio": dropout_ratio,
        #         "weight_decay": weight_decay,
        #         "batch_size": batch_size,
        #         "patience": patience,
        #         "loss_function":criterion.__class__.__name__,
        #         "optimizer":model_optim.__class__.__name__,
        #         "scheduler":scheduler.__class__.__name__,
        #     }

        #     # Log training parameters
        #     mlflow.log_params(params)

        #     # Log model summary
        #     with open("model_summary.txt", "w") as f:
        #         f.write(str(tinfo(self.model)))
        #     mlflow.log_artifact("model_summary.txt")

            for epoch in range(epochs):
                print("{:{}^50} ".format( 'Epoch_('+str(epoch+1)+')_Start', '-'))
                iter_count = 0
                train_loss = []
                self.model.train()
                epoch_time = time.time()

                for i, (batch_x, batch_y) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                    
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    outputs = self.model(batch_x)
                    # print(f"batch_y: {batch_y.shape}")
                    # print(f"outputs: {outputs.shape}")
                    # raise

                    loss = criterion(outputs, batch_y)
                    accuracy = self.accuracy(outputs, batch_y)
                    train_loss.append(loss.item())

                    loss.backward()
                    model_optim.step()

                    if (i + 1) % 1000 == 0:
                        # loss, current = loss.item() 
                        # mlflow.log_metric("loss", f"{loss:3f}", step=(len(batch_x) // 1000))
                        # mlflow.log_metric("accuracy", f"{accuracy:3f}", step=(len(batch_x) // 1000))
                        # print(
                        #     f"loss: {loss:3f} accuracy: {self.accuracy:3f} [{current} / {len(train_loader)}]"
                        # )
                        # mlflow.log_metric("train_loss", loss.item())
                        # mlflow.log_metric("accruocy",)

                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                        self.writer.add_scalar('Loss/train', np.average(train_loss), i + 1)

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} \n".format(epoch + 1, train_steps, train_loss))

                scheduler.step()

                val_loss = self.vali(val_loader, epoch=epoch)
                now = datetime.now().strftime('%Y-%m-%d') #_%H:%M:%S')
                self.save_name = self.model_name + str(now)
                early_stopping(val_loss, self.model, path, self.save_name, epoch, model_optim, train_loss)
                if early_stopping.early_stop:
                    print("\n{:^50}\n".format("Early Stopping"))
                    print(f"Model - {self.model_select} - Trainning Stop in Epoch : {epoch + 1} at {now}")
                    break

                self.writer.flush()
            self.writer.close()
            # mlflow.pytorch.log_model(self.model, "model")

        return scaler, oneHot # metrices

    def vali(self, val_loader, epoch):        

        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        val_loss = []

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss.append(loss.item())
        val_loss = np.average(val_loss)
        print("Validation Loss: {0:.7f} \n".format(val_loss))
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        return val_loss #, metrices

    def test(self, learning_lr=0.5, dropout_ratio=0.5, weight_decay=0.3, batch_size=300, load_path=None):

        self.model = self._build_model(dropout_ratio).to(self.device)
        test_data, test_loader =  self.DataProvider.getTestLoader()
        path = load_path

        self.model.eval()
        model_optim = optim.Adam(
                        self.model.parameters(), 
                        lr=learning_lr, 
                        weight_decay=weight_decay
                        )

        # load model
        if self.load_model:
            find_latest_ckp = self.find_latest_save_checkpoint(path)
            checkpoint = torch.load(os.path.join(path,find_latest_ckp), weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f"model's lastest_checkpoint is on the model")
        
        criterion = nn.CrossEntropyLoss()
        test_loss = []
        test_predict = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss.append(loss.item())
                # print(f"batch_y.shape : {batch_y.shape} ")
                # print(f"batch_y : {batch_y}")
                # print(f"outputs.shape : {outputs.shape}")
                # print(f"outputs : {outputs}")
                test_predict.append(outputs)
                # print(f"test_predict.length : {len(test_predict)}")
                # print(f"test_predict : {test_predict}")
        test_loss = np.average(test_loss)
        # test_predict = torch.max(outputs, 1)[1]
        test_predict = torch.cat(test_predict, dim=0)
        print(f'test_predict :{test_predict.shape}')
        print("Test Loss: {0:.7f} \n".format(test_loss))

        test_input, test_label = test_loader.dataset.data, test_loader.dataset.label

        # Evaluation Model & Check accuracy matrix
        return test_predict, test_label, test_loss
        

    def predict(self, batch_size):
        pred_Data, pred_loader = self.DataProvider.getPredLoader()
        # pred_data, pred_loader =  data_provider(flag='pred', batch_size=batch_size)
        self.model.eval()
        with torch.no_grad():
            pred = torch.tensor([])
            for i, (batch_x, _) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                pred = torch.cat((pred, self.model(batch_x)), dim=0)
        return pred
    
if __name__ == '__main__':
    pass