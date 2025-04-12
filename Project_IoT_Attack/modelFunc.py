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
from dataProvider import data_provider
from torchsummary import summary


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

    def __call__(self, val_loss, model, path, save_name):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save:
                self.save_checkpoint(val_loss, model, path, save_name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save:
                self.save_checkpoint(val_loss, model, path, save_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, save_name):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + f'{save_name}.pth')

        '''
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }
        '''

        self.val_loss_min = val_loss


class ModelFunction:
    def __init__(self,  model_select, level=1, save=False, savePath=None):
        self.level = level
        self.model_select = model_select
        self.device = self.set_device()
        self.save = save
        self.savePath = savePath
        
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

    def train(self, epochs=100, learning_lr=0.5, dropout_ratio=0.5, weight_decay=0.3, batch_size=300, patience=5):

        '''
         - model load for vail
            checkpoint = torch.load(PATH, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
        '''

        self.model = self._build_model(dropout_ratio).to(self.device)
        self.model_name = self.model_select + f"_level_{self.level}_lr({learning_lr})_dropout({dropout_ratio})_wdecay({weight_decay})_batch({batch_size})"
        print("\n {:{}^50}".format(self.model_name,'='))
        summary(self.model, (45,), batch_size)

        train_data, train_loader =  data_provider(flag='train', batch_size=batch_size)
        
        # set save path locate at project subpath
        # if self.save:
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
        early_stopping = EarlyStopping(
                                       patience=patience, 
                                       verbose=True, 
                                       delta=0,
                                       save=self.save
                                       )

        scheduler = optim.lr_scheduler.LambdaLR(
                                                optimizer=model_optim, 
                                                lr_lambda=lambda epoch: 0.95 ** epoch, 
                                                last_epoch=-1, 
                                                verbose=False
                                            ) 

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
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

                if (i + 1) % 1000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} \n".format(epoch + 1, train_steps, train_loss))

            scheduler.step()

            val_loss = self.vali(batch_size)
            now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            save_name = self.model_name + str(now)
            early_stopping(val_loss, self.model, path, save_name)
            if early_stopping.early_stop:
                print("\n{:^50}\n".format("Early Stopping"))
                print(f"Model - {self.model_select} - Trainning Stop in Epoch : {epoch + 1} at {now}")
                break


    def vali(self, batch_size):        
        # data loader locate
        val_data, val_loader =  data_provider(flag='val', batch_size=batch_size)

        '''
         - model load for vail
            checkpoint = torch.load(PATH, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
        '''

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
        return val_loss

    def test(self, batch_size):
        test_data, test_loader =  data_provider(flag='test', batch_size=batch_size)

        '''
         - model load for vail
            checkpoint = torch.load(PATH, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
        '''

        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device).to(torch.long)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss.append(loss.item())
        test_loss = np.average(test_loss)
        test_predict = torch.max(outputs, 1)[1]
        print("Test Loss: {0:.7f} \n".format(test_loss))

        # Evaluation Model & Check accuracy matrix
        return test_predict, test_loss
        

    def predict(self, batch_size):
        pred_data, pred_loader =  data_provider(flag='pred', batch_size=batch_size)
        self.model.eval()
        with torch.no_grad():
            pred = torch.tensor([])
            for i, (batch_x, _) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                pred = torch.cat((pred, self.model(batch_x)), dim=0)
        return pred
    
if __name__ == '__main__':
    pass