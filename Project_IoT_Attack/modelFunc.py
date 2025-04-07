import os
import time
import numpy as np

import torch
import torch.nn as nn
# from torch.nn.functional import log_softmax, softmax
# from torch.nn.functional import cross_entropy, nll_loss
import torch.nn.functional as F
from torch import optim
from config import model_select, model_setting, train_setting
from model.models import test_model
from dataProvider import data_provider


# class EarlyStopping:
#     def __init__(self, patience=7, verbose=False, delta=0):
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta

#     def __call__(self, val_loss, model, path):
#         score = -val_loss
#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, path)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, path)
#             self.counter = 0

#     def save_checkpoint(self, val_loss, model, path):
#         if self.verbose:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
#         self.val_loss_min = val_loss


class ModelFunction:
    def __init__(self):
        self.device = self.set_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        # model_dict = {
        #         "test_model" : test_model
        # }
        # model = model_dict[model_select["model"]].Model(model_setting).float()
        model = test_model().float()

        return model

    def set_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device:', device)
        return device

    def train(self):
        train_data, train_loader =  data_provider(flag='train')
        iter_count = 0
        train_loss = list()

        # path = os.path.join(self.args.checkpoints, setting)
        # if not os.path.exists(path):
        #     os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        model_optim = optim.Adam(self.model.parameters(), lr=train_setting['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        # early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(train_setting["epochs"]):

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # print(f"batch_y: {batch_y}")
                
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((train_setting["epochs"] - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps, train_loss))


            # early_stopping(train_loss, self.model, path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break






    def vali(self):
        pass


    def test(self):
        pass