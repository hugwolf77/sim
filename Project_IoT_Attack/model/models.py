import torch
import torch.nn as nn
import torch.nn.functional as F


class test_model(nn.Module):
    def __init__(self):
        super(test_model,self).__init__()
        self.linear_1 = nn.Linear(45, 30)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(30, 19)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        out = self.softmax(x)
        return out