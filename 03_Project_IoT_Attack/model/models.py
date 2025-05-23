import torch
import torch.nn as nn
import torch.nn.functional as F

class test_model(nn.Module):
    def __init__(self, level=1, dropout_ratio=0.5):
        super(test_model,self).__init__()
        self.class_level_label = {
                                    "class_1" : 3,
                                    "class_2" : 9,
                                    "class_3" : 11,
                                    "class_4" : 19,

        }
            
        self.class_level = f"class_{level}"
        self.dropout_ratio = dropout_ratio

        self.linear_1 = nn.Linear(45, 90, bias=True)
        self.relu_1 = nn.ReLU()

        self.linear_2 = nn.Linear(90, 180, bias=True)
        self.batchNM_1 = nn.BatchNorm1d(180)
        self.relu_2 = nn.ReLU()        
        
        self.linear_3 = nn.Linear(180, 180, bias=True)
        self.batchNM_2 = nn.BatchNorm1d(180)
        self.relu_3 = nn.ReLU()
        
        self.dropout = nn.Dropout1d(p=self.dropout_ratio)
        
        self.linear_4 = nn.Linear(180, 90, bias=True)
        self.relu_4 = nn.ReLU()
        
        self.linear_5 = nn.Linear(90, 45, bias=True)
        self.relu_5 = nn.ReLU()
        
        self.linear_6 = nn.Linear(45, 30, bias=True)
        self.sigmoid_6 = nn.Sigmoid()
        
        self.linear_7 = nn.Linear(30, self.class_level_label[self.class_level], bias=True)
        # self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaming Initialization
                # 모듈의 가중치를 kaming he normal로 초기화합니다.
                # 편차를 0.1으로 초기화합니다.
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, x):
        x_1 = self.linear_1(x)
        x_1 = self.relu_1(x_1)

        x_2 = self.linear_2(x_1)
        x_2 = self.batchNM_1(x_2)
        x_2 = self.relu_2(x_2)
        
        x_3 = self.linear_3(x_2)
        x_3 = self.batchNM_2(x_3)
        x_3 = self.relu_3(x_3)

        x_4 = self.dropout(x_3)
        
        x_5 = self.linear_4(x_4)
        x_5 = self.relu_4(x_5)

        x_6 = self.linear_5(x_5)
        x_6 = self.relu_5(x_6)

        x_7 = self.linear_6(x_6)
        x_7 = self.sigmoid_6(x_7)

        out = self.linear_7(x_7)
        # out = self.softmax(x_8)
        return out