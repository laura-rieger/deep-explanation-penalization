import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 3)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 3)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.25)
        
        self.conv2_1 = nn.Conv2d(32, 64, 3)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 3)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.25)
        
        
        
        self.fc3_1 = nn.Linear(16 * 5 * 5*4, 512)
        self.relu3_1 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
        self.fc3_2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1_1(x)
        x =self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = self.conv2_1(x)
        x =self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        x = x.view(-1, 16 * 5 * 5*4)
        x = self.fc3_1(x)
        x = self.relu3_1(x)
        
        x = self.drop3(x)
        x = self.fc3_2(x)
        return F.log_softmax(x, dim=1)

    def logits(self, x):
        x = self.conv1_1(x)
        x =self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = self.conv2_1(x)
        x =self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc3_1(x)
        x = self.relu3_1(x)
        x = self.fc3_2(x)
        return x
    