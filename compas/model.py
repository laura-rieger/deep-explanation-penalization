import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        
        self.fc1 = nn.Linear(17, 5)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(5, 5)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        
        x = self.drop2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def logits(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        return x
    