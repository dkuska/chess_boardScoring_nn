import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(832, 832)
        self.fc2 = nn.Linear(832, 416)
        self.fc3 = nn.Linear(416, 208)
        self.fc4 = nn.Linear(208, 104)
        self.fc5 = nn.Linear(104, 1)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = x.float()
        x = torch.flatten(x)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.fc2(x)
        x = self.lrelu(x)
        x = self.fc3(x)
        x = self.lrelu(x)
        x = self.fc4(x)
        x = self.lrelu(x)
        x = self.fc5(x)
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        x = x.float()
        x = torch.flatten(x)
        # TODO: Implement
        return x
