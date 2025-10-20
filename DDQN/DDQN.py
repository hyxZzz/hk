import torch
import torch.nn as nn



class Double_DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(Double_DQN, self).__init__()
        # 12_5 - 12_13都是128
        # self.fc1 = nn.Linear(state_size, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, action_size)

        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, action_size)

        self.relu = nn.ReLU()

    def forward(self, state):
        out = self.relu(self.fc1(state))
        out = self.relu(self.fc2(out))
        q = self.fc3(out)
        return q
