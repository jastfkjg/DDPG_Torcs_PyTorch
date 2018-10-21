import numpy as np
import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V


HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork(nn.Module):
    def __init__(self, state_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.fc2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.steering = nn.Linear(HIDDEN2_UNITS, 1)
        nn.init.normal_(self.steering.weight, 0, 1e-4)
        self.acceleration = nn.Linear(HIDDEN2_UNITS, 1)
        nn.init.normal_(self.acceleration.weight, 0, 1e-4)
        self.brake = nn.Linear(HIDDEN2_UNITS, 1)
        nn.init.normal_(self.brake.weight, 0, 1e-4)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out1 = t.tanh(self.steering(x))
        out2 = t.sigmoid(self.acceleration(x))
        out3 = t.sigmoid(self.brake(x))
        out = t.cat((out1, out2, out3), 1) 
        return out




