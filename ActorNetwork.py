import numpy as np
import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V


HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork(nn.Module):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_size, HIDEEN1_UNITS)
        self.fc2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.steering = nn.Linear(HIDDEN2_UNITS, 1)
        self.acceleration = nn.Linear(HIDDEN1_UNITS, 1)
        self.brake = nn.Linear(HIDDEN1_UNITS, 1)

        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = F.tanh(self.steering(x))
        out2 = F.sigmoid(self.acceleration(x))
        out3 = F.sigmoid(self.brake(x))
        out += out2
        out += out3
        return out




