import numpy as np
import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class CriticNetwork(nn.Module):
    def __init__(self, sess, state_size, action_size, action_dim, BATCH_SIZE, TAU, LEARNING_RATE):
        super(Net, self).__init__()
        self.w1 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.a1 = nn.Linear(action_dim, HIDDEN2_UNITS)
        self.h1 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.h3 = nn.Linear(HIDDEN2_UNITS, HIDDEN2_UNITS)
        self.V = nn.Linear(HIDDEN2_UNITS, action_dim)

        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

    def forward(self, x):
        x = F.relu(self.w1(x))
        x = 
