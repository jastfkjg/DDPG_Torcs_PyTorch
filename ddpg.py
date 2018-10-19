from gym_torcs import TorcsEnv
import numpy as np 
import random 
import argparse
import torch
import torchvision
import torch.nn as nn

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit

OU = OU()

def palygame(train_indicator = 0):
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001   #target network hyperparameters
    LRA = 0.0001  #learning rate for Actor
    LRC = 0.001   #learning rate for Critic

    action_dim = 3
    state_dim = 29

    np.random.seed(1337)

    vision = False

    EXPLORE = 100000
    episode_count = 2000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    #Torch GPU optimization
    #to do
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    env = TorcsEnv(vision=vision, throttle=True, gear_change=False)

    actor = ActorNetwork()
    critic = CriticNetwork()
    buff = ReplayBuffer(BUFFER_SIZE)

    #load weight ???


    print(" Torcs Experiment Start")

    for i in range(episode_count):

        print("Episode:" + str(i) + "Replay Buffer " + str(buff.count()))

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch = True)   # in case of the memory leak error
        else:
            ob =  env.reset()

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        total_reward = 0

        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))   #change in Torch ....

            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            ob, r_t, done, info = env.step(a_t[0])

            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

            buff.add(s_t, a_t[0], r_t, s_t1, done)   # add to Replay Buffer

            #batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = #to do

            #calculate y_t as the label of online Q network
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            if (train_indicator):
                loss +=
                #update Q network and policy network

            total_reward += r_t

            s_t = s_t1  #prepare for next step

            #print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
            step += 1
            
            if done:
                break

        if np.mod(i, 3) == 0:
            if(train_indicator):
                print("save model...")
                #to do


        print("Total Reward for " + str(i) + " episode  :  " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()   #shutting down Torcs
    print("Finish.")

if __name__ == "__main__":
    playGame()








