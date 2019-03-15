# coding: utf-8

# environment and agent for n-armed bandit task

# Hongli Wang

import numpy as np
import math
import random


class Bandit:

    def __init__(self,mean=0,var=1):
        self.arm = 10
        self.task = 2000
        self.mean = mean
        self.var = var
        self.q = self.gen_task()
        self.opt = self.find_opt()

    def gen_task(self):
        # generate reward for every arm, every task
        Q = np.zeros((self.task,self.arm))
        for i in range(self.task):
            for j in range(self.arm):
                Q[i, j] = np.random.normal(self.mean, math.sqrt(self.var))

        return Q

    def get_reward(self,action):
        # action ranging from 0 to 9
        reward = np.zeros(self.task)
        for i in range(self.task):
            reward[i] = self.q[i, int(action[i])] + np.random.normal(0, 1)
        return reward

    def find_opt(self):
        # count the optimal action
        optList = np.zeros(2000)
        for i in range(self.task):
            opt = max(self.q[i, :])
            Ind = np.where(self.q[i, :]==opt)
            optList[i] = Ind[0]
        return optList


class Agent:

    def __init__(self, lr=0, init=0):
        self.qEst = np.zeros((2000,10))+init
        self.step = 0
        self.actionCount = np.zeros((2000,10))
        self.optimalCount = 0
        if lr == 0:
            self.lr = 0
            self.lr_change = 1
        else:
            self.lr = lr
            self.lr_change = 0

    def get_action(self):
        pass

    def update_q(self, action, reward):

        for i in range(len(reward)):
            if self.lr_change == 1:
                self.lr = 1 / self.actionCount[i, int(action[i])]
            self.qEst[i, int(action[i])] = self.qEst[i, int(action[i])] + self.lr * (
                        reward[i] - self.qEst[i, int(action[i])])

    def count_action(self, action):
        # count the chosen actions
        for i in range(len(action)):
            self.actionCount[i, int(action[i])] += 1

    def optimal(self,action,opt):
        self.optimalCount = 0
        for i in range(2000):
            if action[i] == opt[i]:
                self.optimalCount += 1


class PlayerEGreedy(Agent):

    def __init__(self,lr=0,init=0,epsilon=0):
        super(PlayerEGreedy,self).__init__(lr,init)
        self.epsilon = epsilon

    def get_action(self):
        # make decision greedily
        action = np.zeros(2000)

        for i in range(len(action)):
            if random.random() < self.epsilon:
                act = random.choice(range(10))
                action[i] = act
            else:
                maxV = max(self.qEst[i,:])
                maxInd = np.where(self.qEst[i,:]==maxV)
                act = random.choice(maxInd[0])
                action[i] = act
        self.step += 1
        return action


class PlayerUCB(Agent):

    def __init__(self,lr=0,init=0,c=1):
        super(PlayerUCB,self).__init__(lr,init)
        self.c = c
        self.actionCount=np.zeros((2000,10)) + 1e-5

    def get_action(self):
        # make decision greedily
        action = np.zeros(2000)
        UCBEst = self.qEst + self.c * np.sqrt(np.log((self.step + 1)) / self.actionCount)
        for i in range(len(action)):
            maxV = max(UCBEst[i,:])
            maxInd = np.where(UCBEst[i,:]==maxV)
            act = random.choice(maxInd[0])
            action[i] = act
        self.step += 1
        return action


class PlayerGradient(Agent):

    def __init__(self,lr=0,init=0):
        super(PlayerGradient,self).__init__(lr,init)
        self.H = np.zeros((2000,10))
        self.baseline = np.zeros(2000)
        self.pi = self.get_policy()

    def update_baseline(self,reward):
        self.baseline += 1/self.step * (reward-self.baseline)

    def get_policy(self):
        policy = np.zeros((2000,10))
        for i in range(2000):
            for j in range(10):
                policy[i,j] = np.exp(self.H[i,j]) / sum(np.exp(self.H[i,:]))
        self.pi = policy
        return policy

    def update_H(self,action,reward):
        for i in range(len(action)):
            for j in range(10):
                if j == int(action[i]):
                    self.H[i, j] += self.lr * (1-self.pi[i, j]) * (reward[i] - self.baseline[i])
                else:
                    self.H[i, j] -= self.lr * self.pi[i, j] * (reward[i] - self.baseline[i])

    def get_action(self):
        action = np.zeros(2000)

        for i in range(len(action)):
            act = np.random.choice(range(10), p=self.pi[i, :])
            action[i] = act
        self.step += 1
        return action

