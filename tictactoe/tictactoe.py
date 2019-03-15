# coding: utf-8

# training RL agent to play tic-tac-toe
# temporal difference learning

# Hongli Wang

# notes:
# 1. update method is wrong
# 2. self-play?
import numpy as np
import matplotlib.pyplot as plt
import random
# environment class


class TicBoard:

    def __init__(self):
        self.board=np.zeros((3,3))  # initiate the board, 0: empty; 1: O; 2: X;
        self.step = 0

    def show_board(self):
        temp_board = np.zeros((3,3))
        for x in range(3):
            for y in range(3):
                temp_board[x, y] = self.board[x, y]
        return temp_board

    def draw_board(self):
        # method to draw current board situation
        plt.figure()
        for x in range(2):
            plt.plot([0,1,2,3],[x+1,x+1,x+1,x+1],'black')
            plt.plot([x+1,x+1,x+1,x+1],[0,1,2,3],'black')
        plt.xlim((0,3))
        plt.ylim((0,3))
        for y in range(3):
            for z in range(3):
                if self.board[y,z] == 1:
                    plt.plot([y+0.5],[z+0.5],color = 'black', marker = 'o',markersize = 24)
                elif self.board[y,z] == 2:
                    plt.plot([y+0.5],[z+0.5],color = 'black', marker = 'x',markersize = 24)

        plt.show()

    def update_board(self, player, action):
        # method to update the board after one move
        # action should be a coordinate
        if player == 2:
            self.board[action[0], action[1]] = 2
        else:
            self.board[action[0], action[1]] = 1
        self.step += 1

    def play(self,player1, player2):
        # wrap the play process in the method
        pass

    def is_end(self):
        # method to determine the result
        win = [0, 0]
        if self.step != 0:
            for i in range(2):
                if self.board[i,0] == self.board[i,1] and self.board[i,1] == self.board[i,2] and self.board[i,0] != 0:
                    if self.board[i,0] == 1:
                        win = [1, 0]
                    else:
                        win = [0, 1]
                    return win
                elif self.board[0,i] == self.board[1,i] and self.board[1,i] == self.board[2,i] and self.board[0,i] != 0:
                    if self.board[0,i] == 1:
                        win = [1, 0]
                    else:
                        win = [0, 1]
                    return win
            if self.board[0,0] == self.board[1,1] and self.board[1,1] == self.board[2,2] and self.board[1,1] != 0:
                if self.board[0,0] == 1:
                    win = [1,0]
                else:
                    win = [0,1]
                return win
            elif self.board[0,2] == self.board[1,1] and self.board[1,1] == self.board[2,0] and self.board[1,1] != 0:
                if self.board[2,0] == 1:
                    win = [1,0]
                else:
                    win = [0,1]
                return win
            elif 0 not in self.board:
                win = [0.5,0.5] # draw
                return win
        return win


class AgentTD:
    # players
    def __init__(self, player):
        self.player = player # specify 'o' or 'x'
        self.value = np.zeros((3**9,1))
        self.lr = 0.1  # learning rate
        self.epsilon = 0.05  # epsilon-greedy policy

    def action(self, board):
        action_space = np.where(board == 0)
        choice_size = np.shape(action_space)[1]
        if random.random() < self.epsilon:
            # exploration
            actionInd = random.randrange(choice_size)
            action = [action_space[0][actionInd],action_space[1][actionInd]]
        else:
            valueChoice = np.zeros(choice_size)

            for c in range(choice_size):
                tempBoard = np.copy(board)
                if self.player == 1:
                    tempBoard[action_space[0][c],action_space[1][c]] = 1
                else:
                    tempBoard[action_space[0][c], action_space[1][c]] = 2
                value = self.eval_board(tempBoard)
                valueChoice[c] = value
            maxValue = max(valueChoice)
            Indexes = np.where(valueChoice == maxValue)
            actionInd = random.choice(Indexes[0])

            action = [action_space[0][actionInd],action_space[1][actionInd]]
        return action

    def eval_board(self, board):
        Ind = board.reshape(1, 9)
        Indstr = ''
        for tt in range(9):
            Indstr += str(int(Ind[0, tt]))
        IndDec = int(Indstr, 3)
        return self.value[IndDec]

    def update_value(self, state0, state1, win):
        # method to get the current value function estimation
        Ind0 = state0.reshape(1, 9)
        Ind1 = state1.reshape(1, 9)
        Indstr0 = ''
        Indstr1 = ''
        for tt in range(9):
            Indstr0 += str(int(Ind0[0, tt]))
            Indstr1 += str(int(Ind1[0, tt]))
        Ind0Dec = int(Indstr0, 3)
        Ind1Dec = int(Indstr1, 3)
        if not sum(win):
            self.value[Ind0Dec] = self.value[Ind0Dec] + self.lr * (self.value[Ind1Dec] - self.value[Ind0Dec])
        else:
            if self.player == 1:
                reward = win[0]
            else:
                reward = win[1]
            self.value[Ind1Dec] = self.value[Ind1Dec] + self.lr * (reward - self.value[Ind1Dec])

class AgentRan:
    # a agent playing randomly
    # players
    def __init__(self, player):
        self.player = player  # specify 'o' or 'x'

    def action(self, board):
        action_space = np.where(board == 0)
        choice_size = np.shape(action_space)[1]
        # exploration

        actionInd = random.randrange(choice_size)
        action = [action_space[0][actionInd], action_space[1][actionInd]]

        return action
