from bandit import Bandit, PlayerEGreedy, PlayerUCB, PlayerGradient
import numpy as np
import matplotlib.pyplot as plt

# create environment
banditTest = Bandit()
realQ = banditTest.gen_task()
opt = banditTest.find_opt()

# create agent
player1 = PlayerEGreedy()
player2 = PlayerEGreedy(0,0,0.1)
player3 = PlayerEGreedy(0,0,0.01)
player4 = PlayerEGreedy(0.1,5,0)
player5 = PlayerEGreedy(0.1,0,0.1)
player6 = PlayerUCB(0,0,2) # UCB player
# start to play!

maxIter = 1000
averageReward = np.zeros((3,maxIter))
averageRUCB = np.zeros((2,maxIter))
optimalChoice = np.zeros((3,maxIter))
optChoiceInit = np.zeros((2,maxIter))

for i in range(maxIter):
    action1 = player1.get_action()
    action2 = player2.get_action()
    action3 = player3.get_action()
    action4 = player4.get_action()
    action5 = player5.get_action()
    action6 = player6.get_action()
    # count the actions
    player1.count_action(action1)
    player2.count_action(action2)
    player3.count_action(action3)
    player4.count_action(action4)
    player5.count_action(action5)
    player6.count_action(action6)

    # calculate the optimal
    player1.optimal(action1, opt)
    player2.optimal(action2, opt)
    player3.optimal(action3, opt)
    player4.optimal(action4, opt)
    player5.optimal(action5, opt)
    optimalChoice[:,i] = [player1.optimalCount/2000,player2.optimalCount/2000,player3.optimalCount/2000]
    optChoiceInit[:,i] = [player4.optimalCount/2000, player5.optimalCount/2000]

    # get reward
    reward1 = banditTest.get_reward(action1)
    reward2 = banditTest.get_reward(action2)
    reward3 = banditTest.get_reward(action3)
    reward4 = banditTest.get_reward(action4)
    reward5 = banditTest.get_reward(action5)
    reward6 = banditTest.get_reward(action6)
    # update value estimation
    player1.update_q(action1, reward1)
    player2.update_q(action2, reward2)
    player3.update_q(action3, reward3)
    player4.update_q(action4, reward4)
    player5.update_q(action5, reward5)
    player6.update_q(action6, reward6)

    averageReward[:, i] = [np.mean(reward1),np.mean(reward2),np.mean(reward3)]
    averageRUCB[:,i] = [np.mean(reward2), np.mean(reward6)]
    print(i)
plt.figure()
plt.plot(np.transpose(averageReward))
plt.legend(['greedy','e=0.1', 'e=0.01'])
plt.show()

plt.figure()
plt.plot(np.transpose(optimalChoice))
plt.legend(['greedy','e=0.1', 'e=0.01'])
plt.show()

plt.figure()
plt.plot(np.transpose(optChoiceInit))
plt.legend(['greedy, init=5','egreedy, init=0'])
plt.show()

plt.figure()
plt.plot(np.transpose(averageRUCB))
plt.legend(['egreedy', 'UCB'])
plt.show()
"""
# gradient bandits
banditTest = Bandit(mean=4)
realQ = banditTest.gen_task()
opt = banditTest.find_opt()

player1 = PlayerGradient(lr=0.1)
player2 = PlayerGradient(lr=0.4)
player3 = PlayerGradient(lr=0.1) # without baseline
player4 = PlayerGradient(lr=0.4) # without baseline

maxIter = 1000
optimalChoice = np.zeros((4,maxIter))

for i in range(maxIter):
    action1 = player1.get_action()
    action2 = player2.get_action()
    action3 = player3.get_action()
    action4 = player4.get_action()

    # count the actions
    player1.count_action(action1)
    player2.count_action(action2)
    player3.count_action(action3)
    player4.count_action(action4)

    # calculate the optimal
    player1.optimal(action1, opt)
    player2.optimal(action2, opt)
    player3.optimal(action3, opt)
    player4.optimal(action4, opt)

    optimalChoice[:,i] = [player1.optimalCount/2000,player2.optimalCount/2000,player3.optimalCount/2000,player4.optimalCount/2000]

    # get reward
    reward1 = banditTest.get_reward(action1)
    reward2 = banditTest.get_reward(action2)
    reward3 = banditTest.get_reward(action3)
    reward4 = banditTest.get_reward(action4)

    # update value estimation
    player1.update_H(action1, reward1)
    player2.update_H(action2, reward2)
    player3.update_H(action3, reward3)
    player4.update_H(action4, reward4)

    player1.get_policy()
    player2.get_policy()
    player3.get_policy()
    player4.get_policy()

    player1.update_baseline(reward1)
    player2.update_baseline(reward2)
    print(i)

plt.figure()
plt.plot(np.transpose(optimalChoice))
plt.legend(['0.1,b','0.4,b', '0.1,nb', '0.4,nb'])
plt.show()
"""