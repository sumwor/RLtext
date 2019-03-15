from tictactoe import TicBoard, AgentTD, AgentRan
import matplotlib.pyplot as plt
board = TicBoard()
player1 = AgentTD(1)
player2 = AgentRan(2)


# board.draw_board()


# start training, playing against a random choosing agent
win = board.is_end()
maxIter = 10000
results = []

print(win)
for i in range(maxIter):
    board = TicBoard()
    win = board.is_end()
    stateTrace = [] # keep tracking the visited state
    while not sum(win):
        # start play
        state0_1 = board.show_board()
        curr_board = board.show_board()
        action1 = player1.action(curr_board)
        board.update_board(1, action1)
        state1_1 = board.show_board()
        win = board.is_end()
        if sum(win):
            # board.draw_board()
            break
        # update value function for player 1
        player1.update_value(state0_1, state1_1, win)
        # board.draw_board()

        # agent 2 playing
        curr_board = board.show_board()
        action2 = player2.action(curr_board)

        board.update_board(2, action2)
        #board.draw_board()
        # update value funtion for player 2

        win=board.is_end()
    # update the value function when game ends
    player1.update_value(state0_1, state1_1, win)
    print(i)

    if i % 100 == 0:
        # check the strength
        checkIter = 1000
        checkResult = []
        for j in range(checkIter):
            board = TicBoard()
            win = board.is_end()
            while not sum(win):
                # start play

                curr_board = board.show_board()
                action1 = player1.action(curr_board)
                board.update_board(1, action1)
                # board.draw_board()
                win = board.is_end()
                if sum(win):
                    # board.draw_board()
                    break

                # agent 2 playing
                curr_board = board.show_board()
                action2 = player2.action(curr_board)

                board.update_board(2, action2)
                # board.draw_board()
                win = board.is_end()
            checkResult.append(win[0])
        winRate = sum(checkResult)/checkIter
    results.append(winRate)
# plot the running average

# show a game playing
board = TicBoard()
win = board.is_end()
while not sum(win):
    # start play
    state0_1 = board.show_board()
    curr_board = board.show_board()
    action1 = player1.action(curr_board)
    board.update_board(1, action1)
    state1_1 = board.show_board()
    win = board.is_end()
    if sum(win):
        board.draw_board()
        break
    # update value function for player 1
    # player1.update_value(state0_1, state1_1, win)
    board.draw_board()

    # agent 2 playing
    curr_board = board.show_board()
    action2 = player2.action(curr_board)

    board.update_board(2, action2)
    board.draw_board()
    # update value funtion for player 2

    win=board.is_end()
plt.plot(results)
plt.show()
