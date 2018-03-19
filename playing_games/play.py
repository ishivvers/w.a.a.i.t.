import mcts
from games import tictactoe

board = tictactoe.Board()
MC = mcts.MonteCarlo(board, max_moves=1000, time=5)
for imove in range(100):
    move = MC.get_play()
    next_state = board.next_state(MC.states[-1], move)
    MC.update(next_state)
    print(board.display(MC.states[-1], move))
    winner = board.win_values(MC.states)
    if winner:
        print('We have a winner!')
        break