import mcts
from games import checkers

board = checkers.Board()
MC = mcts.MonteCarlo(board, max_moves=100, time=0.1)
for imove in range(100):
    move = MC.get_play()
    board.load_state(MC.states[-1])
    board.single_move(*move)
    MC.update(board.state())
