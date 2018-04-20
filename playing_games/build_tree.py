"""
Quick script to accumulate an MCTS tree, saving
it to file each hour during simulation.
"""

import mcts
from games import checkers

def run_hourly():
    board = checkers.Board()
    MC = mcts.MCTS(board)
    MC.log.setLevel(mcts.logging.INFO)
    i = 0
    while True:
        print('starting hour', i)
        MC.run(t=60*60)
        MC.save_tree('trees/tree_{}h.p'.format(i))
        i += 1

if __name__ == '__main__':
    run_hourly()
