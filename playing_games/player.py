import mcts
from games import checkers
import threading

class player(object):
    """
    A player class

    Upon instantiation, will start populating an MCTS tree on another threads.
    Whenever called upon to choose a play, given some board state, will pause
    the MCTS tree population, use it to predict a move, and then restart the tree
    from that position.
    """
    def __init__(self):
        """
        Args:
            board_state: the starting state of the board
        """
        self.event = threading.Event()
        self.event.clear()
        self.board = checkers.Board()
        self.tree = mcts.MCTS(self.board).tree
        self.mcts_thread = threading.Thread(target=self._start_er_up, args=(None, ))
        self.mcts_thread.start()

    def _start_er_up(self, root):
        MC = mcts.MCTS(self.board, tree=self.tree, root=root)
        MC.run(event=self.event)

    def get_next_move(self, board_state):
        """
        Given a board state, find the next move to take
        """
        # stop updating the tree in the background
        self.event.set()
        self.mcts_thread.join()
        move = mcts.MCTS(self.board, tree=self.tree).choose_play(board_state)
        # restart the thread, from this position as the root
        self.event.clear()
        self.board.load_state(board_state)
        next_state = self.board.single_move(*move)
        self.mcts_thread = threading.Thread(target=self._start_er_up, args=(next_state, ))
        self.mcts_thread.start()
        return move
