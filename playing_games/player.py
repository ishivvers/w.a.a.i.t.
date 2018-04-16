import threading
from copy import copy
from json import dumps, loads
import socket
import numpy as np
from time import sleep
from random import choice
from sys import argv

import mcts
from games import checkers


class player(object):
    """
    The base player class
    """
    def __init__(self, player_id, game_id, server_address=('localhost', 4242)):
        """
        Args:
            board_state: the starting state of the board
        """
        self.board = checkers.Board()
        self.player = player_id
        self.game_id = game_id
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(server_address)

    def get_next_play(self, board_state):
        """
        Given a board state, find the next move to take
        """
        raise NotImplemented()

    @staticmethod
    def _response2state(response):
        s2c_map = {0: 1, 1: 2, 2: 11, 3: 22, 4: 0, 5: -1}
        board = np.empty((8, 8))
        for j, col in enumerate(response['game_state']['board']):
            for i, val in enumerate(col):
                board[7 - i, j] = s2c_map[val]
        return (
            tuple(board.flatten()),
            response['game_state']['turn'],
            response['game_state']['moves_without_capture']
        )

    @staticmethod
    def _translate_coords(coords):
        return int(8 - coords[0]), int(coords[1] + 1)

    def _play2json(self, path):
        play_array = []
        for move in path:
            play_array.append(self._translate_coords(move[0]))
        play_array.append(self._translate_coords(path[-1][1]))
        out = {'command': 'move', 'game_id': str(self.game_id), 'player': self.player,
               'move': play_array}
        return dumps(out)

    def perform_next_play(self, response):
        play = self.get_next_play(self._response2state(response))
        send_json = self._play2json(play)
        self.socket.send((send_json + '\n').encode('utf-8'))

    def join_game(self):
        send_json = dumps({'command': 'join', 'game_id': str(self.game_id), 'player': self.player})
        self.socket.send((send_json + '\n').encode('utf-8'))

    def get_game_status(self):
        response = b''
        while b'\n' not in response:
            response += self.socket.recv(1)
        response = loads(response)
        assert(response['result'] == 'ok')
        return response

    def run(self):
        self.join_game()
        while True:
            response = self.get_game_status()
            if 'game_state' not in response:
                print('waiting to start ...')
                continue
            if response['game_state']['turn'] == self.player:
                try:
                    self.perform_next_play(response)
                except ValueError:
                    print('game over, man')
                    return


class random_player(player):
    """
    Will simply choose randomly from the allowed moves at each state.
    """

    def get_next_play(self, board_state):
        self.board.load_state(board_state)
        self.board.display()
        play = choice(self.board.legal_moves())
        print('Making move: {}'.format(play))
        print('\n' * 3)
        return play


class mcts_player(player):
    """
    Upon instantiation, will start populating an MCTS tree on another thread.
    Whenever called upon to choose a play, given some board state, will pause
    the MCTS tree population, use it to predict a move, and then restart the tree
    from that position.
    """
    def __init__(self, player_id, game_id, server_address=('localhost', 4242),
                 time_allowed=5, n_threads=1):
        """
        Args:
            time_allowed (int): time allowed for thinking before a response is required
            n_threads (int): number of worker MCTS threads to spin up
        """
        super().__init__(player_id, game_id, server_address=('localhost', 4242))
        self.event = threading.Event()
        self.event.clear()
        self.time_allowed = time_allowed
        self.n_threads = n_threads
        self.tree_keeper = mcts.MCTS(checkers.Board())  # holds the merged tree
        self.mcts_threads = [None] * self.n_threads  # holds the pointers to worker threads
        self.thread_trees = [{} for _ in range(self.n_threads)]  # holds the trees built by worker threads
        self._start_threads()

    def _run_from(self, root, tree):
        MC = mcts.MCTS(checkers.Board(), tree=tree, root=root)
        MC.run(event=self.event)

    def _start_threads(self, root=None):
        self.event.clear()
        for i in range(self.n_threads):
            self.thread_trees[i] = {}
            self.mcts_threads[i] = threading.Thread(target=self._run_from,
                                                    args=(root, self.thread_trees[i]))
            self.mcts_threads[i].start()

    def _stop_threads(self):
        self.event.set()
        for i in range(self.n_threads):
            self.mcts_threads[i].join()
            self.tree_keeper.update_tree_from(self.thread_trees[i])

    def get_next_play(self, board_state):
        """
        Given a board state, find the next move to take
        """
        self.board.load_state(board_state)
        self.board.display()
        print('thinking ...')
        # restart the thread from this state
        self._stop_threads()
        self._start_threads(board_state)
        sleep(self.time_allowed)
        # choose a play
        print('Size of main tree:', len(self.tree_keeper.tree))
        play = self.tree_keeper.choose_play(board_state)
        print('Making move: {}'.format(play))
        print('\n' * 3)
        # restart the thread after choosing this step
        self.board.make_moves(play)
        next_state = self.board.state()
        self._stop_threads()
        self._start_threads(next_state)
        return play


if __name__ == '__main__':
    if argv[1] == 'random':
        p = random_player(2, 'game_1')
    elif argv[1] == 'mcts':
        p = mcts_player(1, 'game_1')
    p.run()
