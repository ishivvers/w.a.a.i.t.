import mcts
from games import checkers
import threading
from json import dumps, loads
import socket
import numpy as np
from time import sleep


class player(object):
    """
    A player class

    Upon instantiation, will start populating an MCTS tree on another thread.
    Whenever called upon to choose a play, given some board state, will pause
    the MCTS tree population, use it to predict a move, and then restart the tree
    from that position.
    """
    def __init__(self, time_allowed=5,
                 player_id=2, game_id='game_1', server_address=('localhost', 4242)):
        """
        Args:
            board_state: the starting state of the board
        """
        self.event = threading.Event()
        self.event.clear()
        self.board = checkers.Board()
        self.player = player_id
        self.game_id = game_id
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(server_address)
        self.time_allowed = time_allowed
        self.tree = mcts.MCTS(self.board).tree
        self.mcts_thread = threading.Thread(target=self._start_er_up, args=(None, ))
        self.mcts_thread.start()

    def _start_er_up(self, root):
        MC = mcts.MCTS(self.board, tree=self.tree, root=root)
        MC.run(event=self.event)

    def get_next_play(self, board_state):
        """
        Given a board state, find the next move to take
        """
        print('thinking ...')
        sleep(self.time_allowed)
        # stop updating the tree in the background
        self.event.set()
        self.mcts_thread.join()
        play = mcts.MCTS(self.board, tree=self.tree).choose_play(board_state)
        # restart the thread, from this position as the root
        self.event.clear()
        self.board.load_state(board_state)
        self.board.display()
        self.board.make_moves(play)
        print('Making move: {}'.format(play))
        self.board.display()
        print('\n' * 3)
        self.mcts_thread = threading.Thread(target=self._start_er_up, args=(self.board.state(), ))
        self.mcts_thread.start()
        return play

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
            if response['game_state']['turn'] == self.player:
                self.perform_next_play(response)

if __name__ == '__main__':
    p = player()
    p.run()
