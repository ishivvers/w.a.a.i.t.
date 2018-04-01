import numpy as np
from random import choice
from time import sleep

jump_test_board = np.array([
       [ 0, -1,  0, -1,  0, -1,  0, -1],
       [-1,  0, -1,  0, -1,  0, -1,  0],
       [ 0, -1,  0, -1,  0, -1,  1, -1],
       [-1,  0, -1,  0, -1,  0, -1,  0],
       [ 0, -1,  1, -1,  1, -1,  0, -1],
       [-1,  0, -1,  0, -1,  0, -1,  0],
       [ 0, -1,  1, -1,  0, -1,  0, -1],
       [-1,  2, -1,  0, -1,  0, -1,  0]], dtype=np.int8)

class Board():

    def __init__(self, state=None):
        """
        Args:
            state: the state of the board.
                   if not given, initializes a new game board.
        """
        self.cols, self.rows = np.meshgrid(range(8), range(8))
        if state is None:
            self.board = np.zeros((8, 8), dtype=np.int8)
            # mask out the unavailable spots, and put pieces down
            for i in range(self.board.shape[0]):
                for j in range(self.board.shape[1]):
                    if (i + j) % 2:
                        self.board[i, j] = -1
                    elif i <= 2:
                        self.board[i, j] = 1
                    elif i > 4:
                        self.board[i, j] = 2
            self.player = 1  # 1 (red) or 2 (black)
            self.other = 2
            self.plays_without_capture = 0
        else:
            self.load_state(state)
        self._update_masks()

    def _update_masks(self):
        p1mask = (self.board == 1) | (self.board == 11)
        p2mask = (self.board == 2) | (self.board == 22)
        if self.player == 1:
            pmask = p1mask
            omask = p2mask
        else:
            pmask = p2mask
            omask = p1mask
        self.pieces = set(zip(*(self.rows[pmask], self.cols[pmask])))
        self.others = set(zip(*(self.rows[omask], self.cols[omask])))
        self.occupied = self.pieces.union(self.others)

    def state(self):
        """
        Return the current state of the game as a tuple of tuples
        """
        return (tuple(self.board.flatten()), self.player, self.plays_without_capture)

    @staticmethod
    def unpack_state(state):
        board, player, plays_without_capture = state
        board = np.array(board).reshape((8, 8))
        return board, player, plays_without_capture

    def load_state(self, state):
        """
        Given a state tuple, unpack it into the board
        """
        self.board, self.player, self.plays_without_capture = self.unpack_state(state)

    def _row_moves(self, i, j):
        if self.board[i, j] > 10:
            # is kinged, can move either way
            return [-1, 1]
        elif self.board[i, j] == 1:
            # can only move down
            return [1]
        elif self.board[i, j] == 2:
            # can only move up
            return [-1]
        else:
            raise Exception('This should not have happened...')

    def _switch_player(self):
        self.player, self.other = self.other, self.player

    def _possible_single_moves(self, i, j):
        """
        returns list of available non-jumping moves, in the format
          (start_position, end_position, position_jumped)
        """
        return [((i, j), (i + r, j + c), None)
                 for r in self._row_moves(i, j)
                 for c in [-1, 1]
                 if min(i + r, j + c) >= 0
                 and max(i + r, j + c) < 8
                 and (i + r, j + c) not in self.occupied]

    def _possible_single_jumps(self, i, j):
        """
        returns list of available jumping moves, in the format
          (start_position, end_position, position_jumped)
        """
        possibilities = []
        for r in self._row_moves(i, j):
            # jumping to the right
            if (i + 2*r < 8 and i + 2*r >= 0 and j < 6) and\
               (i + r, j + 1) in self.others and (i + 2*r, j + 2) not in self.occupied:
                possibilities += [((i, j), (i + 2*r, j + 2), (i + r, j + 1))]
            # jumping to the left
            elif (i + 2*r < 8 and i + 2*r >= 0 and j > 1) and\
                 (i + r, j - 1) in self.others and (i + 2*r, j - 2) not in self.occupied:
                possibilities += [((i, j), (i + 2*r, j - 2), (i + r, j - 1))]
        return possibilities

    def legal_moves(self):
        """
        Create a list of the possible moves, given the current game state.

        Returns: list of list, each containing three entries as tuples of integers
            (start, end, jumped)
        """
        self._update_masks()
        possibilities = []
        # find all possible jump moves first
        for i, j in self.pieces:
            possibilities += self._possible_single_jumps(i, j)
        # if no jump moves exist, search for normal moves
        if not possibilities:
            for i, j in self.pieces:
                possibilities += self._possible_single_moves(i, j)
        return possibilities

    def win_or_moves(self):
        """
        Returns a tuple: (winner, legal_moves).
        (one of the two must be None).
        """
        if self.plays_without_capture >= 50:
            # it's a tie
            print('tied')
            return ({1: 0.5, 2: 0.5}, None)
        else:
            available_moves = self.legal_moves()
            if not available_moves:
                # other player won
                return ({self.other: 1}, None)
            else:
                # the game is afoot
                return (None, available_moves)

    def _is_kingmaking(self, end_position):
        if self.player == 1 and end_position[0] == 7:
            return True
        elif self.player == 2 and end_position[0] == 0:
            return True
        else:
            return False

    def single_move(self, start, end, jumped=None):
        """
        Declare the board that results from a single move; does not check for legality of move.

        Args:
            start (tuple of ints): row, column of piece's starting point
            end (tuple of ints): same, for piece's ending point
            jumped (tuple of ints): same, for jumped piece
        """
        board = self.board.copy()
        if self._is_kingmaking(end):
            # king me!
            board[end[0], end[1]] = 11 * self.player
        else:
            board[end[0], end[1]] = board[start[0], start[1]]
        board[start[0], start[1]] = 0
        if jumped is not None:
            board[jumped[0], jumped[1]] = 0
            self.plays_without_capture = 0
        else:
            self.plays_without_capture += 1
        self.board = board

    def display(self):
        """
        Print a visualization of the current board to the terminal.
        """
        chars = {
            1: '\u001b[47m\u001b[31m * ',  # red
            11: '\u001b[47m\u001b[31m @ ',  # red king
            2: '\u001b[47m\u001b[30m * ',  # black
            22: '\u001b[47m\u001b[30m @ ',  # black king
            0: '\u001b[47m   ',  # white background
            -1: '\u001b[40m   ',  # black background
            'reset': '\u001b[0m',
        }
        picture = '\n'
        for i in range(self.board.shape[0] + 1):
            for j in range(self.board.shape[1] + 1):
                if i == j == 0:
                    picture += '   '
                elif i == 0 and j == 1:
                    picture += ' j '
                elif i == 0:
                    picture += ' {} '.format(j - 1)
                elif j == 0 and i == 1:
                    picture += ' i '
                elif j == 0:
                    picture += ' {} '.format(i - 1)
                else:
                    picture += chars[self.board[i - 1, j - 1]]
            picture += '{}{}\n'.format(chars[-1], chars['reset'])
        print(picture)
        if self.player == 1:
            pstring = 'Player: red\n' + '-' * 28
        else:
            pstring = 'Player: black\n' + '-' * 28
        print(pstring)

    def run_random(self, pause=0.0):
        """
        Run a whole game out, with each player randomly choosing amongst legal moves.
        """
        i = 0
        while True:
            self.display()
            won, legal_moves = self.win_or_moves()
            if won:
                return won
            else:
                self.single_move(*choice(legal_moves))
                self._switch_player()
            i += 1
            sleep(pause)
        print('Game length: {} moves'.format(i))
