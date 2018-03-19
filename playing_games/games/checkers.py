import numpy as np
from random import choice

class board():

    def __init__(self, board=None):
        """
        Args:
            board (8x8 np array): the state of the board.
                if not given, initializes a new game board.
        """
        self.player = 1  # 1 (red) or 2 (black)
        self.piece_in_play = None
        self.plays_without_capture = 0
        size = 8
        self.cols, self.rows = np.meshgrid(range(size), range(size))
        if board == None:
            self.board = np.zeros((size, size), dtype=np.int8)        
            # mask out the unavailable spots, and put pieces down
            for i in range(self.board.shape[0]):
                for j in range(self.board.shape[1]):
                    if (i + j) % 2:
                        self.board[i, j] = -1
                    elif i <= 2:
                        self.board[i, j] = 1
                    elif i > 4:
                        self.board[i, j] = 2
        else:
            self.board = board

    def _p1_pieces(self):
        return (self.board == 1) | (self.board == 11)
    def _p2_pieces(self):
        return (self.board == 2) | (self.board == 22)

    def winner(self):
        """
        If a player has won, return their player number.
        If we have a tie, return 0.
        Otherwise, return None.
        """
        p1mask = self._p1_pieces()
        p2mask = self._p2_pieces()
        if not np.sum(p1mask):
            return 2
        elif not np.sum(p2mask):
            return 1
        elif self.plays_without_capture > 50:
            # call this a tie
            return 0
        else:
            return None

    def state(self):
        """
        Return the current state of the game
        """
        return (self.board, self.player, self.piece_in_play, self.winner())

    def legal_moves(self):
        """
        Create a list of the possible moves, given the current game state.

        Returns: list of list, each containing three entries as tuples of integers
            (start, end, jumped)
        """
        p1mask = self._p1_pieces()
        p2mask = self._p2_pieces()
        if self.player == 1:
            pmask = p1mask
            omask = p2mask
        else:
            pmask = p2mask
            omask = p1mask

        pieces = set(zip(*(self.rows[pmask], self.cols[pmask])))
        others = set(zip(*(self.rows[omask], self.cols[omask])))
        occupied = pieces.union(others)

        def _row_moves(i, j):
            if self.board[i, j] > 10:
                # is kinged, can move either way
                return [-1, 1]
            elif self.player == 1:
                # can only move down
                return [1]
            else:
                # can only move up
                return [-1]

        def _possible_jumps(i, j, row_moves):
            possibilities = []
            for r in row_moves:
                # jumping to the right
                if (i + 2*r < 8 and i + 2*r >= 0 and j < 6) and\
                   (i + r, j + 1) in others and (i + 2*r, j + 2) not in occupied:
                    possibilities += [[(i, j), (i + 2*r, j + 2), (i + r, j + 1)]]
                # jumping to the left
                elif (i + 2*r < 8 and i + 2*r >= 0 and j > 1) and\
                     (i + r, j - 1) in others and (i + 2*r, j - 2) not in occupied:
                    possibilities += [[(i, j), (i + 2*r, j - 2), (i + r, j - 1)]]
            return possibilities

        def _possible_normal_moves(i, j, row_moves):
            return [((i, j), (i + r, j + c), None)
                     for r in row_moves
                     for c in [-1, 1]
                     if min(i + r, j + c) >= 0
                     and max(i + r, j + c) < 8
                     and (i + r, j + c) not in occupied]

        if self.piece_in_play:
            i, j = self.piece_in_play
            possibilities = _possible_jumps(i, j, _row_moves(i, j))
        else:
            possibilities = []
            # find all possible jump moves first
            for i, j in pieces:
                possibilities += _possible_jumps(i, j, _row_moves(i, j))
            # if no jump moves exist, and we're not in the middle of a jump, search for normal moves
            if not possibilities:
                for i, j in pieces:
                    possibilities += _possible_normal_moves(i, j, _row_moves(i, j))

        return possibilities

    def _switch_player(self):
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

    def single_move(self, start, end, jumped=None):
        """
        Complete a single move.
        Does not check for legality of move.
        Does perform some book-keeping about which player's turn it is and
        (if we're in the middle of a jump move) which piece is currently in play.

        Args:
            start (tuple of ints): row, column of piece's starting point
            end (tuple of ints): same, for piece's ending point
            jumped (tuple of ints): same; if given, will remove this piece
        """
        board = self.board.copy()
        board[end[0], end[1]] = self.board[start[0], start[1]]
        board[start[0], start[1]] = 0
        if self.player == 1:
            if end[0] == 7:
                # king me
                board[end[0], end[1]] = 11
        else:
            if end[0] == 0:
                # king me
                board[end[0], end[1]] = 22
        if jumped is not None:
            # got 'em
            board[jumped[0], jumped[1]] = 0
            self.piece_in_play = end
            self.plays_without_capture = 0
        else:
            # end of move: no pieces in play, and switch the user
            self.piece_in_play = None
            self._switch_player()
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
        picture = ''
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                picture += chars[self.board[i, j]]
            picture += '{}\n'.format(chars[-1])
        picture += chars['reset']
        print(picture)

    def next(self):
        """
        Take the next move, randomly chosen.
        Handles some book-keeping about game state.
        """
        win = None
        _moves = self.legal_moves()
        if self.piece_in_play and not _moves:
            # the extended jump move is over
            self._switch_player()
            self.piece_in_play = None
            _moves = self.legal_moves()
        if not _moves:
            # player has no available moves; other player wins
            self._switch_player()
            win = self.player
        else:
            # randomly play a move
            self.single_move(*choice(_moves))
            self.display()
            win = self.winner()
        if win is not None:
            print('Game is over!')
            if win == 1:
                winner = 'red'
            elif win == 2:
                winner = 'black'
            elif win == 0:
                winner = 'tie'
            print('Winner:', winner)
            return win

    def run_random(self):
        """
        Run a whole game out, with each player randomly choosing amongst legal moves.
        """
        while True:
            res = self.next()
            if res is not None:
                break

