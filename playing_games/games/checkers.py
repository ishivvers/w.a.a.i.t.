import numpy as np

class board():

    def __init__(self):
        self.player = 1  # 1 (red) or 2 (black)
        size = 8
        self.board = np.zeros((size, size), dtype=np.int8)
        self.cols, self.rows = np.meshgrid(range(size), range(size))
        # mask out the unavailable spots, and put pieces down
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if (i + j) % 2:
                    self.board[i, j] = -1
                elif i <= 2:
                    self.board[i, j] = 1
                elif i > 4:
                    self.board[i, j] = 2

    def _p1_pieces(self):
        return (self.board == 1) | (self.board == 11)
    def _p2_pieces(self):
        return (self.board == 2) | (self.board == 22)

    def winner(self):
        p1mask = self._p1_pieces()
        p2mask = self._p2_pieces()
        if not np.sum(p1mask):
            return 2
        elif not np.sum(p2mask):
            return 1
        else:
            return None

    def state(self):
        """
        Return the current state of the game
        """
        return (self.board, self.winner(), self.player)

    def legal_moves(self):
        """
        Create a list of the possible moves, given the current game state.

        Returns: list of list, each containing three entries (start, end, jumped)
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

        def _possible_jumps(i, j, row_moves):
            possibilities = []
            for r in row_moves:
                # jumping to the right
                if (i + 2*r < 8 and i + 2*r >= 0 and j < 6) and\
                   (i + r, j + 1) in others and (i + 2*r, j + 2) not in occupied:
                    possibilities += [[(i, j), (i + 2*r, j + 2), [(i + r, j + 1)]]]
                # jumping to the left
                elif (i + 2*r < 8 and i + 2*r >= 0 and j > 1) and\
                     (i + r, j - 1) in others and (i + 2*r, j - 2) not in occupied:
                    possibilities += [[(i, j), (i + 2*r, j - 2), [(i + r, j - 1)]]]
            return possibilities

        possibilities = []
        for p in pieces:
            i, j = p
            if self.board[i, j] > 10:
                # is kinged, can move either way
                row_moves = [-1, 1]
            elif self.player == 1:
                # can only move down
                row_moves = [1]
            else:
                # can only move up
                row_moves = [-1]
            # find all the normal moves
            possibilities += [(p, (i + r, j + c), None) for r in row_moves for c in [-1, 1] 
                              if min(i + r, j + c) >= 0 and max(i + r, j + c) < 8 and (i + r, j + c) not in occupied]
            # find all the jumps: does not yet include multiple jumps in a row
            jumps = _possible_jumps(i, j, row_moves)
            possibilities += jumps
        return possibilities

    def move(self, start, end, jumped=None):
        """
        Complete a move.  Does not check for legality of move.

        Args:
            start (tuple of ints): column, row of piece's starting point
            end (tuple of ints): column, row of piece's ending point
            jumped list of (tuples of ints): if given, will remove these pieces
        """
        board = self.board.copy()
        board[end[0], end[1]] = self.board[start[0], start[1]]
        board[start[0], start[1]] = 0
        if jumped is not None:
            # got 'em
            for j in jumped:
                board[j[0], j[1]] = 0
        if self.player == 1:
            if end[0] == 7:
                # king me
                board[end[0], end[1]] = 11
            self.player = 2
        else:
            if end[0] == 0:
                # king me
                board[end[0], end[1]] = 22
            self.player = 1
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
