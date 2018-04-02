import datetime
from random import choice
import logging
from threading import Thread
import numpy as np

class MCTS(object):
    """
    Monte Carlo Tree Search
    """
    def __init__(self, board, root=None, tree=None):
        """
        Args:
            board: an instance of the board class, having the appropriate API.
            root (board state): the starting point within the tree. If not given,
                will begin runs at the current board state.
            tree (MCTS tree): an already-extant tree whithin which to run further
                runs.  If not given, will instantiate a new tree starting from the root.
        """
        self.log = logging.Logger('MCTS')
        self.log.setLevel(logging.INFO)
        self.log.addHandler(logging.StreamHandler())

        self.board = board
        if root == None:
            self.root = self.board.state()
        else:
            self.root = root
        children, moves = self._get_children_moves(self.root)
        if tree == None:
            self.tree = {self.root: {
                'parent': None,
                'children': children,
                'moves': moves,
                'plays': 0,
                'wins': 0,
                'depth': 0,
                }
            }
        else:
            self.tree = tree
            if self.root not in self.tree:
                self.tree[self.root] = {
                    'parent': None,
                    'children': children,
                    'moves': moves,
                    'plays': 0,
                    'wins': 0,
                    'depth': 0,
                }
            if not self.tree[self.root].get('children'):
                self.tree[self.root]['children'] = children
                self.tree[self.root]['moves'] = moves

    def _print_branch(self, branch):
        print('\n{}: (player: {}) \n parent: {}\n children: {}\n depth: {}\n wins/plays: {}/{}'.format(
            id(branch), self.board.get_player(branch), id(self.tree[branch]['parent']),
            [(self.tree[branch]['moves'][i], id(child)) for i, child in enumerate(self.tree[branch]['children'])],
            self.tree[branch]['depth'], self.tree[branch]['wins'], self.tree[branch]['plays']))
    def print_tree(self):
        """
        Starting at the current root, print a representation of the tree.
        """
        self._print_branch(self.root)
        children = self.tree[self.root]['children']
        while children:
            branch = children.pop(0)
            if self.tree.get(branch, None):
                self._print_branch(branch)
                children += self.tree[branch]['children']

    def _get_children_moves(self, state):
        self.log.debug('Populating children')
        children_nodes = []
        self.board.load_state(state)
        legal_moves = self.board.legal_moves()
        for move in legal_moves:
            self.board.load_state(state)
            self.board.single_move(*move)
            next_state = self.board.state()
            children_nodes.append(next_state)
        self.board.load_state(state)
        return children_nodes, legal_moves

    def _choose_child(self, branch):
        # if any of the children have not yet been seen, choose them
        for child in self.tree[branch]['children']:
            if self.tree.get(child, None) == None:
                self.log.debug('Choosing a quiet child')
                grandchildren, moves = self._get_children_moves(child)
                self.tree[child] = {
                    'parent': branch,
                    'children': grandchildren,
                    'moves': moves,
                    'plays': 0,
                    'wins': 0,
                    'depth': self.tree[branch]['depth'] + 1
                }
                return child
        # otherwise, choose via the UCB1 formula:
        return self.tree[branch]['children'][self._ucb1(branch)]

    def _ucb1(self, branch, C=1.4):
        self.log.debug('Choosing the best child')
        plays = np.array([self.tree[child]['plays'] for child in self.tree[branch]['children']])
        wins = np.array([self.tree[child]['wins'] for child in self.tree[branch]['children']])
        ucb1 = wins / plays + C * (np.log(np.sum(plays)) / plays) ** 0.5
        return np.argmax(ucb1)

    def selection_expansion_simulation(self):
        """
        Starting from the current root, navigate the tree.
        Returns the leaf node and the win state.
        Upon hitting a leaf node that is not the end of the game, will expand the tree
          by one and then randomly play the game out from there, returning the new leaf and the win.
        Upon hitting a leaf node that is the end of the game, will not expand the tree,
          and instead will return the already-extant leaf and the win.
        """
        self.log.debug('Traversing the tree')
        branch = self.root
        while self.tree[branch]['plays'] > 0:
            self.log.debug('Descending a level')
            branch = self._choose_child(branch)
            self.board.load_state(branch)
            winner, _ = self.board.win_or_moves()
            if winner is not None:
                self.log.debug('Found a winner within the tree')
                return branch, winner
        # if we got here, that means we've traversed the tree to a leaf and nobody's won yet
        self.log.debug('Running a random playout')
        winner = self.board.run_random()
        return branch, winner

    def _update_stats(self, branch, winner):
        self.log.debug('Updating {}'.format(id(branch)))
        self.tree[branch]['plays'] += 1
        self.tree[branch]['wins'] += winner.get(self.board.get_player(branch), 0)

    def backpropagation(self, branch, winner):
        """
        Given a node in the tree and a win state calculated at that node,
        navigate back up the tree and update the counts.
        """
        self._update_stats(branch, winner)
        parent = self.tree[branch]['parent']
        while parent is not None:
            self._update_stats(parent, winner)
            parent = self.tree[parent]['parent']

    def run(self, event=None, t=10):
        """
        Run the MCTS

        Args:
            event (threading.Event): run the MCTS until this event is set
            t (int): if no event is given, run the MCTS for <t> seconds
        """
        n_games = 0
        if event is not None:
            while not event.is_set():
                self.backpropagation(*self.selection_expansion_simulation())
                n_games += 1
        else:
            begin = datetime.datetime.utcnow()
            dt = datetime.timedelta(seconds=t)
            while (datetime.datetime.utcnow() - begin) < dt:
                self.backpropagation(*self.selection_expansion_simulation())
                n_games += 1
        self.log.info("Played {} games.".format(n_games))

    def choose_play(self, state):
        """
        Given a board state, choose the next play based upon the MC tree in memory
        """
        plays = np.array([self.tree[child]['plays'] for child in self.tree[state]['children']])
        wins = np.array([self.tree[child]['wins'] for child in self.tree[state]['children']])
        return self.tree[state]['moves'][np.argmax(wins / plays)]
