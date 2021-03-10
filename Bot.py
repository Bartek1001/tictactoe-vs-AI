import random
import numpy as np
from collections import defaultdict


class Bot:
    def __init__(self, alpha=0.5, gamma=0.9, eps=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = eps
        self.rewards = []

        self.Q = {}
        for action in range(9):
            self.Q[action] = defaultdict(int)

    def make_move(self, posible_moves, s):
        if random.random() < self.epsilon:
            action = random.choice(posible_moves)
        else:
            values = np.array([self.Q[a][s] for a in posible_moves])
            # Find location of max
            ix_max = np.where(values == np.max(values))[0]
            if len(ix_max) > 1:
                # If multiple actions were max, then sample from them
                ix_select = np.random.choice(ix_max, 1)[0]
            else:
                # If unique max action, select that one
                ix_select = ix_max[0]
            action = posible_moves[ix_select]
        return action


class Qlearning(Bot):
    def __init__(self, alpha=0.5, gamma=0.9, eps=0.1):
        super().__init__(alpha, gamma, eps)

    def update(self, s, s_, a, a_, r, posible_moves):
        if s_ is not None:
            # hold list of Q values for all a_,s_ pairs. We will access the max later
            Q_options = [self.Q[action][s_] for action in posible_moves]
            # update
            self.Q[a][s] += self.alpha * (r + self.gamma * max(Q_options) - self.Q[a][s])
        else:
            # terminal state update
            self.Q[a][s] += self.alpha * (r - self.Q[a][s])

        # add r to rewards list
        self.rewards.append(r)


class SARSA(Bot):
    def __init__(self, alpha=0.5, gamma=0.9, eps=0.1):
        super().__init__(alpha, gamma, eps)

    def update(self, s, s_, a, a_, r, *args):

        if s_ is not None:
            self.Q[a][s] += self.alpha * (r + self.gamma * self.Q[a_][s_] - self.Q[a][s])
        else:
            # terminal state update
            self.Q[a][s] += self.alpha * (r - self.Q[a][s])

        # add r to rewards list
        self.rewards.append(r)


class RealPlayer(Bot):
    def __init__(self):
        super().__init__()

    def update(self, *args):
        pass

    def make_move(self, possible_moves, *args):
        pos = int(input('Wybierz pole: '))
        while pos not in possible_moves:
            pos = int(input('Błąd! Wybierz pole: '))
        return pos


class RandomPlayer(Bot):
    def __init__(self):
        super().__init__()

    def make_move(self, possible_moves, *args):
        return random.choice(possible_moves)

    def update(self, *args):
        pass


class LazyPlayer(Bot):
    def __init__(self):
        super().__init__()

    def make_move(self, possible_moves, *args):
        return possible_moves[0]

    def update(self, *args):
        pass
