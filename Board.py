import numpy as np


class Board:
    def __init__(self):
        self.state = [0] * 9

    def reset(self):
        self.state = [0] * 9

    def play_turn(self, position, sign):
        self.state[position] = sign

    def available_moves(self):
        moves = [i for i, e in enumerate(self.state) if e == 0]
        return moves

    def get_state(self):
        state = ''
        for i in self.state:
            if i == 1:
                state += 'O'
            elif i == -1:
                state += 'X'
            else:
                state += ' '
        return state

    def check_end(self):
        state = np.array(self.state).reshape((3, 3))
        best = max(list(state.sum(axis=0)) +  # columns
                   list(state.sum(axis=1)) +  # rows
                   [state.trace()] +  # main diagonal
                   [np.fliplr(state).trace()],  # other diagonal
                   key=abs)
        if abs(best) == state.shape[0]:  # assumes square board
            return np.sign(best)  # winning player, +1 or -1
        if len(self.available_moves()) == 0:
            return 0  # a draw (otherwise, return None by default)

    def draw_board(self):
        state = []
        for i in self.state:
            if i == 0:
                state.append(' ')
            elif i == -1:
                state.append('X')
            else:
                state.append('O')
        print(f'{state[0]} | {state[1]} | {state[2]}')
        print('---------')
        print(f'{state[3]} | {state[4]} | {state[5]}')
        print('---------')
        print(f'{state[6]} | {state[7]} | {state[8]}')
        print('***************************')
