from Board import Board
from Bot import SARSA, Qlearning, RandomPlayer, RealPlayer, LazyPlayer
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class Game:
    def __init__(self, board, player1, player2, change_p=0.5):
        self.board = board
        self.player1 = player1
        self.player2 = player2
        self.changed = 1
        self.change_p = change_p

    def play(self, draw=False, update_players=True):
        if random.random() > self.change_p:
            self.changed = -1 * self.changed
            self.player1, self.player2 = self.player2, self.player1
        self.board.reset()
        game_end = None
        reward = 0
        prev_state_p2 = None
        prev_action_p2 = None
        if draw:
            self.board.draw_board()
        prev_state_p1 = self.board.get_state()
        prev_action_p1 = self.player1.make_move(self.board.available_moves(), prev_state_p1)

        while game_end is None:
            if draw:
                self.board.draw_board()
            self.board.play_turn(prev_action_p1, 1)

            game_end = self.board.check_end()
            reward = 0
            if game_end is not None:
                reward = game_end
                break

            if draw:
                self.board.draw_board()

            new_state_p2 = self.board.get_state()
            new_action_p2 = self.player2.make_move(self.board.available_moves(), new_state_p2)
            if update_players:
                self.player2.update(new_state_p2, prev_state_p2, new_action_p2, prev_action_p2, -1 * self.changed * reward, self.board.available_moves())
            self.board.play_turn(new_action_p2, -1)
            prev_state_p2 = new_state_p2
            prev_action_p2 = new_action_p2

            if draw:
                self.board.draw_board()

            game_end = self.board.check_end()
            reward = 0
            if game_end is not None:
                reward = game_end
                break

            new_state_p1 = self.board.get_state()
            new_action_p1 = self.player1.make_move(self.board.available_moves(), new_state_p1)
            if update_players:
                self.player1.update(prev_state_p1, new_state_p1, prev_action_p1, new_action_p1, self.changed * reward, self.board.available_moves())

            prev_state_p1 = new_state_p1
            prev_action_p1 = new_action_p1



        if update_players:
            self.player1.update(prev_state_p1, None, prev_action_p1, None, self.changed * reward, self.board.available_moves())
            self.player2.update(prev_state_p2, None, prev_action_p2, None, -1 * self.changed * reward, self.board.available_moves())
        if draw:
            self.board.draw_board()
        return self.changed * reward


board = Board()
bot1 = Qlearning(alpha=0.5, gamma=0.9, eps=0.1)
bot2 = RandomPlayer()
game = Game(board, bot1, bot2, change_p=1)  # change_p ustaw na 1 jak chcesz żeby zawsze zaczynali w tej samej kolejności
r = []
epochs = 10000
for i in tqdm(range(epochs)):
    r.append(game.play(draw=False, update_players=True))

plt.figure()
plt.plot([r[:i].count(1) / len(r[:i]) for i in tqdm(range(1, epochs))], label='Player1 wins')
plt.plot([r[:i].count(0) / len(r[:i]) for i in tqdm(range(1, epochs))], label='Draw')
plt.plot([r[:i].count(-1) / len(r[:i]) for i in tqdm(range(1, epochs))], label='Player2 wins')
plt.legend()
plt.show()

while True:
    game = Game(board, bot1, RealPlayer(), change_p=1)
    game.play(draw=True, update_players=False)
    game = Game(board, RealPlayer(), bot1, change_p=1)
    game.play(draw=True, update_players=False)