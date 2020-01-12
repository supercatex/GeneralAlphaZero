import numpy as np
from typing import List, Tuple
from tools import Environment, Observation, Action, Agent
from tools.alphazero import AlphaZeroAgent, AlphaZeroModel


class GameState(Observation):
    EMPTY: str = "　"
    BLACK: str = "Ｘ"
    WHITE: str = "Ｏ"
    STATES: List[str] = [EMPTY, WHITE, BLACK]

    def __init__(self):
        self.nb_row: int = 6
        self.nb_col: int = 7
        super().__init__()

    def reset(self):
        self.data = []
        for _ in range(self.nb_row):
            temp: List = []
            for _ in range(self.nb_col):
                temp.append(0)
            self.data.append(temp)

    def input_shape(self) -> Tuple:
        return len(self.STATES) - 1, self.nb_row, self.nb_col

    def nb_actions(self) -> int:
        return self.nb_col

    def do_action(self, action: Action, agent: Agent):
        # print("Call do_action method:", action.index)
        for i in range(self.nb_row):
            index = action.index
            if self.data[i][index] == 0:
                self.data[i][index] = agent.index
                break

    def check_win(self) -> int:
        def check_cell(ii, jj, pp, ll) -> int:
            if not 0 <= ii < self.nb_row or not 0 <= jj < self.nb_col:
                return 0
            if self.data[ii][jj] == pp:
                ll += 1
            else:
                ll = 0
            return ll

        for p in range(1, len(self.STATES)):
            for i in range(self.nb_row):                # -
                length = 0
                for j in range(self.nb_col):
                    length = check_cell(i, j, p, length)
                    if length >= 4:
                        return p

            for j in range(self.nb_col):                # |
                length = 0
                for i in range(self.nb_row):
                    length = check_cell(i, j, p, length)
                    if length >= 4:
                        return p

            for j in range(self.nb_col):                # Top Right
                length = 0
                for i in range(self.nb_row):
                    length = check_cell(i, j + i, p, length)
                    if length >= 4:
                        return p

            for j in range(self.nb_col - 1, -1, -1):    # Top Left
                length = 0
                for i in range(self.nb_row):
                    length = check_cell(i, j - i, p, length)
                    if length >= 4:
                        return p

            for i in range(self.nb_row):                # Bottom Left
                length = 0
                for j in range(self.nb_col):
                    length = check_cell(i + j, j, p, length)
                    if length >= 4:
                        return p

            for i in range(self.nb_row - 1, -1, -1):    # Bottom Right
                length = 0
                for j in range(self.nb_col):
                    length = check_cell(i - j, j, p, length)
                    if length >= 4:
                        return p

        return -1

    def record_encode(self) -> str:
        s = ""
        for row in self.data:
            for col in row:
                s += self.STATES[col]
        return s

    def __str__(self):
        s = ""
        for i in range(self.nb_col):
            s += "_＿"
        s += "_\n"
        for row in self.data:
            for col in row:
                s += "|" + self.STATES[col]
            s += "|\n"
        asc2 = ord("１")
        for i in range(self.nb_col):
            s += "*" + chr(asc2 + i)
        s += "*"
        return s


class GameAgent(AlphaZeroAgent):
    def __init__(self, index: int, model: AlphaZeroModel, env: Environment):
        super().__init__(index, model, env)

    def valid_actions(self, env: Environment) -> List[int]:
        # noinspection PyTypeChecker
        state: GameState = env.observation
        actions = []
        for _ in range(state.nb_col):
            actions.append(0)
        for j in range(state.nb_col):
            if state.data[state.nb_row - 1][j] == 0:
                actions[j] = 1
        return actions


class HumanAgent(Agent):

    def action(self, env: Environment) -> Action:
        s = input("Your turn: ")
        return Action(int(s))

    def reset(self):
        pass


class GameEnv(Environment):
    def __init__(self):
        self.state = GameState()
        super().__init__(self.state)
        # noinspection PyTypeChecker
        self.observation: GameState = self.state

    def step(self, action: Action) -> Tuple:
        if action.index is None:
            self.done = True
            return self.observation.data, {}

        state: GameState = self.observation
        agent: Agent = self.current_agent()

        state.do_action(action, agent)
        self.winner = state.check_win()
        self.turn += 1

        if self.winner != -1:
            self.done = True
        else:
            if self.turn == state.nb_col * state.nb_row:
                self.done = True

        return self.observation.data, {}

    def new_env(self, data, agents: List) -> object:
        return _new_env(data, agents)


def _new_env(data, agents: List) -> GameEnv:
    env = GameEnv()
    env.observation = GameState()
    env.observation.data = np.copy(data)
    env.turn = 0
    shape = env.observation.data.shape
    for i in env.observation.data.reshape(-1):
        if i != 0:
            env.turn += 1
    env.observation.data.reshape(shape)
    env.done = False
    env.winner = False
    env.agents = agents
    return env

# class GameOptimizer(AlphaZeroOptimizer):
#
#     def convert_to_training_data(self, data):
#         state_list = []
#         policy_list = []
#         z_list = []
#         for state, policy, z in data:
#             board = list(state)
#             board = np.reshape(board, (6, 7))
#             env: GameEnv = GameEnv().new_env(board, [GameAgent(1), GameAgent(2)])
#
#             planes = env.planes()
#             black_ary, white_ary = planes[0], planes[1]
#             state = [black_ary, white_ary] if env.current_agent_index() == 2 else [white_ary, black_ary]
#
#             state_list.append(state)
#             policy_list.append(policy)
#             z_list.append(z)
#
#         return np.array(state_list), np.array(policy_list), np.array(z_list)


if __name__ == "__main__":
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=0.2,
            allow_growth=None,
        )
    )
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    _env = GameEnv()
    _model = AlphaZeroModel(_env.observation)
    _model.load("model_best_config.json", "model_best_weight.h5")

    _p1 = GameAgent(1, _model, _env)
    _p2 = GameAgent(2, _model, _env)
    _env.add_agent(_p1)
    _env.add_agent(_p2)

    from tools.alphazero import SelfPlayWorker
    _worker = SelfPlayWorker(_env)
    _worker.start()

    # _optimizer = GameOptimizer(_env, "play_data")
    # _optimizer.training()
