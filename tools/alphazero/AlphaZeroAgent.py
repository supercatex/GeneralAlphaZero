from abc import abstractmethod
from typing import List
from collections import defaultdict, namedtuple
from asyncio.queues import Queue
import asyncio
import numpy as np
from tools import Agent, Environment, Action
from tools.alphazero import AlphaZeroModel


CounterKey = namedtuple("CounterKey", "board next_player")
QueueItem = namedtuple("QueueItem", "state future")
HistoryItem = namedtuple("HistoryItem", "action policy values visit")


class AlphaZeroAgent(Agent):
    def __init__(self, index: int, model: AlphaZeroModel, env: Environment):
        super().__init__(index)

        self.prediction_queue_size = 16
        self.parallel_search_num = 2
        self.thinking_loop = 2
        self.logging_thinking = False
        self.change_tau_turn = 5
        self.simulation_num_per_move = 50
        self.prediction_worker_sleep_sec = 0.0001
        self.wait_for_expanding_sleep_sec = 0.00001
        self.virtual_loss = 3
        self.noise_eps = 0.1
        self.dirichlet_alpha = 0.03
        self.c_puct = 1.5

        self.model: AlphaZeroModel = model
        self.env: Environment = env
        self.labels_n = self.env.observation.nb_actions()
        self.var_n = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.var_w = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.var_q = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.var_u = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.var_p = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.expanded = set()
        self.now_expanding = set()
        self.prediction_queue = Queue(self.prediction_queue_size)
        self.sem = asyncio.Semaphore(self.parallel_search_num)

        self.moves: List[List] = []
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0

        self.thinking_history = {}  # for fun

    def action(self, env: Environment) -> Action:
        new_env: Environment = self.env.new_env(env.observation.data, env.agents)
        key = self.counter_key(new_env)

        action: int = -1
        policy = None
        action_by_value: int = -1
        for tl in range(self.thinking_loop):
            if tl > 0 and self.logging_thinking:
                print(f"continue thinking: policy move=({action % 8}, {action // 8}), "
                      f"value move=({action_by_value % 8}, {action_by_value // 8})")
            self.search_moves(env)
            policy = self.calc_policy(env)
            action = int(np.random.choice(range(self.labels_n), p=policy))
            action_by_value = int(np.argmax(self.var_q[key] + (self.var_n[key] > 0) * 100))
            if action == action_by_value or new_env.turn < self.change_tau_turn:
                break

        # this is for play_gui, not necessary when training.
        self.thinking_history[new_env.observation] = HistoryItem(
            action,
            policy,
            list(self.var_q[key]),
            list(self.var_n[key])
        )

        self.moves.append([new_env.observation.record_encode(), [policy.tolist()]])
        return Action(action)

    def reset(self):
        self.moves = []

    def calc_policy(self, env: Environment):
        """calc π(a|s0)
        :return:
        """
        new_env: Environment = self.env.new_env(env.observation.data, env.agents)
        key = self.counter_key(new_env)
        if new_env.turn < self.change_tau_turn:
            return self.var_n[key] / np.sum(self.var_n[key])  # tau = 1
        else:
            action = np.argmax(self.var_n[key])  # tau = 0
            ret = np.zeros(self.labels_n)
            ret[action] = 1
            return ret

    @staticmethod
    def counter_key(env: Environment):
        return CounterKey(str(env.observation.data), env.turn)

    def search_moves(self, env: Environment):
        loop = self.loop
        self.running_simulation_num = 0

        coroutine_list = []
        for it in range(self.simulation_num_per_move):
            cor = self.start_search_my_move(env)
            coroutine_list.append(cor)

        coroutine_list.append(self.prediction_worker())
        loop.run_until_complete(asyncio.gather(*coroutine_list))

    async def prediction_worker(self):
        """For better performance, queueing prediction requests and predict together in this worker.

        speed up about 45sec -> 15sec for example.
        :return:
        """
        q = self.prediction_queue
        margin = 10  # avoid finishing before other searches starting.
        while self.running_simulation_num > 0 or margin > 0:
            if q.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(self.prediction_worker_sleep_sec)
                continue
            item_list = [q.get_nowait() for _ in range(q.qsize())]
            # logger.debug(f"predicting {len(item_list)} items")
            data = np.array([x.state for x in item_list])
            policy_ary, value_ary = self.model.predict(data)
            for p, v, item in zip(policy_ary, value_ary, item_list):
                item.future.set_result((p, v))

    async def start_search_my_move(self, env: Environment):
        self.running_simulation_num += 1
        with await self.sem:  # reduce parallel search number
            new_env: Environment = self.env.new_env(env.observation.data, env.agents)
            leaf_v = await self.search_my_move(new_env, is_root_node=True)
            self.running_simulation_num -= 1
            return leaf_v

    async def search_my_move(self, env: Environment, is_root_node=False):
        """

        Q, V is value for this Player(always white).
        P is value for the player of next_player (black or white)
        :param env:
        :param is_root_node:
        :return:
        """
        if env.done:
            # return env.winner
            if env.winner == 1:
                return 1
            elif env.winner == 2:
                return -1
            else:
                return 0

        key = self.counter_key(env)

        while key in self.now_expanding:
            await asyncio.sleep(self.wait_for_expanding_sleep_sec)

        # is leaf?
        if key not in self.expanded:  # reach leaf node
            leaf_v = await self.expand_and_evaluate(env)
            if env.current_agent_index() == 1:
                return leaf_v  # Value for white
            else:
                return -leaf_v  # Value for white == -Value for white

        action_t = self.select_action_q_and_u(env, is_root_node)
        _, _ = env.step(Action(action_t))

        virtual_loss = self.virtual_loss
        self.var_n[key][action_t] += virtual_loss
        self.var_w[key][action_t] -= virtual_loss
        leaf_v = await self.search_my_move(env)  # next move

        # on returning search path
        # update: N, W, Q, U
        n = self.var_n[key][action_t] = self.var_n[key][action_t] - virtual_loss + 1
        w = self.var_w[key][action_t] = self.var_w[key][action_t] + virtual_loss + leaf_v
        self.var_q[key][action_t] = w / n
        return leaf_v

    @abstractmethod
    def valid_actions(self, env: Environment) -> List[int]:
        pass

    async def expand_and_evaluate(self, env: Environment):
        """expand new leaf

        update var_p, return leaf_v

        :param ChessEnv env:
        :return: leaf_v
        """
        key = self.counter_key(env)
        self.now_expanding.add(key)

        planes = env.planes()
        black_ary, white_ary = planes[0], planes[1]
        state = [black_ary, white_ary] if env.current_agent_index() == 2 else [white_ary, black_ary]
        future = await self.predict(np.array(state))
        await future
        leaf_p, leaf_v = future.result()

        self.var_p[key] = leaf_p  # P is value for next_player (black or white)
        self.expanded.add(key)
        self.now_expanding.remove(key)
        return float(leaf_v)

    async def predict(self, x):
        future = self.loop.create_future()
        item = QueueItem(x, future)
        await self.prediction_queue.put(item)
        return future

    def select_action_q_and_u(self, env: Environment, is_root_node):
        key = self.counter_key(env)

        legal_moves = self.valid_actions(env)

        # noinspection PyUnresolvedReferences
        xx_ = np.sqrt(np.sum(self.var_n[key]))  # SQRT of sum(N(s, b); for all b)
        xx_ = max(xx_, 1)  # avoid u_=0 if N is all 0
        p_ = self.var_p[key]

        if is_root_node:  # Is it correct?? -> (1-e)p + e*Dir(0.03)
            p_ = (1 - self.noise_eps) * p_ + \
                 self.noise_eps * np.random.dirichlet([self.dirichlet_alpha] * self.labels_n)

        u_ = self.c_puct * p_ * xx_ / (1 + self.var_n[key])
        if env.current_agent_index() == 1:
            v_ = (self.var_q[key] + u_ + 1000) * legal_moves
        else:
            # When enemy's selecting action, flip Q-Value.
            v_ = (-self.var_q[key] + u_ + 1000) * legal_moves

        # noinspection PyTypeChecker
        action_t = int(np.argmax(v_))
        return action_t

    def finish_game(self, z):
        """

        :param z: win=1, lose=-1, draw=0
        :return:
        """
        for move in self.moves:  # add this game winner result to all past moves.
            if len(move) == 2:
                move += [z]
