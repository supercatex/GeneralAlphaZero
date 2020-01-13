import os
import time
import json
from glob import glob
from typing import List
from datetime import datetime
from tools import Environment, Agent


class SelfPlayWorker(object):
    def __init__(self, env: Environment):
        self.env: Environment = env
        self.buffer: List = []

        self.nb_game_in_file = 100
        self.play_data_dir = "play_data"
        self.play_data_filename_tmpl = "play_%s.json"
        self.max_file_num = 50

    def start(self):
        self.buffer = []

        for i in range(1, self.nb_game_in_file + 1):
            start_time = time.time()
            self.start_game(i)
            end_time = time.time()
            print(f"Game: {i} -- Spent:{end_time - start_time}.")

    def start_game(self, idx):
        self.env.reset()
        while not self.env.done:
            agent: Agent = self.env.current_agent()
            action = agent.action(self.env)
            self.env.step(action)
            # print(self.env.observation, self.env.turn, self.env.winner)
        self.finish_game()
        self.save_play_data(write=idx % self.nb_game_in_file == 0)
        self.remove_play_data()

    def save_play_data(self, write=True):
        data = self.env.agents[0].moves + self.env.agents[1].moves
        self.buffer += data

        if not write:
            return

        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(self.play_data_dir, self.play_data_filename_tmpl % game_id)
        if not os.path.exists(self.play_data_dir):
            os.mkdir(self.play_data_dir)
        with open(path, "wt") as f:
            json.dump(self.buffer, f)
        self.buffer = []

    def remove_play_data(self):
        pattern = os.path.join(self.play_data_dir, self.play_data_filename_tmpl % "*")
        files = list(sorted(glob(pattern)))
        if len(files) < self.max_file_num:
            return
        for i in range(len(files) - self.max_file_num):
            os.remove(files[i])

    def finish_game(self):
        if self.env.winner == 1:
            black_win = 1
        elif self.env.winner == 2:
            black_win = -1
        else:
            black_win = 0

        for agent in self.env.agents:
            if agent.index == 1:
                agent.finish_game(black_win)
            else:
                agent.finish_game(-black_win)
