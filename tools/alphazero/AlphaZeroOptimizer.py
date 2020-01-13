import os
import json
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
from tools import Environment
from tools.alphazero import AlphaZeroModel
from tools.alphazero.AlphaZeroModel import objective_function_for_policy, objective_function_for_value


class AlphaZeroOptimizer(ABC):
    def __init__(self, env: Environment, data_path: str):
        self.model: AlphaZeroModel = None
        self.optimizer = None
        self.loaded_data = {}
        self.dataset = None
        self.env: Environment = env
        self.data_path = data_path
        self.epochs = 1
        self.batch_size = 32
        self.save_model_steps = 300
        self.next_generation_model_dir = os.path.join("", "next_generation")
        self.next_generation_model_dirname_tmpl = "model_%s"
        self.next_generation_model_config_filename = "model_config.json"
        self.next_generation_model_weight_filename = "model_weight.h5"

    def training(self):
        self.model = self.load_model()
        self.compile_model()
        self.load_data(self.data_path)

        total_steps = 0
        while True:
            self.update_learning_rate(total_steps)
            steps = self.train_epoch(self.epochs)
            total_steps += steps
            if (total_steps // steps) % self.save_model_steps == 0:
                self.save_current_model()

    def train_epoch(self, epochs):
        state_ary, policy_ary, z_ary = self.dataset
        self.model.model.fit(state_ary, [policy_ary, z_ary],
                             batch_size=self.batch_size,
                             epochs=epochs)
        steps = (state_ary.shape[0] // self.batch_size) * epochs
        return steps

    def save_current_model(self):
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        model_dir = os.path.join(self.next_generation_model_dir, self.next_generation_model_dirname_tmpl % model_id)
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, self.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, self.next_generation_model_weight_filename)
        self.model.save(config_path, weight_path)

    def update_learning_rate(self, total_steps):
        # The deepmind paper says
        # ~400k: 1e-2
        # 400k~600k: 1e-3
        # 600k~: 1e-4

        if total_steps < 500:
            lr = 1e-2
        elif total_steps < 2000:
            lr = 1e-3
        elif total_steps < 9000:
            lr = 1e-4
        else:
            lr = 2.5e-5  # means (1e-4 / 4): the paper batch size=2048, ours is 512.
        K.set_value(self.optimizer.lr, lr)
        print(f"total step={total_steps}, set learning rate to {lr}")

    def load_model(self, config_path: str = None, weight_path: str = None) -> AlphaZeroModel:
        model = AlphaZeroModel(self.env.observation)
        model.load(config_path, weight_path)
        return model

    def compile_model(self):
        self.optimizer = SGD(lr=1e-2, momentum=0.9)
        losses = [objective_function_for_policy, objective_function_for_value]
        self.model.model.compile(optimizer=self.optimizer, loss=losses)

    def load_data(self, path):
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), "rt") as f:
                data = json.load(f)
                print(data[0])
                print(len(data[1]))
                print(len(data))
                self.loaded_data[filename] = self.convert_to_training_data(data)

        self.dataset = self.collect_all_loaded_data()

    def collect_all_loaded_data(self):
        state_ary_list, policy_ary_list, z_ary_list = [], [], []
        for s_ary, p_ary, z_ary_ in self.loaded_data.values():
            state_ary_list.append(s_ary)
            policy_ary_list.append(p_ary)
            z_ary_list.append(z_ary_)

        state_ary = np.concatenate(state_ary_list)
        policy_ary = np.concatenate(policy_ary_list)
        z_ary = np.concatenate(z_ary_list)
        return state_ary, policy_ary, z_ary

    @abstractmethod
    def convert_to_training_data(self, data):
        pass
