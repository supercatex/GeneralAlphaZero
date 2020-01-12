from abc import ABC, abstractmethod
from typing import List, Tuple
from tools import Observation
from tools import Agent
from tools import Action
import numpy as np


class Environment(ABC):
    def __init__(self, observation: Observation):
        self.observation: Observation = observation
        self.turn: int = 0
        self.done: bool = False
        self.winner: int = -1
        self.agents: List[Agent] = list()

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def reset(self):
        self.observation.reset()
        for agent in self.agents:
            agent.reset()
        self.turn = 0
        self.done = False

    @abstractmethod
    def new_env(self, data, agents: List) -> object:
        pass

    def nb_agent(self) -> int:
        return len(self.agents)

    def current_agent_index(self) -> int:
        assert self.nb_agent() != 0, "No Agent!"
        return self.turn % self.nb_agent()

    def current_agent(self) -> Agent:
        return self.agents[self.current_agent_index()]

    def planes(self):
        data = []
        for agent in self.agents:
            plane = np.copy(self.observation.data)
            shape = plane.shape
            plane = plane.reshape(-1)
            for i in range(len(plane)):
                if plane[i] != agent.index:
                    plane[i] = 0
                else:
                    plane[i] = 1
            plane = plane.reshape(shape)
            data.append(plane)
        return data

    @abstractmethod
    def step(self, action: Action) -> Tuple:
        pass
