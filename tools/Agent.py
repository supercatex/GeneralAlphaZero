from abc import ABC, abstractmethod
from tools import Environment, Action


class Agent(ABC):
    def __init__(self, index: int):
        self.index: int = index

    @abstractmethod
    def action(self, env: Environment) -> Action:
        pass

    @abstractmethod
    def reset(self):
        pass
