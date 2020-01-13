from typing import List
from abc import ABC, abstractmethod


class Observation(ABC):
    def __init__(self):
        self.data: List = list()
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def input_shape(self):
        pass

    @abstractmethod
    def nb_actions(self) -> int:
        pass

    @abstractmethod
    def record_encode(self) -> str:
        pass

    @abstractmethod
    def record_decode(self, data: str) -> List:
        pass
