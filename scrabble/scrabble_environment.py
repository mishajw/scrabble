from enum import Enum
import numpy as np

import tensorforce.environments


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


letters = ["a", "b", "c", "d", "e"]


class ScrabbleEnvironment(tensorforce.environments.Environment):
    def __init__(self, board_size: int):
        self.board_size = board_size

    def reset(self):
        return np.zeros(self.__state_shape, dtype=float)

    def execute(self, actions):
        return np.zeros(self.__state_shape, dtype=float), False, 0

    @property
    def states(self):
        return dict(shape=self.__state_shape, type="float")

    @property
    def actions(self):
        return dict(type="float", shape=self.__num_actions, min_value=0, max_value=1)

    @property
    def __num_actions(self):
        return len(Direction) + len(letters)

    @property
    def __state_shape(self):
        return (self.board_size * self.board_size * self.__num_actions,)

    def __str__(self):
        return "ScrabbleEnvironment"
