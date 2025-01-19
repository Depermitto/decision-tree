from .decision_tree import DecisionTree
import numpy as np


class ClassicTree(DecisionTree):
    def __init__(self, features, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        super().__init__(features, max_depth, min_samples_split, min_samples_leaf)

    def choose_test(self, info_gains) -> np.int64:
        """
        Choose the best test based on the information gains.
        Classic tree chooses the test with the highest information gain.
        Args:
            info_gains (ArrayLike): Information gains of the tests
        Returns:
            int: the indef of the chosen test
        """
        return np.argmax(info_gains)
