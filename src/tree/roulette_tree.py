from .decision_tree import DecisionTree
import numpy as np


class RouletteTree(DecisionTree):
    def __init__(
        self, features, max_depth=None, min_samples_split=2, min_samples_leaf=5
    ):
        super().__init__(features, max_depth, min_samples_split, min_samples_leaf)

    def choose_test(self, info_gains) -> np.int64:
        """
        Choose the best test based on the information gains.
        Roulette tree chooses the test randomly based on the information gains.
        Args:
            info_gains (ArrayLike): Information gains of the tests
        Returns:
            int: the indef of the chosen test
        """
        total = np.sum(info_gains)
        probs = info_gains / total
        return np.random.choice(range(len(info_gains)), p=probs)
