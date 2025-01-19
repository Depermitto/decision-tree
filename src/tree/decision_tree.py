from abc import ABC, abstractmethod
from collections import Counter
from typing import Optional, Any
import numpy as np
import io
from itertools import combinations

from .node import Node


class DecisionTree(ABC):
    features: list[str]
    max_depth: Optional[int]
    min_samples_split: int
    min_samples_leaf: int
    root: Node | None = None

    def __init__(
        self,
        features,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 5,
    ) -> None:
        """
        Args:
            features (ArrayLike[StringLike]): Used exclusively for pretty printing
            max_depth (Optional[int], optional): Maximum tree height. Defaults to None, which means no limit.
            min_samples_split (int, optional): Minimum amount of data to split. Defaults to 2.
            min_samples_leaf (int, optional): Minimum amount of data to stop computing and create a leaf. Defaults to 5.
        """
        self.features = features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y) -> None:
        """
        Fit the model to the data.
        Args:
            X (ArrayLike): Features of data samples
            y (ArrayLike): Target labels of the data
        """

        def build_tree(data, depth: int) -> Node:
            """
            Build the decision tree recursively.
            Args:
                data (ArrayLike): Data samples
                depth (int): Current depth of the tree
            Returns:
                Node: Root node of the tree
            """
            if (
                depth == self.max_depth
                or len(data) <= self.min_samples_split
                or len(np.unique(data[:, -1])) == 1  # purity check
            ):
                return Node(
                    Counter(data[:, -1]).most_common()[0][0]
                )  # return the most common class

            feat_idx, threshold = self.test(data)
            left_data, right_data = DecisionTree.split_data(feat_idx, threshold, data)

            if (
                len(left_data) <= self.min_samples_leaf
                or len(right_data) <= self.min_samples_leaf
            ):
                return Node(Counter(data[:, -1]).most_common()[0][0])

            return Node(
                (feat_idx, threshold),
                build_tree(left_data, depth + 1),
                build_tree(right_data, depth + 1),
            )

        self.root = build_tree(np.hstack([X, y]), 0)

    def predict(self, X) -> np.ndarray:
        """
        Return the predicted classes of the input samples.
        Args:
            X (ArrayLike): Input samples
        Returns:
            ArrayLike: Predicted classes of the input samples
        """
        if not self.root:
            raise RuntimeError("Tree is not properly fitted")

        return np.array([self.root.predict(x) for x in X]).reshape(-1, 1)

    def test(self, data) -> tuple[int, Any]:
        """
        Determine the best data split.
        Args:
            data (ArrayLike): Data samples
        Returns:
            Tuple[int, Any]: Index of the feature and the threshold value
        """

        test_data = []
        for feat_idx in range(len(self.features) - 1):
            if isinstance(data[0, feat_idx], str):  # categorical feature
                # create permutations of the unique values (max 50% of the unique values)
                unique_categories = np.unique(data[:, feat_idx])
                subsets = [
                    list(set(c))
                    for i in range(1, len(unique_categories) // 2 + 1)
                    for c in combinations(unique_categories, i)
                ]
                test_data.append((feat_idx, subsets))
            else:  # continuous feature
                # get every unique value
                test_data.append((feat_idx, np.unique(data[:, feat_idx])))

        # calculating the information gain for each test
        tests: list[tuple[int, Any]] = []
        info_gains = []
        for feat_idx, thresholds in test_data:
            best_info_gain = 0
            best_threshold = None
            # pick best threshold for each feature
            for threshold in thresholds:
                info_gain = self.info_gain(feat_idx, threshold, data)
                if not best_info_gain or info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_threshold = threshold
            tests.append((feat_idx, best_threshold))
            info_gains.append(best_info_gain)

        # pick best feature
        chosen_test_idx = self.choose_test(info_gains)
        if chosen_test_idx is None:
            raise ValueError("No valid test found")
        return tests[chosen_test_idx]

    @abstractmethod
    def choose_test(self, info_gains) -> np.int64:
        """
        Choose the best test based on the information gains.
        Args:
            info_gains (ArrayLike): Information gains of the tests
        Returns:
            int: the indef of the chosen test
        """

    @staticmethod
    def info_gain(feat_idx, threshold, data) -> int:
        """
        Calculate the information gain of the split.
        Args:
            feat_idx (int): Index of the feature
            threshold (Any): Split threshold
            data (ArrayLike): Data samples
        Returns:
            float: Information gain of the split
        """
        left, right = DecisionTree.split_data(feat_idx, threshold, data)
        left_weight = left.size / data.size
        right_weight = right.size / data.size
        info_gain = (
            DecisionTree.entropy(data)
            - left_weight * DecisionTree.entropy(left)
            - right_weight * DecisionTree.entropy(right)
        )
        return max(info_gain, 0)

    @staticmethod
    def entropy(data) -> float:
        """
        Calculate the entropy of the data.
        Args:
            data (ArrayLike): Data samples
        Returns:
            float: Entropy of the data
        """
        entropy = 0
        for _, count in Counter(data[:, -1]).most_common():
            proportion = count / data.shape[0]
            entropy -= proportion * np.log(proportion)
        return entropy

    @staticmethod
    def split_data(feat_idx, threshold, data) -> tuple[np.ndarray, np.ndarray]:
        """
        Split the data based on the feature and threshold.
        Args:
            feat_idx (int): Index of the feature
            threshold (Any): Split threshold
            data (ArrayLike): Data samples
        Returns:
            Tuple[ArrayLike, ArrayLike]: Left and right split of the data
        """
        if isinstance(threshold, list):  # categorical feature
            left_mask = np.isin(data[:, feat_idx], threshold)
        else:  # continuous feature
            left_mask = data[:, feat_idx] <= threshold
        return data[left_mask], data[~left_mask]

    def __str__(self) -> str:
        def inorder(output: io.StringIO, node: Node, depth: int):
            indent = (depth * 4) * " "
            if not isinstance(node.value, tuple):
                output.write(f"{node.value}\n")
            else:
                feat_idx, threshold = node.value
                output.write(f"({self.features[feat_idx]}, {threshold})\n")

            if node.left:
                output.write(f"{indent}L: ")
                inorder(output, node.left, depth + 1)
            if node.right:
                output.write(f"{indent}R: ")
                inorder(output, node.right, depth + 1)

        output = io.StringIO()
        output.write("DecisionTree{")
        if self.root:
            output.write("\n")
            inorder(output, self.root, 1)

        output.write("}")
        result = output.getvalue()
        output.close()
        return result
