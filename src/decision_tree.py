from collections import Counter
from dataclasses import dataclass
from typing import Optional, Self, Any
import pandas as pd
import numpy as np
import io


class DecisionTree:
    def __init__(self, features, X, y, max_depth: int) -> None:
        data = np.hstack([X, y])
        self.features = features
        self.root = DecisionTree.build_tree(features.copy(), data, max_depth)

    @dataclass
    class Node:
        value: Any
        left: Optional[Self] = None
        right: Optional[Self] = None

    @staticmethod
    def build_tree(features, data, max_depth) -> Node:
        if max_depth == 0 or len(features) == 0 or len(data) <= 10:  # arbitrary limit
            return DecisionTree.Node(Counter(data[:, -1]).most_common()[0][0])

        feat_idx, threshold = DecisionTree.test(features, data)
        left_data, right_data = DecisionTree.split_data(feat_idx, threshold, data)

        if len(left_data) == 0 or len(right_data) == 0:
            return DecisionTree.Node(Counter(data[:, -1]).most_common()[0][0])

        np.delete(features, feat_idx)

        return DecisionTree.Node(
            (feat_idx, threshold),
            DecisionTree.build_tree(features, left_data, max_depth - 1),
            DecisionTree.build_tree(features, right_data, max_depth - 1),
        )

    @staticmethod
    def test(features, data):
        # test k thresholds for every attribute except the classes
        k = 50
        test_data = [
            (feat_idx, t)
            for feat_idx in range(len(features) - 1)
            for t in np.linspace(data[feat_idx].min(), data[feat_idx].max(), k)
        ]
        info_gains = [
            DecisionTree.info_gain(feat_idx, t, data) for feat_idx, t in test_data
        ]

        # roulette
        total = np.sum(info_gains)
        probs = info_gains / total
        i = np.random.choice(range(len(info_gains)), p=probs)

        return test_data[i]

    @staticmethod
    def info_gain(feat_idx, threshold, data):
        left, right = DecisionTree.split_data(feat_idx, threshold, data)
        left_weight = left.size / data.size
        right_weight = right.size / data.size
        return (
            DecisionTree.entropy(data)
            - left_weight * DecisionTree.entropy(left)
            - right_weight * DecisionTree.entropy(right)
        )

    @staticmethod
    def entropy(data):
        entropy = 0
        for _, count in Counter(data[:, -1]).most_common():
            proportion = count / data.shape[0]
            entropy -= proportion * np.log(proportion)
        return entropy

    @staticmethod
    def split_data(feat_idx, threshold, data):
        left_mask = data[:, feat_idx] <= threshold

        return data[left_mask], data[~left_mask]

    def __str__(self) -> str:
        def inorder(output, node, depth):
            indent = (depth * 4) * " "
            if isinstance(node.value, np.float64):
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
