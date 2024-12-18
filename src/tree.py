from collections import Counter
from dataclasses import dataclass
from typing import Optional, Self, Any
import numpy as np
import io


class DecisionTreeClassifier:
    """DecisionTreeClassifier for continuous and categorical variables"""

    def __init__(
        self,
        features,
        max_depth: Optional[int] = None,
        min_samples: int = 5,
    ) -> None:
        """
        Args:
              features (ArrayLike[StringLike]): Used exclusively for pretty printing
              max_depth (Optional[int], optional): Maximum tree height. Defaults to None, which means no limit.
              min_samples (int, optional): Minimum amount of data to stop computing and create a leaf. Defaults to 5.
        """
        self.features = features
        self.max_depth = max_depth
        self.min_samples = min_samples

    @dataclass
    class Node:
        value: Any
        left: Optional[Self] = None
        right: Optional[Self] = None

        def predict(self, x):
            if isinstance(self.value, np.number):
                return self.value

            feat_idx, threshold = self.value
            if not self.right or x[feat_idx] < threshold:
                return self.left.predict(x)
            else:
                return self.right.predict(x)

    def fit(self, X, y):
        def build_tree(features, data, depth: int) -> DecisionTreeClassifier.Node:
            if (
                depth == self.max_depth
                or len(data) <= self.min_samples
                or len(np.unique(data[:, -1])) == 1  # purity check
            ):
                return DecisionTreeClassifier.Node(
                    Counter(data[:, -1]).most_common()[0][0]
                )

            feat_idx, threshold = DecisionTreeClassifier.test(features, data)
            left_data, right_data = DecisionTreeClassifier.split_data(
                feat_idx, threshold, data
            )

            return DecisionTreeClassifier.Node(
                (feat_idx, threshold),
                build_tree(features, left_data, depth + 1),
                build_tree(features, right_data, depth + 1),
            )

        self.root = build_tree(
            features=self.features.copy(),
            data=np.hstack([X, y]),
            depth=0,
        )

    def predict(self, X):
        if not self.root:
            raise RuntimeError("Tree is not properly fitted")

        return [self.root.predict(x) for x in X]

    @staticmethod
    def test(features, data):
        # test k thresholds for every attribute except the classes
        k = 20
        test_data = [
            (feat_idx, t)
            for feat_idx in range(len(features) - 1)
            for t in np.linspace(data[:, feat_idx].min(), data[:, feat_idx].max(), k)
        ]
        info_gains = np.trim_zeros(
            [
                DecisionTreeClassifier.info_gain(feat_idx, t, data)
                for feat_idx, t in test_data
            ]
        )

        # roulette
        total = np.sum(info_gains)
        probs = info_gains / total
        i = np.random.choice(range(len(info_gains)), p=probs)

        return test_data[i]

    @staticmethod
    def info_gain(feat_idx, threshold, data):
        left, right = DecisionTreeClassifier.split_data(feat_idx, threshold, data)
        left_weight = left.size / data.size
        right_weight = right.size / data.size
        info_gain = (
            DecisionTreeClassifier.entropy(data)
            - left_weight * DecisionTreeClassifier.entropy(left)
            - right_weight * DecisionTreeClassifier.entropy(right)
        )
        return max(info_gain, 0)

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
        def inorder(output: io.StringIO, node: DecisionTreeClassifier.Node, depth: int):
            indent = (depth * 4) * " "
            if isinstance(node.value, np.number):
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
