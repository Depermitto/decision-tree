from collections import Counter
from dataclasses import dataclass
from typing import Optional, Self, Any
import itertools
import numpy as np
import io


@dataclass
class Node:
    value: Any
    left: Optional[Self] = None
    right: Optional[Self] = None

    def predict(self, x):
        if not isinstance(self.value, InternalNode):
            return self.value

        feat_idx, threshold = self.value
        iscatin = isinstance(threshold, tuple) and x[feat_idx] in threshold
        ismore = isinstance(threshold, np.number) and x[feat_idx] > threshold
        if iscatin or ismore:
            return self.right.predict(x)
        else:
            return self.left.predict(x)


@dataclass
class InternalNode:
    feature_idx: int
    threshold: Any

    def __iter__(self):
        return iter((self.feature_idx, self.threshold))


class DTClassifier:
    """Classifier for continuous and categorical variables using decision trees"""

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

    def fit(self, X, y):
        def build_tree(features, data, depth: int) -> Node:
            if (
                depth == self.max_depth
                or len(data) <= self.min_samples
                or len(np.unique(data[:, -1])) == 1  # purity check
            ):
                return Node(Counter(data[:, -1]).most_common()[0][0])

            feat_idx, threshold = DTClassifier.test(features, data)
            left_data, right_data = DTClassifier.split_data(feat_idx, threshold, data)
            if len(left_data) == 0 or len(right_data) == 0:
                return Node(Counter(data[:, -1]).most_common()[0][0])

            return Node(
                InternalNode(feat_idx, threshold),
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
        assert len(data) != 0 and len(data[0]) != 0
        # Test k thresholds for every attribute except the classes
        k = 20
        test_data = []
        for f_idx in range(len(features) - 1):
            first = data[0, f_idx]
            if isinstance(first, np.number):
                f_col = data[:, f_idx]
                test_data += [
                    (f_idx, t) for t in np.linspace(f_col.min(), f_col.max(), k)
                ]
            elif isinstance(first, str):
                uniq_categories = np.unique(data[:, f_idx])
                test_data += [
                    (f_idx, combinations)
                    for r in range(1, len(uniq_categories) + 1)
                    for combinations in list(itertools.combinations(uniq_categories, r))
                ]

        # Calculate all information gains
        info_gains = np.trim_zeros(
            [DTClassifier.info_gain(f_idx, t, data) for f_idx, t in test_data]
        )

        # Roulette
        total = np.sum(info_gains)
        probs = info_gains / total
        i = np.random.choice(range(len(info_gains)), p=probs)

        return test_data[i]

    @staticmethod
    def info_gain(feat_idx, threshold, data):
        left, right = DTClassifier.split_data(feat_idx, threshold, data)
        left_weight = left.size / data.size
        right_weight = right.size / data.size
        info_gain = (
            DTClassifier.entropy(data)
            - left_weight * DTClassifier.entropy(left)
            - right_weight * DTClassifier.entropy(right)
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
        if isinstance(threshold, np.number):
            left_mask = data[:, feat_idx] <= threshold
        elif isinstance(threshold, tuple):  # (cat1, cat2, cat3,..., catn)
            left_mask = np.isin(data[:, feat_idx], threshold)

        return data[left_mask], data[~left_mask]

    def __str__(self) -> str:
        def inorder(output: io.StringIO, node: Node, depth: int):
            indent = (depth * 4) * " "
            if not isinstance(node.value, InternalNode):
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
