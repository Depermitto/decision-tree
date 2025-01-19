from dataclasses import dataclass
from typing import Optional, Self, Any

@dataclass
class Node:
    """
    Node class for Decision Tree
    Fields:
        value (Any): Value of the node. If it is a tuple, it represents a split condition.
        left (Optional[Self]): Left child node
        right (Optional[Self]): Right child node
    """
    value: Any
    left: Optional[Self] = None
    right: Optional[Self] = None

    def predict(self, x):
        """
        Traverse the tree to predict the class of the input sample.
        """
        if not isinstance(self.value, tuple):
            return self.value

        feat_idx, threshold = self.value
        # Categorical feature
        if isinstance(x[feat_idx], str):
            if not self.right or x[feat_idx] in threshold:
                return self.left.predict(x)
            else:
                return self.right.predict(x)
        # Continuous feature
        if not self.right or x[feat_idx] < threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)