from dataclasses import dataclass
from typing import Optional, Self, Any

@dataclass
class Node:
    value: Any
    left: Optional[Self] = None
    right: Optional[Self] = None

    def predict(self, x):
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