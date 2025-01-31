import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from tree import RouletteTree, ClassicTree


def main():
    datasets = [
        "data/winequality-red.csv",
        "data/nursery.csv",
        "data/loan_data.csv",
        "data/user_behavior_dataset.csv",
    ]
    data = pd.read_csv(datasets[1])

    # Preprocess data
    features = data.columns.values
    X, y = data.values[:, :-1], data.values[:, -1].reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )

    # Our classifier
    dt = RouletteTree(features, len(features))
    dt.fit(X_train, y_train)
    print(dt)

    y_pred = dt.predict(X_test)
    acc = np.sum(y_pred == y_test) / len(y_pred)
    print(f"Roulette model accuracy: {acc}")

    # Compare with classic tree
    dt = ClassicTree(features, len(features))
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    acc = np.sum(y_pred == y_test) / len(y_pred)
    print(f"Classic model accuracy: {acc}")


if __name__ == "__main__":
    main()
