import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.tree

from tree import DTClassifier


def main():
    datasets = [
        "data/winequality-red.csv",
        "data/nursery.csv",
        "data/loan_data.csv",
        "data/user_behavior_dataset.csv",
    ]
    data = pd.read_csv(datasets[3])

    # Preprocess data
    features = data.columns.values
    X, y = data.values[:, :-1], data.values[:, -1].reshape(-1, 1)

    # Encode categorical variables, has no effect on continuos variables.
    for f_idx in range(X.shape[1]):
        if isinstance(X[0, f_idx], str):
            uniq_categories = np.unique(X[:, f_idx])
            cat_map = {cat: ordinal for ordinal, cat in enumerate(uniq_categories)}
            X[:, f_idx] = [cat_map[x] for x in X[:, f_idx]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )

    # Our classifier
    dt = DTClassifier(features, 20)
    dt.fit(X_train, y_train)
    print(dt)

    y_pred = dt.predict(X_test)
    acc = np.sum(y_pred == y_test) / len(y_pred)
    print(f"Model accuracy: {acc:.2f}")

    # Sklearn gotowiec
    dt_sklearn = sklearn.tree.DecisionTreeClassifier(
        criterion="entropy", splitter="random"
    )
    dt_sklearn.fit(X_train, y_train)
    print(dt_sklearn)

    y_pred = dt_sklearn.predict(X_test).reshape(-1, 1)
    acc = np.sum(y_pred == y_test) / len(y_pred)
    print(f"Sklearn model accuracy: {acc:.2f}")


if __name__ == "__main__":
    main()
