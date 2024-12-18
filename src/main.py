import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.tree

from tree import DecisionTreeClassifier


def main():
    wine = pd.read_csv("data/winequality-red.csv")

    features = wine.columns.values
    X, y = wine.values[:, :-1], wine.values[:, -1].reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )

    # Our classifier
    dt = DecisionTreeClassifier(features)
    dt.fit(X_train, y_train)
    print(dt)

    y_pred = dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.2f}")

    # Sklearn gotowiec
    sklearn_dt = sklearn.tree.DecisionTreeClassifier(
        criterion="entropy", splitter="random"
    )
    sklearn_dt.fit(X_train, y_train)
    print(sklearn_dt)

    y_pred = sklearn_dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Sklearn model accuracy: {acc:.2f}")


if __name__ == "__main__":
    main()
