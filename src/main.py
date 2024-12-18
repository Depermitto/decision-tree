import pandas as pd

from decision_tree import DecisionTree


def main():
    data = pd.read_csv("data/winequality-red.csv")
    dt = DecisionTree(data, 4)

    print(dt)


if __name__ == "__main__":
    main()
