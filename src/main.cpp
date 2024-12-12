#include <decision_tree.hpp>

#include "rapidcsv.h"

int main() {
    rapidcsv::Document data("data/winequality-red.csv");
    uma::DecisionTree<int> tree(data, 3);

    // for (const auto& a : attributes) {
    //     std::cout << a << "\n";
    // }
    // std::cout << label;

    std::cout << tree << std::endl;
}
