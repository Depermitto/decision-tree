#include <decision_tree.hpp>
#include <tuple>

#include "rapidcsv.h"

template <typename T, typename I>
std::tuple<std::vector<T>, std::vector<T>> train_test_split(I start, I end, float ratio) {
    // TODO
}

int main() {
    rapidcsv::Document doc("data/winequality-red.csv");

    const auto attributes = doc.GetColumnNames();
    std::vector<std::vector<double>> data(doc.GetRowCount());
    for (size_t i = 0; i < doc.GetRowCount(); i++) {
        data[i] = doc.GetRow<double>(i);
    }

    const auto [train, test] = train_test_split<std::vector<double>>(data.cbegin(), data.cend(), 0.8);
    uma::DecisionTree tree(attributes, train, 4);

    // for (const auto& a : attributes) {
    //     std::cout << a << "\n";
    // }
    // std::cout << label;

    std::cout << tree << std::endl;
}
