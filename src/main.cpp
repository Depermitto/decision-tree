#include <algorithm>
#include <decision_tree.hpp>
#include <iterator>
#include <tuple>

#include "randshow/engines.hpp"
#include "rapidcsv.h"

template <typename T, typename It>
std::tuple<std::vector<T>, std::vector<T>> train_test_split(It start, It end, float ratio) {
    It split_it = start + std::distance(start, end) * ratio;
    std::vector<T> train_data(start, split_it), test_data(split_it, end);

    randshow::DefaultEngine.Shuffle(train_data.begin(), train_data.end());
    randshow::DefaultEngine.Shuffle(test_data.begin(), test_data.end());

    return std::make_tuple(train_data, test_data);
}

int main() {
    rapidcsv::Document doc("data/winequality-red.csv");

    // Read and prepare data
    const auto attributes = doc.GetColumnNames();
    std::vector<std::vector<double>> data(doc.GetRowCount());
    for (size_t i = 0; i < doc.GetRowCount(); i++) {
        data[i] = doc.GetRow<double>(i);
    }

    // Train
    const auto [train, test] = train_test_split<std::vector<double>>(data.cbegin(), data.cend(), 0.8);
    uma::DecisionTree dt(train, attributes, 1);
    std::cout << dt << '\n';

    // Test
    size_t correct = 0;
    std::for_each(test.cbegin(), test.cend(), [&dt, &correct](const auto& test_data) {
        correct += test_data[test_data.size() - 1] == dt.predict(test_data);
    });
    std::cout << "accuracy: " << (double)correct / test.size() << '\n';
}
