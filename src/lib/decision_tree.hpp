#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#include "randshow/engines.hpp"

namespace uma {
constexpr size_t CHUNK = 5;  // NOTE: maybe add this as parameter or template parameter?
class DecisionTree {
    using Class = double;                             // TODO change to std::tuple<double, std::string>
    using Cell = double;                              // TODO change to template parameter
    using Vector2D = std::vector<std::vector<Cell>>;  // NOTE: possible optimization in introducing a type
                                                      // with heterogeneous memory layout

    struct Node {
        std::variant<Class, std::tuple<std::string, Cell>> m_value;
        std::unique_ptr<Node> m_left;
        std::unique_ptr<Node> m_right;

        Node(const std::variant<Class, std::tuple<std::string, Cell>> &value,
             std::unique_ptr<Node> left = nullptr, std::unique_ptr<Node> right = nullptr)
            : m_value(value), m_left(std::move(left)), m_right(std::move(right)) {}
    };

   public:
    DecisionTree(const std::vector<std::string> &labels, const Vector2D &data, size_t max_depth = 0) {
        m_labels = labels;
        std::vector<std::string> attributes(m_labels.begin(), m_labels.end() - 1);

        if (max_depth == 0 || max_depth > attributes.size()) {
            max_depth = std::log2(attributes.size());
        }
        m_root = build_tree(attributes, data, max_depth);
        if (not m_root) throw std::runtime_error("could not fit the tree!");
    }

    Class predict() const {
        // TODO
    }

    friend std::ostream &operator<<(std::ostream &os, const DecisionTree &dt) {
        os << "DecisionTree{";
        if (dt.m_root) {
            std::function<void(const Node &, size_t)> inorder = [&os, &inorder](const Node &node,
                                                                                size_t depth) {
                if (std::holds_alternative<Class>(node.m_value)) {
                    os << '(' << std::get<Class>(node.m_value);
                } else {
                    const auto [a, t] = std::get<std::tuple<std::string, Cell>>(node.m_value);
                    os << '(' << a << ", " << t;
                }
                os << ")\n";

                if (node.m_left) {
                    os << std::setw(depth * 4 + 3) << "L: ";
                    inorder(*node.m_left, depth + 1);
                }
                if (node.m_right) {
                    os << std::setw(depth * 4 + 3) << "R: ";
                    inorder(*node.m_right, depth + 1);
                }
            };
            os << '\n';
            inorder(*dt.m_root, 1);
        }
        os << '}';
        return os;
    }

   private:
    std::unique_ptr<Node> m_root;
    std::vector<std::string> m_labels;

    std::unique_ptr<Node> build_tree(std::vector<std::string> &attributes, const Vector2D &data,
                                     size_t max_depth) const {
        if (data.size() == 0) return nullptr;
        if (attributes.size() == 0 || max_depth == 0) {
            std::unordered_map<Class, size_t> common;
            size_t class_idx = m_labels.size() - 1;
            for (size_t row_idx = 0; row_idx < data.size(); row_idx++) {
                common[data[row_idx][class_idx]]++;
            }

            // TODO: should be roulette
            return std::make_unique<Node>(
                std::max_element(common.begin(), common.end(), [](const auto &p1, const auto &p2) {
                    return p1.second < p2.second;
                })->first);
        }

        const auto [attr_idx, threshold] = test(attributes, data);
        const auto [left_data, right_data] = split_data(attr_idx, threshold, data);
        // std::cout << left_data.size() << ' ' << right_data.size() << '\n';
        std::string attr = attributes[attr_idx];
        attributes.erase(attributes.begin() + attr_idx);

        return std::make_unique<Node>(std::make_tuple(std::move(attr), threshold),
                                      build_tree(attributes, left_data, max_depth - 1),
                                      build_tree(attributes, right_data, max_depth - 1));
    }

    std::tuple<size_t, Cell> test(const std::vector<std::string> &attributes, const Vector2D &data) const {
        size_t attr_idx;
        Cell threshold;
        double max_gain = 0;
        for (size_t a_idx = 0; a_idx < attributes.size(); a_idx++) {
            for (size_t row_idx = randshow::DefaultEngine.Next(CHUNK); row_idx < data.size();
                 row_idx += CHUNK) {
                Cell t = data[row_idx][a_idx];
                double gain = information_gain(a_idx, t, data);
                if (gain >= max_gain) {
                    max_gain = gain;
                    attr_idx = a_idx;
                    threshold = t;
                }
            }
        }
        return std::make_tuple(attr_idx, threshold);
    }

    double information_gain(size_t attr_idx, Cell threshold, const Vector2D &data) const {
        size_t left_data_amount = 0, right_data_amount = 0;
        std::unordered_map<Class, size_t> left_class_counts, right_class_counts, class_counts;
        size_t class_idx = m_labels.size() - 1;
        for (size_t row_idx = 0; row_idx < data.size(); row_idx++) {
            if (data[row_idx][attr_idx] < threshold) {
                left_class_counts[data[row_idx][class_idx]]++;
                left_data_amount++;
            } else {
                right_class_counts[data[row_idx][class_idx]]++;
                right_data_amount++;
            }
            class_counts[data[row_idx][class_idx]]++;
        }

        double left_weight = (double)left_data_amount / data.size();
        double right_weight = (double)right_data_amount / data.size();
        return entropy(class_counts, data.size()) -
               left_weight * entropy(left_class_counts, left_data_amount) -
               right_weight * entropy(right_class_counts, right_data_amount);
    }

    template <typename Map>
    double entropy(const Map &class_counts, size_t data_amount) const {
        double entropy = 0;
        for (auto [_, count] : class_counts) {
            double proportion = (double)count / data_amount;
            entropy -= proportion * std::log(proportion);
        }
        return entropy;
    }

    std::tuple<Vector2D, Vector2D> split_data(size_t attr_idx, Cell threshold, const Vector2D &data) const {
        Vector2D left, right;
        for (size_t row_idx = 0; row_idx < data.size(); row_idx++) {
            if (data[row_idx][attr_idx] < threshold) {
                left.push_back(data[row_idx]);
            } else {
                right.push_back(data[row_idx]);
            }
        }
        return std::make_tuple(left, right);
    }
};
}  // namespace uma
