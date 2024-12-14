#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "rapidcsv.h"

namespace uma {
class DecisionTree {
    using Class = double;
    using Cell = double;
    using Vector2D = std::vector<std::vector<Cell>>;

    struct Node {
        std::variant<Class, std::tuple<size_t, Cell>> m_value;
        std::unique_ptr<Node> m_left;
        std::unique_ptr<Node> m_right;

        Node(const std::variant<Class, std::tuple<size_t, Cell>> &value, std::unique_ptr<Node> left = nullptr,
             std::unique_ptr<Node> right = nullptr)
            : m_value(value), m_left(std::move(left)), m_right(std::move(right)) {}
    };

    std::unique_ptr<DecisionTree::Node> build_tree(std::vector<std::string> &attributes, const Vector2D &data,
                                                   size_t max_depth) {
        if (attributes.size() == 0 || max_depth == 0) {
            std::map<Class, size_t> common;
            size_t col_idx = data[0].size() - 1;
            for (size_t row_idx = 0; row_idx < data.size(); row_idx++) {
                common[data[row_idx][col_idx]] += 1;
            }

            // TODO: should be roulette
            return std::make_unique<Node>(
                std::max_element(common.begin(), common.end(), [](const auto &p1, const auto &p2) {
                    return p1.second < p2.second;
                })->first);
        }

        const auto [attr_idx, threshold] = test(attributes, data);
        const auto [left_data, right_data] = split_data(attr_idx, threshold, data);
        attributes.erase(attributes.begin() + attr_idx);

        return std::make_unique<Node>(std::make_tuple(attr_idx, threshold),
                                      build_tree(attributes, left_data, max_depth - 1),
                                      build_tree(attributes, right_data, max_depth - 1));
    }

    std::tuple<size_t, Cell> test(const std::vector<std::string> &attributes, const Vector2D &data) {
        size_t attr_idx;
        Cell threshold;
        double max_gain;
        for (size_t a_idx = 0; a_idx < attributes.size(); a_idx++) {
            for (size_t row_idx = 0; row_idx < data.size(); row_idx += 10) {
                Cell t = data[row_idx][a_idx];
                double gain = information_gain(a_idx, t, data);
                if (gain > max_gain) {
                    max_gain = gain;
                    attr_idx = a_idx;
                    threshold = t;
                }
            }
        }
        return std::make_tuple(attr_idx, threshold);
    }

    double information_gain(size_t attr_idx, Cell threshold, const Vector2D &data) {
        const auto [left_data, right_data] = split_data(attr_idx, threshold, data);
        double left_weight = static_cast<double>(left_data.size()) / data.size();
        double right_weight = static_cast<double>(right_data.size()) / data.size();
        return entropy(data) - left_weight * entropy(left_data) - right_weight * entropy(right_data);
    }

    double entropy(const Vector2D &data) {
        if (data.size() == 0) {
            return 0.0;
        }

        size_t total = data.size();
        size_t col_idx = data[0].size() - 1;
        std::map<Class, size_t> common;
        for (size_t row_idx = 0; row_idx < total; row_idx++) {
            common[data[row_idx][col_idx]] += 1;
        }

        double entropy = 0;
        for (const auto [_, count] : common) {
            double proportion = static_cast<double>(count) / total;
            entropy -= proportion * std::log(proportion);
        }
        return entropy;
    }

    std::tuple<Vector2D, Vector2D> split_data(size_t attr_idx, Cell threshold, const Vector2D &data) {
        Vector2D left;
        Vector2D right;
        for (size_t row_idx = 0; row_idx < data.size(); row_idx++) {
            if (data[row_idx][attr_idx] < threshold) {
                left.push_back(data[row_idx]);
            } else {
                right.push_back(data[row_idx]);
            }
        }

        return std::make_tuple(left, right);
    }

   public:
    DecisionTree(const rapidcsv::Document &doc, size_t max_depth = 0) {
        m_attributes = doc.GetColumnNames();
        m_attributes.pop_back();
        std::vector<std::string> attributes(m_attributes.begin(), m_attributes.end());

        Vector2D data(doc.GetRowCount());
        for (size_t i = 0; i < doc.GetRowCount(); i++) {
            data[i] = doc.GetRow<Cell>(i);
        }

        if (max_depth == 0 || max_depth > attributes.size()) {
            max_depth = attributes.size();
        }
        m_root = build_tree(attributes, data, max_depth);
        if (not m_root) throw std::runtime_error("could not fit the tree!");
    }

    Class predict() const;

    friend std::ostream &operator<<(std::ostream &os, const DecisionTree &decision_tree) {
        os << "DecisionTree{";
        if (decision_tree.m_root) {
            std::function<void(const Node &, uint)> inorder = [&decision_tree, &os, &inorder](
                                                                  const Node &node, uint depth) {
                if (std::holds_alternative<Class>(node.m_value)) {
                    os << "(" << std::get<Class>(node.m_value);
                } else {
                    const auto [a, t] = std::get<std::tuple<size_t, Cell>>(node.m_value);
                    os << "(" << decision_tree.m_attributes[a] << ", " << t;
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
            os << "\n";
            inorder(*decision_tree.m_root, 1);
        }
        os << "}";
        return os;
    }

   private:
    std::unique_ptr<Node> m_root;
    std::vector<std::string> m_attributes;
};
}  // namespace uma
