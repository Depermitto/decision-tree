#pragma once
#include <algorithm>
#include <iterator>
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
template <typename Class>
class DecisionTree {
   public:
    DecisionTree(const rapidcsv::Document &data, int max_depth = -1) {
        m_data = data;

        m_attributes = data.GetColumnNames();
        const std::string class_column_label = m_attributes.back();
        m_attributes.pop_back();

        std::vector<Class> classes = data.GetColumn<Class>(class_column_label);
        std::sort(classes.begin(), classes.end());
        std::unique_copy(classes.begin(), classes.end(), std::back_inserter(m_classes));

        if (max_depth <= 0 || max_depth > m_attributes.size()) {
            max_depth = m_attributes.size();
        }
        m_root = build_tree(m_attributes, m_data, m_classes, max_depth);
        if (not m_root) throw std::runtime_error("could not fit the tree!");
    }

    Class predict() const;

    std::vector<Class> classes() const { return m_classes; }

    friend std::ostream &operator<<(std::ostream &os, const DecisionTree<Class> &decision_tree) {
        os << "DecisionTree{";
        if (decision_tree.m_root) {
            os << "\n";
            stream(os, decision_tree.m_root);
        }
        os << "}";
        return os;
    }

   private:
    struct Node {
        std::variant<Class, std::tuple<std::string, double>> m_value;
        std::unique_ptr<Node> m_left;
        std::unique_ptr<Node> m_right;

        Node(const std::variant<Class, std::tuple<std::string, double>> &value,
             std::unique_ptr<Node> left = nullptr, std::unique_ptr<Node> right = nullptr)
            : m_value(value), m_left(std::move(left)), m_right(std::move(right)) {}
    };
    std::unique_ptr<Node> m_root;
    rapidcsv::Document m_data;
    std::vector<std::string> m_attributes;
    std::vector<Class> m_classes;

    static std::unique_ptr<Node> build_tree(std::vector<std::string> &attributes,
                                            const rapidcsv::Document &data, const std::vector<Class> &classes,
                                            uint max_depth) {
        if (attributes.size() == 0 || max_depth == 0) {
            std::map<Class, uint> common;
            for (uint row_idx = 0; row_idx < data.GetRowCount(); row_idx++) {
                common[data.GetCell<Class>(data.GetColumnCount() - 1, row_idx)] += 1;
            }

            return std::make_unique<Node>(
                std::max_element(common.begin(), common.end(), [](const auto &p1, const auto &p2) {
                    return p1.second < p2.second;
                })->first);
        }

        const auto [attribute, threshold] = test(attributes, data, classes);
        const auto [left_data, right_data] = split_data(attribute, data, threshold);
        attributes.erase(std::find(attributes.begin(), attributes.end(), attribute));

        return std::make_unique<Node>(std::make_tuple(attribute, threshold),
                                      build_tree(attributes, left_data, classes, max_depth - 1),
                                      build_tree(attributes, right_data, classes, max_depth - 1));
    }

    static std::tuple<std::string, double> test(const std::vector<std::string> &attributes,
                                                const rapidcsv::Document &data,
                                                const std::vector<Class> &classes) {
        // TODO
        const auto attr = attributes[0];
        const auto tr = data.GetCell<double>(data.GetColumnIdx(attr), 0);
        return std::make_tuple(attr, tr);
    }

    static std::tuple<rapidcsv::Document, rapidcsv::Document> split_data(const std::string &attribute,
                                                                         const rapidcsv::Document &data,
                                                                         double threshold) {
        rapidcsv::Document left = data;
        rapidcsv::Document right = data;
        uint attribute_column_idx = data.GetColumnIdx(attribute);
        for (uint row_idx = 0; row_idx < data.GetRowCount(); row_idx++) {
            const auto val = data.GetCell<double>(attribute_column_idx, row_idx);
            if (val < threshold) {
                right.RemoveRow(row_idx);
            } else {
                left.RemoveRow(row_idx);
            }
        }

        return std::make_tuple(left, right);
    }

    // static double ig(const std::string &attribute, const rapidcsv::Document &data,
    //                  std::optional<double> threshold) {
    // }

    static void stream(std::ostream &os, const std::unique_ptr<Node> &node, uint depth = 1) {
        if (std::holds_alternative<Class>(node->m_value)) {
            os << "(" << std::get<Class>(node->m_value);
        } else {
            const auto [a, t] = std::get<std::tuple<std::string, double>>(node->m_value);
            os << "(" << a << ", " << t;
        }

        os << ")\n";
        if (node->m_left) {
            os << std::setw(depth * 4 + 3) << "L: ";
            stream(os, node->m_left, depth + 1);
        }
        if (node->m_right) {
            os << std::setw(depth * 4 + 3) << "R: ";
            stream(os, node->m_right, depth + 1);
        }
    }
};
}  // namespace uma
