#pragma once
#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "rapidcsv.h"

namespace uma {
template <typename ClassType>
class DecisionTree {
   public:
    DecisionTree(const rapidcsv::Document &data, int max_depth = -1) {
        m_data = data;

        m_attributes = data.GetColumnNames();
        const std::string class_column_label = m_attributes.back();
        m_attributes.pop_back();

        std::vector<ClassType> classes = data.GetColumn<ClassType>(class_column_label);
        std::sort(classes.begin(), classes.end());
        std::unique_copy(classes.begin(), classes.end(), std::back_inserter(m_classes));

        if (max_depth <= 0 || max_depth > m_attributes.size()) {
            max_depth = m_attributes.size();
        }
        m_root = build_tree(m_attributes, m_data, m_classes, max_depth);
        if (not m_root) throw std::runtime_error("could not fit the tree!");
    }

    ClassType predict() const;

    std::vector<ClassType> classes() const { return m_classes; }

    friend std::ostream &operator<<(std::ostream &os, const DecisionTree<ClassType> &decision_tree) {
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
        std::string m_attribute;
        std::optional<double> m_threshold;
        std::unique_ptr<Node> m_left;
        std::unique_ptr<Node> m_right;

        Node(const std::string &attribute, std::optional<double> threshold = std::nullopt,
             std::unique_ptr<Node> left = nullptr, std::unique_ptr<Node> right = nullptr)
            : m_attribute(attribute),
              m_threshold(threshold),
              m_left(std::move(left)),
              m_right(std::move(right)) {}
    };
    std::unique_ptr<Node> m_root;
    rapidcsv::Document m_data;
    std::vector<std::string> m_attributes;
    std::vector<ClassType> m_classes;

    static std::unique_ptr<Node> build_tree(std::vector<std::string> &attributes, rapidcsv::Document &data,
                                            const std::vector<ClassType> &classes, uint max_depth) {
        if (attributes.size() == 0 || max_depth == 0) {
            return std::make_unique<Node>("leaf");  // TODO: most common class from data
        }

        const auto [attribute, threshold] = test(attributes, data, classes);
        // const auto [left_data, right_data] = TODO;
        attributes.erase(std::find(attributes.begin(), attributes.end(), attribute));

        return std::make_unique<Node>(attribute, threshold,
                                      build_tree(attributes, data, classes, max_depth - 1),
                                      build_tree(attributes, data, classes, max_depth - 1));
    }

    static std::tuple<std::string, double> test(const std::vector<std::string> &attributes,
                                                const rapidcsv::Document &data,
                                                const std::vector<ClassType> &classes) {
        const auto attr = attributes[0];
        const auto tr = data.GetCell<double>(data.GetColumnIdx(attr), 0);
        return std::make_tuple(attr, tr);
    }

    static double ig(std::string_view attribute, const rapidcsv::Document &data,
                     std::optional<double> threshold);

    static void stream(std::ostream &os, const std::unique_ptr<Node> &node, uint depth = 1) {
        os << "(" << node->m_attribute;
        if (node->m_threshold) {
            os << ", " << *node->m_threshold;
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
