import random
import string
from math import log2
import pydot
import numpy as np

class Node:
    def __init__(self):
        self.children = []
        self.is_leaf = False
        self.value = ""
        self.predict = ""
        self.is_feature = True

class ID3:
    def __init__(self):
        self.root = None

    def base_entropy(self, p):
        if p == 0:
            return 0
        return -1*p*log2(p)

    def attribute_entropy(self, dataset, attribute_column_name):
        attribute_column = dataset[attribute_column_name]
        categories, count = np.unique(attribute_column, return_counts=True)
        target_attribute = dataset[self.target_class_name]
        labels, labels_count = np.unique(target_attribute, return_counts=True)
        # probabilities = count / len(dataset)
        probabilities = count / self.record_length  #todo : probabilities = count / len(dataset)
        categories_ent = np.zeros((len(categories),))
        i = 0
        for count, category in zip(count, categories):
            data = dataset[dataset[attribute_column_name] == category]
            ent = 0
            for label in labels:
                p = len(data[data[self.target_class_name] == label])
                ent += self.base_entropy(p/count)
            categories_ent[i] = ent
            i += 1
        weighted_ent = np.multiply(categories_ent, probabilities)
        attribute_entropy = np.sum(weighted_ent)
        return attribute_entropy

    def dataset_entropy(self, dataset):
        length = len(dataset)
        target_attribute = dataset[self.target_class_name]
        labels, labels_count = np.unique(target_attribute, return_counts=True)
        ent = 0
        for lbl_count in labels_count:
            ent += self.base_entropy(lbl_count/length)
        return ent

    def is_pure(self, dataset):
        return self.dataset_entropy(dataset) == 0

    def remove_best_from_attributes(self, attrs, best):
        new_attrs = attrs.copy()
        new_attrs.remove(best)
        return new_attrs

    def fit(self, examples, attributes):
        self.target_class_name = examples.columns[-1]
        self.record_length = len(examples)
        self.root = self.id3(examples, attributes)

    def id3(self, examples, attributes):
        root = Node()
        min_ent = 1
        best_feature = None
        for feature in attributes:
            ent = self.attribute_entropy(examples, feature)
            if ent < min_ent:
                min_ent = ent
                best_feature = feature
        root.value = best_feature
        uniq_values = np.unique(examples[best_feature])
        for v in uniq_values:
            sub_dataset = examples[examples[best_feature] == v]
            if self.is_pure(sub_dataset):
                leaf_node = Node()
                leaf_node.is_feature = False
                leaf_node.is_leaf = True
                leaf_node.value = v
                leaf_node.predict = np.unique(sub_dataset[self.target_class_name])[0]
                root.children.append(leaf_node)
            else:
                dummy = Node()
                dummy.value = v
                dummy.is_feature = False
                new_attributes = self.remove_best_from_attributes(attributes, best_feature)
                if len(new_attributes) == 0:
                    dummy.is_leaf = True
                    counts = np.bincount(sub_dataset[self.target_class_name])
                    common_label = np.argmax(counts)
                    dummy.predict = common_label
                else:
                    child = self.id3(sub_dataset, new_attributes)
                    dummy.children.append(child)
                root.children.append(dummy)
        return root

    def id_generator(self, size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    def classify(self, sample, root):
        if root.is_leaf == True:
            return root.predict
        if root.is_feature == True:
            child = [child for child in root.children if child.value == sample[root.value]]
            if child != []:
                return self.classify(sample, child[0])
            else:
                results = []
                for child in root.children:
                    results.append(self.classify(sample, child))
                counts = np.bincount(results)
                common_label = np.argmax(counts)
                return common_label
        else:
            return self.classify(sample, root.children[0])

    def predict(self, samples):
        predictions = []
        for _, v in samples.iterrows():
            predictions.append(self.classify(v, self.root))
        return predictions

    def export_graphviz(self, tree_name = "my_decision_tree" ,back_color="white"):
        tree = pydot.Dot(graph_name=tree_name, graph_type="graph", bgcolor=back_color)
        self.visual_tree(self.root, tree)
        self.tree = tree
        return self.tree

    def visual_tree(self, root, tree, depth=0):
        if root.is_leaf:
            node = pydot.Node(str("category")+str(root.value)+str(depth)+self.id_generator(), label=str(root.value), shape='hexagon', color='black')
            tree.add_node(node)
            pred_node = pydot.Node(str("class")+str(root.predict) + str(depth)+self.id_generator(), label=str(root.predict), shape='circle', color='red')
            tree.add_node(pred_node)
            tree.add_edge(pydot.Edge(node.get_name(), pred_node.get_name(), color='blue'))
            return node
        if root.is_feature:
            node = pydot.Node(str("feature")+str(root.value)+str(depth)+self.id_generator(), label=str(root.value), shape='box', color='black')
        else:
            node = pydot.Node(str("category")+str(root.value)+str(depth)+self.id_generator(), label=str(root.value), shape='hexagon', color='black')
        tree.add_node(node)
        for child in root.children:
            child_node = self.visual_tree(child, tree, depth+1)
            tree.add_node(child_node)
            tree.add_edge(pydot.Edge(node.get_name(), child_node.get_name(), color='blue'))
        return node

    def accuracy(self, classified_as, true_class):
        if len(true_class) != len(classified_as):
            raise Exception("The length of both arrays must be equal")
        length = len(true_class)
        trues = 0
        for classified, true in zip(classified_as, true_class):
            if classified == true:
                trues += 1
        return (trues/length)*100
