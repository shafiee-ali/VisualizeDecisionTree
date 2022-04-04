import os
import random
import string
from math import log2
import pydot
import pandas as pd
import numpy as np
from PIL import Image
import ntpath


class Node:
    def __init__(self):
        self.children = []
        self.is_leaf = False
        self.value = ""
        self.predict = ""
        self.is_feature = True

def base_entropy(p):
    if p == 0:
        return 0
    return -1*p*log2(p)

def attribute_entropy(dataset, attribute_column_name):
    attribute_column = dataset[attribute_column_name]
    categories, count = np.unique(attribute_column, return_counts=True)
    target_attribute = tarin[target_class_name]
    labels, labels_count = np.unique(target_attribute, return_counts=True)
    probabilities = count / record_length
    categories_ent = np.zeros((len(categories),))
    i = 0
    for count, category in zip(count, categories):
        data = dataset[dataset[attribute_column_name] == category]
        ent = 0
        for label in labels:
            p = len(data[data[target_class_name] == label])
            ent += base_entropy(p/count)
        categories_ent[i] = ent
        i += 1
    weighted_ent = np.multiply(categories_ent, probabilities)
    attribute_entropy = np.sum(weighted_ent)
    return attribute_entropy

def dataset_entropy(dataset):
    length = len(dataset)
    target_attribute = dataset[target_class_name]
    labels, labels_count = np.unique(target_attribute, return_counts=True)
    ent = 0
    for lbl_count in labels_count:
        ent += base_entropy(lbl_count/length)
    return ent

def is_pure(dataset):
    return dataset_entropy(dataset) == 0

def remove_best_from_attributes(attrs, best):
    new_attrs = attrs.copy()
    new_attrs.remove(best)
    return new_attrs

def ID3(examples, attributes):
    root = Node()
    min_ent = 1
    best_feature = None
    for feature in attributes:
        ent = attribute_entropy(examples, feature)
        if ent < min_ent:
            min_ent = ent
            best_feature = feature
    root.value = best_feature
    uniq_values = np.unique(examples[best_feature])
    for v in uniq_values:
        sub_dataset = examples[examples[best_feature] == v]
        if is_pure(sub_dataset):
            leaf_node = Node()
            leaf_node.is_feature = False
            leaf_node.is_leaf = True
            leaf_node.value = v
            leaf_node.predict = np.unique(sub_dataset[target_class_name])[0]
            root.children.append(leaf_node)
        else:
            dummy = Node()
            dummy.value = v
            dummy.is_feature = False
            new_attributes = remove_best_from_attributes(attributes, best_feature)
            if len(new_attributes) == 0:
                dummy.is_leaf = True
                counts = np.bincount(sub_dataset[target_class_name])
                common_label = np.argmax(counts)
                dummy.predict = common_label
            else:
                child = ID3(sub_dataset, new_attributes)
                dummy.children.append(child)
            root.children.append(dummy)
    return root


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def visual_tree(root, tree, depth=0):
    if root.is_leaf:
        node = pydot.Node(str("category")+str(root.value)+str(depth)+id_generator(), label=str(root.value), shape='hexagon', color='black')
        tree.add_node(node)
        pred_node = pydot.Node(str("class")+str(root.predict) + str(depth)+id_generator(), label=str(root.predict), shape='circle', color='red')
        tree.add_node(pred_node)
        tree.add_edge(pydot.Edge(node.get_name(), pred_node.get_name(), color='blue'))
        return node
    if root.is_feature:
        node = pydot.Node(str("feature")+str(root.value)+str(depth)+id_generator(), label=str(root.value), shape='box', color='black')
    else:
        node = pydot.Node(str("category")+str(root.value)+str(depth)+id_generator(), label=str(root.value), shape='hexagon', color='black')
    tree.add_node(node)
    for child in root.children:
        child_node = visual_tree(child, tree, depth+1)
        tree.add_node(child_node)
        tree.add_edge(pydot.Edge(node.get_name(), child_node.get_name(), color='blue'))
    return node



""" create Images dir for save visualizations """
current_path = os.getcwd()
images_dir = os.path.join(current_path, 'VisualizeDecisionTrees')
is_exist = os.path.exists(images_dir)
if not is_exist:
    os.mkdir(images_dir)

train_dir = os.path.join(current_path, 'DataSet/Train')
test_dir = os.path.join(current_path, 'DataSet/Test')


for training_file in os.listdir(train_dir):
    tarin = pd.read_csv(f'{train_dir}\\{training_file}')
    test = pd.read_csv('./DataSet/Test/monks_test.csv')

    target_class_name = tarin.columns[-1]
    record_length = len(tarin)

    attributes = list(tarin.columns)[1:-1]
    root = ID3(tarin, attributes)
    tree = pydot.Dot(graph_name="my_decision_tree", graph_type="graph", bgcolor="white")
    visual_tree(root, tree)
    tree.write("tree.dot")
    tree.write_png(f'{images_dir}\\{training_file}_decision_tree.png')

