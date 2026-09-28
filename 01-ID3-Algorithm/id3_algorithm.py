import pandas as pd
import math
import numpy as np


# Create dataset
data = pd.read_csv("weather.csv")

features = list(data.columns)
features.remove("play")
features.remove("id")


# Node class for Decision Tree
class Node:
    def __init__(self):
        self.children = []
        self.value = ""
        self.isLeaf = False
        self.pred = ""


# Calculate entropy
def entropy(examples):
    pos = 0.0
    neg = 0.0

    for _, row in examples.iterrows():
        if row["play"] == "yes":
            pos += 1
        else:
            neg += 1

    if pos == 0.0 or neg == 0.0:
        return 0.0

    p = pos / (pos + neg)
    n = neg / (pos + neg)

    return -(p * math.log(p, 2) + n * math.log(n, 2))


# Calculate information gain
def info_gain(examples, attr):

    uniq = np.unique(examples[attr])

    gain = entropy(examples)

    for u in uniq:
        subdata = examples[examples[attr] == u]
        sub_entropy = entropy(subdata)

        gain -= (len(subdata) / len(examples)) * sub_entropy

    return gain


# ID3 Algorithm
def ID3(examples, attrs):

    root = Node()

    max_gain = 0
    max_feat = ""

    for feature in attrs:

        gain = info_gain(examples, feature)

        if gain > max_gain:
            max_gain = gain
            max_feat = feature


    root.value = max_feat


    unique_values = np.unique(examples[max_feat])


    for value in unique_values:

        subdata = examples[examples[max_feat] == value]


        if entropy(subdata) == 0.0:

            leaf = Node()

            leaf.isLeaf = True
            leaf.value = value
            leaf.pred = np.unique(subdata["play"])[0]

            root.children.append(leaf)


        else:

            child_node = Node()

            child_node.value = value

            new_attrs = attrs.copy()
            new_attrs.remove(max_feat)


            child = ID3(subdata, new_attrs)

            child_node.children.append(child)

            root.children.append(child_node)


    return root



# Display decision tree
def printTree(root, depth=0):

    print("\t" * depth, end="")

    print(root.value, end="")


    if root.isLeaf:

        print(" -> ", root.pred)

    else:

        print()


    for child in root.children:

        printTree(child, depth + 1)



# Classification
def classify(root, new):

    for child in root.children:

        if child.value == new[root.value]:

            if child.isLeaf:

                print(
                    "Predicted Label:",
                    child.pred
                )

            else:

                classify(child.children[0], new)



# Build tree
root = ID3(data, features)


print("Decision Tree:")
printTree(root)


print("------------------")


# Test sample
new_example = {
    "outlook": "sunny",
    "temperature": "hot",
    "humidity": "normal",
    "windy": "False"
}


classify(root, new_example)