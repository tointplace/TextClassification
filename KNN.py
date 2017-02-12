import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

class Node:

    def __init__(self, data, parent):
        self.data = data
        self.parent = parent
        self.left_child = None
        self.right_child = None

    def add_left_child(self, Node):
        self.left_child = Node

    def add_right_child(self, Node):
        self.right_child = Node

    def get_left_child(self):
        return self.left_child

    def get_right_child(self):
        return self.right_child

    def get_parent(self):
        return self.parent

    def get_data(self):
        return self.data


class KNN:

    def __init__(self, data_matrix, k, weight_vector):
        self.data_matrix = data_matrix
        self.weight_vector = weight_vector
        self.k = k
        self.n_pts, self.dim = data_matrix.shape()

    def build_tree(self):
        #shuffle the data in data_matrix, sequentially get the points
        np.random.shuffle(self.data_matrix)
        root_node = Node(self.data_matrix[0], None)

        parent_node = root_node
        cd = 0
        for i in range(self.n_pts):
            if self.data_matrix[i][cd] > parent_node.data[cd]:
                parent_node.add_right_child()
            else:
                parent_node.add_left_child()



    def