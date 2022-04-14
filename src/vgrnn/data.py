import pickle
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    # assert dataset in ['dblp', 'enron10', 'fb']
    with open(dataset + '/adj_time_list.pickle'.format(dataset), 'rb') as handle:
        adj_time_list = pickle.load(handle, encoding='latin1')

    with open(dataset + '/adj_orig_dense_list.pickle'.format(dataset), 'rb') as handle:
        adj_orig_dense_list = pickle.load(handle, fix_imports=True, encoding='bytes')

    return adj_time_list, adj_orig_dense_list