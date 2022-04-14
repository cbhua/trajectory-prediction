import numpy as np
import torch
import scipy.sparse as sp
from src.vgrnn.data import load_data

seed = 146
np.random.seed(seed)

def sparse_to_tuple(sparse_mx):
# Input: a sparse matrix
# Output: Coords, values and shape
# coords.shape = (#Edges, 2)
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
# return symmetrically normalized adjacency matrix
    edge_feature = adj.shape[0]
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(edge_feature)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(
        degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def ismember(a, b, tol=5):
# If a is/has a member of b
    rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    return np.any(rows_close)

def getEdges(adj):
# Input: Sparse adjacent
# Output: Single-direction graph edges with and without self-loop: List
#         edges.shpae = (#Edges, 2)
    adj = adj - sp.dia_matrix(
        (adj.diagonal()[np.newaxis, :], [0]),
        shape=adj.shape)
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0
    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    return edges, edges_all

def makeFalseEdges(nums, size, edgeSet):
    edges_false = []
    while len(edges_false) < nums:
        h = np.random.randint(0, size)
        t = np.random.randint(0, size)
        if h == t: continue
        e = [[h, t], [t, h]]
        if edges_false and ismember(e, np.array(edges_false)): continue
        if ismember(e, edgeSet): continue
        edges_false.append([h, t])
    assert ~ismember(edges_false, edgeSet)
    return edges_false

def mask_edges_det(adjs_list):
# Input: Time series Sparse adjacent
# Output: Adjacents used to train: Sparse
#         Train(70%)/validate(20%)/test(10%) edges: List
    adj_train_l, train_edges_l = [], []
    val_edges_l, val_edges_false_l = [], []
    test_edges_l, test_edges_false_l = [], []
    edges_list = []
    edge_feature = adjs_list[0].shape[0]
    for i in range(0, len(adjs_list)):
        adj = adjs_list[i]
        edges, edges_all = getEdges(adj)
        num_test = int(np.floor(edges.shape[0] / 10.))
        num_val = int(np.floor(edges.shape[0] / 20.))

        all_edge_idx = range(edges.shape[0])
        np.random.shuffle([all_edge_idx])
        val_edge_idx = all_edge_idx[:num_val]
        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        test_edges = edges[test_edge_idx]
        val_edges = edges[val_edge_idx]
        train_edges = np.delete(edges, np.hstack(
            [test_edge_idx, val_edge_idx]), axis=0)

        edges_list.append(edges)

        test_edges_false = makeFalseEdges(len(test_edges), edge_feature, edges_all)
        val_edges_false = makeFalseEdges(len(val_edges), edge_feature, edges_all)
        assert ~ismember(val_edges, train_edges)
        assert ~ismember(test_edges, train_edges)
        assert ~ismember(val_edges, test_edges)

        data = np.ones(train_edges.shape[0])

        adj_train = sp.csr_matrix(
            (data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        adj_train = adj_train + adj_train.T

        adj_train_l.append(adj_train)
        train_edges_l.append(train_edges)
        val_edges_l.append(val_edges)
        val_edges_false_l.append(val_edges_false)
        test_edges_l.append(test_edges)
        test_edges_false_l.append(test_edges_false)

    return adj_train_l, train_edges_l, val_edges_l, val_edges_false_l, test_edges_l, test_edges_false_l


def mask_edges_prd(adjs_list):
# Input: A list of Sparse adjacent
# Output: Ture edges and false edges
#         pos_edges_l.shape = (t, #Edges, 2)
    pos_edges_l, false_edges_l = [], []
    edge_feature = adjs_list[0].shape[0]
    for i in range(0, len(adjs_list)):
        adj = adjs_list[0]
        edges, edges_all = getEdges(adj)
        num_false = int(edges.shape[0])
        pos_edges_l.append(edges)
        edges_false = makeFalseEdges(num_false, edge_feature, edges_all)
        false_edges_l.append(edges_false)
    return pos_edges_l, false_edges_l


def mask_edges_prd_new(adjs_list, adj_orig_dense_list):
# Input: A list of Sparse adjacent
# Output: Ture new edges and false new edges
#         pos_edges_l.shape = (t, #New Edges, 2)
    pos_edges_l, false_edges_l = [], []
    edge_feature = adjs_list[0].shape[0]
    edges, edges_all = getEdges(adjs_list[0])
    num_false = int(edges.shape[0])
    pos_edges_l.append(edges)
    edges_false = makeFalseEdges(num_false, edge_feature, edges_all)
    false_edges_l.append(np.asarray(edges_false))

    for i in range(1, len(adjs_list)):
        edges_pos = np.transpose(np.asarray(
            np.where((adj_orig_dense_list[i] - adj_orig_dense_list[i-1]) > 0)))
        num_false = int(edges_pos.shape[0])
        adj = adjs_list[i]
        edges, edges_all = getEdges(adj)
        edges_false = makeFalseEdges(len(edges_false), edge_feature, edges_all)
        false_edges_l.append(np.asarray(edges_false))
        pos_edges_l.append(edges_pos)

    return pos_edges_l, false_edges_l

def transpose_list(train_edges_l):
# Input: Edges list for train: List
# Output: Transpose of input
    edge_idx_list = []
    for i in range(len(train_edges_l)):
        edge_idx_list.append(torch.tensor(
        np.transpose(train_edges_l[i]), dtype=torch.long))
    return edge_idx_list