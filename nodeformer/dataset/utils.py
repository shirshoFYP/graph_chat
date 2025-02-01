import os
from collections import defaultdict
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from scipy import sparse as sp
from torch_sparse import SparseTensor


def rand_train_test_idx(label, train_prop=0.5, valid_prop=0.25, ignore_negative=True):
    if ignore_negative:
        label_nodes = torch.where(label != -1)[0]
    else:
        label_nodes = label
    n = label_nodes.shape[0]
    train_n = int(n * train_prop)
    valid_n = int(n * valid_prop)
    perm = torch.as_tensor(np.random.permutation(n))
    train_indices = perm[:train_n]
    valid_indices = perm[train_n : train_n + valid_n]
    test_indices = perm[train_n + valid_n :]

    if not ignore_negative:
        return train_indices, valid_indices, test_indices

    return (
        label_nodes[train_indices],
        label_nodes[valid_indices],
        label_nodes[test_indices],
    )


def load_fixed_splits(data_dir, dataset, name, protocol):
    split_list = []
    if name in ["cora", "citeseer", "pubmed"] and protocol == "semi":
        splits = {}
        splits["train"] = torch.as_tensor(dataset.train_idx)
        splits["valid"] = torch.as_tensor(dataset.val_idx)
        splits["test"] = torch.as_tensor(dataset.test_idx)
        split_list.append(splits)
    elif name in [
        "cora",
        "citeseer",
        "pubmed",
        "chameleon",
        "squirrel",
        "film",
        "cornell",
        "texas",
        "wisconsin",
    ]:
        for i in range(10):
            splits_file_path = (
                "{}/geom-gcn/splits/{}".format(data_dir, name)
                + "_split_0.6_0.2_"
                + str(i)
                + ".npz"
            )
            with np.load(splits_file_path) as splits_file:
                splits = {}
                splits["train"] = torch.BoolTensor(splits_file["train_mask"])
                splits["valid"] = torch.BoolTensor(splits_file["valid_mask"])
                splits["test"] = torch.BoolTensor(splits_file["test_mask"])
            split_list.append(splits)
    else:
        raise ValueError("Invalid dataset name")
    return split_list


def class_rand_splits(label, label_num_per_class):
    train_idx, non_train_idx = [], []
    idx = torch.arange(label.shape[0])
    class_list = label().squeeze().unique()
    valid_num, test_num = 500, 1000
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label.squeeze() == c_i]
        n_i = idx_i.shape[0]
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:label_num_per_class].tolist()
        non_train_idx += rand_idx[label_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    non_train_idx = torch.as_tensor(non_train_idx)
    non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0])]
    valid_idx, test_idx = (
        non_train_idx[:valid_num],
        non_train_idx[valid_num : valid_num + test_num],
    )
    return train_idx, valid_idx, test_idx


def even_quantile_labels(vals, num_classes, verbose=True):
    """partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=int)
    interval_list = []
    lower = np.inf
    for k in range(num_classes - 1):
        upper = np.quantile(vals, (k + 1) / num_classes)
        interval_list.append((lower, upper))
        indices = (vals >= lower) & (vals < upper)  # & instead of *
        label[indices] = k
        lower = upper
    label[vals >= lower] = num_classes - 1
    interval_list.append((lower, np.inf))
    if verbose:
        print("Class Label Intervals:")
        for class_idx, interval in enumerate(interval_list):
            print(f"Class {class_idx}: [{interval[0]}, {interval[1]})]")
    return label


def to_planetoid(dataset):
    """
    Takes in a NCDataset and returns the dataset in H2GCN Planetoid form, as follows:
    x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    test_x => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    all_x => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    test_y => the one-hot labels of the test instances as numpy.ndarray object;
    all_y => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    split_idx => The ogb dictionary that contains the train, valid, test splits
    """
    split_idx = dataset.get_idx_split("random", 0.25)
    train_idx, valid_idx, test_idx = (
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
    )

    graph, label = dataset[0]

    label = torch.squeeze(label)

    print("generate x")
    x = graph["node_feat"][train_idx].numpy()
    x = sp.csr_matrix(x)

    test_x = graph["node_feat"][test_idx].numpy()
    test_x = sp.csr_matrix(test_x)

    all_x = graph["node_feat"].numpy()
    all_x = sp.csr_matrix(all_x)

    y = F.one_hot(label[train_idx]).numpy()
    test_y = F.one_hot(label[test_idx]).numpy()
    all_y = F.one_hot(label).numpy()

    edge_index = graph["edge_index"].T

    graph = defaultdict(list)

    for i in range(0, label.shape[0]):
        graph[i].append(i)

    for start_edge, end_edge in edge_index:
        graph[start_edge.item()].append(end_edge.item())

    return x, test_x, all_x, y, test_y, all_y, graph, split_idx


def to_sparse_tensor(edge_index, edge_weight, num_nodes):
    (row, col) = edge_index
    perm = (col * num_nodes + row).argsort()
    row, col = row[perm], col[perm]
    val = edge_weight[perm] if edge_weight is not None else None
    adj_t = SparseTensor(
        row=row, col=col, value=val, sparse_sizes=(num_nodes, num_nodes), is_sorted=True
    )

    adj_t.storage.rowptr()
    adj_t.storage.csr2csc()
    return adj_t


def normalize(edge_index):
    adj_t = edge_index.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    return deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)


def gen_normalized_adjs(dataset):
    row, col = dataset.graph["edge_index"]
    num_nodes = dataset.graph["num_nodes"]
    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    deg_adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    deg_a = deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(1, -1) * adj
    adj_deg = adj * deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(1, -1)
    return deg_adj, deg_a, adj_deg


def convert_to_adj(edge_index, num_nodes):
    adj = torch.zeros(num_nodes, num_nodes)
    row, col = edge_index
    adj[row, col] = 1
    return adj


def adj_mul(adj_i, adj, num_nodes):
    adj_i_sparse = torch.sparse_coo_tensor(
        adj_i,
        torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device),
        (num_nodes, num_nodes),
    )
    adj_sparse = torch.sparse_coo_tensor(
        adj,
        torch.ones(adj.shape[1], dtype=torch.float).to(adj.device),
        (num_nodes, num_nodes),
    )
    adj_j = torch.sparse.mm(adj_i_sparse, adj_sparse).coalesce().indices()
    return adj_j
