import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from sklearn.neighbors import kneighbors_graph

from model.nodeformer import NodeFormer
from trainer.logger import Logger
from dataset.dataset import load_dataset
from dataset.utils import load_fixed_splits, adj_mul
from dataset.gpu_util import get_gpu_memory_map
from trainer.eval import evaluate, eval_acc, eval_rocauc, eval_f1
from parser import get_parser
import time

import warnings

warnings.filterwarnings("ignore")


# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = get_parser()
    args = parser.parse_args()
    fix_seed(args.seed)

    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)

    # get the splits for all runs
    if args.rand_split:
        split_idx_lst = [
            dataset.get_idx_split(
                train_prop=args.train_prop, valid_prop=args.valid_prop
            )
            for _ in range(args.runs)
        ]
    elif args.rand_split_class:
        split_idx_lst = [
            dataset.get_idx_split(
                split_type="class", label_num_per_class=args.label_num_per_class
            )
            for _ in range(args.runs)
        ]
    elif args.dataset in ["ogbn-arxiv", "ogbn-products", "amazon2m"]:
        split_idx_lst = [dataset.load_fixed_splits() for _ in range(args.runs)]
    else:
        split_idx_lst = load_fixed_splits(
            args.data_dir, dataset, name=args.dataset, protocol=args.protocol
        )

    #
    if args.dataset in ("mini", "20news"):
        adj_knn = kneighbors_graph(
            dataset.graph["node_feat"], n_neighbors=args.knn_num, include_self=True
        )
        edge_index = torch.tensor(adj_knn.nonzero(), dtype=torch.long)
        dataset.graph["edge_index"] = edge_index

    ### Basic information of datasets ###
    n = dataset.graph["num_nodes"]
    e = dataset.graph["edge_index"].shape[1]
    # infer the number of classes for non one-hot and one-hot labels
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph["node_feat"].shape[1]

    print(
        f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}"
    )

    # whether or not to symmetrize
    if not args.directed:
        dataset.graph["edge_index"] = to_undirected(dataset.graph["edge_index"])

    dataset.graph["edge_index"], dataset.graph["node_feat"] = dataset.graph[
        "edge_index"
    ].to(device), dataset.graph["node_feat"].to(device)

    ### Load method ###
    model = NodeFormer(
        d,
        args.hidden_channels,
        c,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_heads=args.num_heads,
        use_bn=args.use_bn,
        nb_random_features=args.M,
        use_gumbel=args.use_gumbel,
        use_residual=args.use_residual,
        use_act=args.use_act,
        use_jk=args.use_jk,
        nb_gumbel_sample=args.K,
        rb_order=args.rb_order,
        rb_trans=args.rb_trans,
    ).to(device)
    ### Loss function (Single-class, Multi-class) ###
    if args.dataset in (
        "deezer-europe",
        "twitch-e",
        "fb100",
    ):
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.NLLLoss()

    ### Performance metric (Acc, AUC, F1) ###
    if args.metric == "rocauc":
        eval_func = eval_rocauc
    elif args.metric == "f1":
        eval_func = eval_f1
    else:
        eval_func = eval_acc

    logger = Logger(args.runs, args)

    model.train()
    print("MODEL:", model)

    ### Adj storage for relational bias ###
    adjs = []
    adj, _ = remove_self_loops(dataset.graph["edge_index"])
    adj, _ = add_self_loops(adj, num_nodes=n)
    adjs.append(adj)
    for i in range(args.rb_order - 1):  # edge_index of high order adjacency
        adj = adj_mul(adj, adj, n)
        adjs.append(adj)
    dataset.graph["adjs"] = adjs

    ### Training loop ###
    for run in range(args.runs):
        if args.dataset in ["cora", "citeseer", "pubmed"] and args.protocol == "semi":
            split_idx = split_idx_lst[0]
        else:
            split_idx = split_idx_lst[run]
        train_idx = split_idx["train"].to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(
            model.parameters(), weight_decay=args.weight_decay, lr=args.lr
        )
        best_val = float("-inf")

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            if args.method == "nodeformer":
                out, link_loss_ = model(
                    dataset.graph["node_feat"], dataset.graph["adjs"], args.tau
                )
            else:
                out = model(dataset)

            if args.dataset in (
                "deezer-europe",
                "twitch-e",
                "fb100",
            ):
                if dataset.label.shape[1] == 1:
                    true_label = F.one_hot(
                        dataset.label, dataset.label.max() + 1
                    ).squeeze(1)
                else:
                    true_label = dataset.label
                loss = criterion(
                    out[train_idx], true_label.squeeze(1)[train_idx].to(torch.float)
                )
            else:
                out = F.log_softmax(out, dim=1)
                loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])
            if args.method == "nodeformer":
                loss -= args.lamda * sum(link_loss_) / len(link_loss_)
            loss.backward()
            optimizer.step()

            if epoch % args.eval_step == 0:
                result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
                logger.add_result(run, result[:-1])

                if result[1] > best_val:
                    best_val = result[1]
                    if args.save_model:
                        torch.save(
                            model.state_dict(),
                            args.model_dir + f"{args.dataset}-{args.method}.pkl",
                        )

                print(
                    f"Epoch: {epoch:02d}, "
                    f"Loss: {loss:.4f}, "
                    f"Train: {100 * result[0]:.2f}%, "
                    f"Valid: {100 * result[1]:.2f}%, "
                    f"Test: {100 * result[2]:.2f}%"
                )
        logger.print_statistics(run)

    results = logger.print_statistics()
    if args.save_result:
        filename = f"results/{args.dataset}.csv"
        with open(filename, "a+") as f:
            f.write(f"{args.method}," + ",".join([str(x) for x in results]) + "\n")


if __name__ == "__main__":
    main()
