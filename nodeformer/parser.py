import argparse


def parser_add_main_args(parser: argparse.ArgumentParser):
    # dataset, protocol
    parser.add_argument("--method", "-m", type=str, default="nodeformer")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--sub_dataset", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument(
        "--device", type=int, default=0, help="which gpu to use if any (default: 0)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval_step", type=int, default=1, help="how often to print")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--runs", type=int, default=1, help="number of distinct runs")
    parser.add_argument(
        "--train_prop", type=float, default=0.5, help="training label proportion"
    )
    parser.add_argument(
        "--valid_prop", type=float, default=0.25, help="validation label proportion"
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="semi",
        help="protocol for cora datasets with fixed splits, semi or supervised",
    )
    parser.add_argument("--rand_split", action="store_true", help="use random splits")
    parser.add_argument(
        "--rand_split_class",
        action="store_true",
        help="use random splits with a fixed number of labeled nodes for each class",
    )
    parser.add_argument(
        "--label_num_per_class",
        type=int,
        default=20,
        help="labeled nodes randomly selected",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="acc",
        choices=["acc", "rocauc", "f1"],
        help="evaluation metric",
    )
    parser.add_argument(
        "--knn_num", type=int, default=5, help="number of k for KNN graph"
    )
    parser.add_argument(
        "--save_model", action="store_true", help="whether to save model"
    )
    parser.add_argument("--model_dir", type=str, default="../model/")

    # hyper-parameter for model arch and training
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument(
        "--num_layers", type=int, default=2, help="number of layers for deep methods"
    )

    # hyper-parameter for nodeformer
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--M", type=int, default=30, help="number of random features")
    parser.add_argument(
        "--use_gumbel",
        action="store_true",
        help="use gumbel softmax for message passing",
    )
    parser.add_argument(
        "--use_residual",
        action="store_true",
        help="use residual link for each GNN layer",
    )
    parser.add_argument("--use_bn", action="store_true", help="use layernorm")
    parser.add_argument(
        "--use_act", action="store_true", help="use non-linearity for each layer"
    )
    parser.add_argument(
        "--use_jk",
        action="store_true",
        help="concat the layer-wise results in the final layer",
    )
    parser.add_argument(
        "--K", type=int, default=10, help="num of samples for gumbel softmax sampling"
    )
    parser.add_argument(
        "--tau", type=float, default=0.25, help="temperature for gumbel softmax"
    )
    parser.add_argument(
        "--lamda", type=float, default=0.1, help="weight for edge reg loss"
    )
    parser.add_argument(
        "--rb_order",
        type=int,
        default=0,
        help="order for relational bias, 0 for not use",
    )
    parser.add_argument(
        "--rb_trans",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "identity"],
        help="non-linearity for relational bias",
    )
    parser.add_argument("--batch_size", type=int, default=10000)

    # hyper-parameter for gnn baseline
    parser.add_argument(
        "--hops",
        type=int,
        default=1,
        help="power of adjacency matrix for certain methods",
    )
    parser.add_argument("--cached", action="store_true", help="set to use faster sgc")
    parser.add_argument(
        "--gat_heads", type=int, default=8, help="attention heads for gat"
    )
    parser.add_argument("--out_heads", type=int, default=1, help="out heads for gat")
    parser.add_argument(
        "--projection_matrix_type",
        type=bool,
        default=True,
        help="use projection matrix or not",
    )
    parser.add_argument(
        "--lp_alpha", type=float, default=0.1, help="alpha for label prop"
    )
    parser.add_argument("--gpr_alpha", type=float, default=0.1, help="alpha for gprgnn")
    parser.add_argument(
        "--directed", action="store_true", help="set to not symmetrize adjacency"
    )
    parser.add_argument(
        "--jk_type",
        type=str,
        default="max",
        choices=["max", "lstm", "cat"],
        help="jumping knowledge type",
    )
    parser.add_argument(
        "--num_mlp_layers", type=int, default=1, help="number of mlp layers in h2gcn"
    )

    parser.add_argument(
        "--save_result", action="store_true", help="whether to save result"
    )


def get_parser():
    parser = argparse.ArgumentParser(description="NodeFormer")
    parser_add_main_args(parser)
    return parser
