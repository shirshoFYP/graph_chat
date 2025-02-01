import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import List


class SeparateGCNConv(MessagePassing):
    """
    A custom Graph Convolutional Network (GCN) layer that separates the input features into content and position components,
    applies linear transformations to each component, and then combines them for message passing.
    Attributes:
        d_content (int): Dimension of the content features, which is half of the input model dimension.
        linear_c (nn.Module): Linear transformation layer for the content features.
        linear_p (nn.Module): Linear transformation layer for the position features.
    Args:
        d_model (int): Dimension of the input features.
    """

    d_content: int
    linear_c: nn.Module
    linear_p: nn.Module

    def __init__(self, d_model: int):
        super(SeparateGCNConv, self).__init__(aggr="add")
        self.d_content = d_model // 2
        d_position = d_model
        self.linear_c = nn.Linear(self.d_content, self.d_content)
        self.linear_p = nn.Linear(d_position, d_position)

    def forward(self, x, edge_index):
        """
        Forward pass for the SeparateGCNConv layer.
            x (Tensor): Input feature matrix.
            edge_index (Tensor): Edge indices.
        Returns:
            Tensor: Output feature matrix after message passing.
        """
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x_c = self.linear_c(x[:, : self.d_content])
        x_p = self.linear_p(x[:, self.d_content :])
        x = torch.cat([x_c, x_p], dim=-1)

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        y = self.propagate(edge_index, x=x, norm=norm, size=(x.size(0), x.size(0)))
        return y

    def message(self, x_j, norm):
        """
        Constructs the messages to be passed to neighboring nodes.
            x_j (Tensor): Feature matrix of neighboring nodes.
            norm (Tensor): Normalization coefficients.
        Returns:
            Tensor: Normalized messages.
        """
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        """
        Updates the node embeddings after message passing.
            aggr_out (Tensor): Aggregated messages.
        Returns:
            Tensor: Updated node embeddings.
        """
        return aggr_out


class GNN(nn.Module):
    """
    A custom Graph Neural Network (GNN) that consists of multiple GCN layers.
    Attributes:
        convs (List[SeparateGCNConv]): List of GCN layers.
    Args:
        d_model (int): Dimension of the input features.
        n_layers (int): Number of GCN layers.
    """

    convs: List[nn.Module]

    def __init__(self, d_model: int, n_layers: int):
        super(GNN, self).__init__()
        self.convs = []
        self.layer_norms = []
        for i in range(n_layers):
            conv = SeparateGCNConv(d_model)
            layer_norm = nn.LayerNorm([d_model])
            self.convs.append(conv)
            self.layer_norms.append(layer_norm)
            self.add_module(f"conv_{i}", conv)
            self.add_module(f"layer_norm_{i}", layer_norm)

    def forward(self, x, edge_index, nodes_mask):
        """
        Forward pass for the GNN.
            x (Tensor): Input feature matrix.
            edge_index (Tensor): Edge indices.
        Returns:
            Tensor: Output feature matrix after message passing.
        """
        # the last layer does not have a ReLU activation
        for conv, layer_norm in zip(self.convs[:-1], self.layer_norms[:-1]):
            x += F.relu(layer_norm(conv(x, edge_index)))
        x += self.layer_norms[-1](self.convs[-1](x, edge_index))

        return x[nodes_mask]
