import torch
import torch.nn as nn
from torch_sparse import SparseTensor

class LINK(nn.Module):
    """
    LINK: Logistic Regression on Adjacency Matrix
    """

    def __init__(self, num_nodes, out_channels):
        super(LINK, self).__init__()
        self.W = nn.Linear(num_nodes, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()
    
    def forward(self, data):
        N = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            A = SparseTensor(row=row, col=col, sparse_sizes=(N, N)).to_torch_sparse_coo_tensor()
        else:
            A = edge_index.to_torch_sparse_coo_tensor()
        return self.W(A)
