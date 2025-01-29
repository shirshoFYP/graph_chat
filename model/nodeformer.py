import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree


def create_projection_matrix(r, c, seed=0, scaling=0, struct_mode=False):
    """
    Create a projection matrix of size r x c. The matrix is created by
    stacking random orthogonal matrices of size c x c. The number of
    orthogonal matrices is equal to the number of full blocks of c x c
    matrices that can be fit in the r x c matrix. The remaining rows are
    filled with a random orthogonal matrix of size remaining_rows x c.

    Args:
        m (int): Number of rows of the projection matrix.
        c (int): Number of columns of the projection matrix.
        seed (int): Seed for random number generator.
        scaling (int): Scaling mode for the projection matrix.
            0: Random scaling.
            1: Fixed scaling.
        struct_mode (bool): If True, the orthogonal matrices are created by
            multiplying random Householder reflections. If False, the orthogonal
            matrices are created by generating random orthogonal matrices.

    Returns:
        torch.Tensor: The projection matrix.

    Changes:
        1. torch.qr() -> torch.linalg.qr()
    """

    num_full_blocks = int(r // c)
    block_list = []
    curr_seed = seed
    for _ in range(num_full_blocks):
        torch.manual_seed(curr_seed)
        if struct_mode:
            q = create_products_of_given_rotations(c, curr_seed)
        else:
            unstructured_block = torch.randn((c, c))
            # qr decomposition
            q, _ = torch.linalg.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q)
        curr_seed += 1
    remaining_rows = r - c * num_full_blocks
    if remaining_rows > 0:
        torch.manual_seed(curr_seed)
        if struct_mode:
            q = create_products_of_given_rotations(remaining_rows, curr_seed)
        else:
            unstructured_block = torch.randn((c, c))
            q, _ = torch.linalg.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q[:remaining_rows])
    final_matrix = torch.vstack(block_list)

    curr_seed += 1
    torch.manual_seed(curr_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((r, c)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(c))) * torch.ones(r)
    else:
        raise ValueError("Invalid scaling value")

    return torch.matmul(torch.diag(multiplier), final_matrix)


def create_products_of_given_rotations(dim, seed):
    """
    Create a random orthogonal matrix by multiplying random Householder
    reflections.

    Args:
        dim (int): Dimension of the orthogonal matrix.
        seed (int): Seed for random number generator.

    Returns:
        torch.Tensor: The orthogonal matrix.

    Changes:
        1. np.random.choice(dim, 2) -> np.random.choice(dim, 2, replace=False)
    """

    num_given_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim)
    np.random.seed(seed)
    for _ in range(num_given_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2, replace=False)  # without replacement
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i, slice_j = q[index_i], q[index_j]
        new_slice_i = (
            math.cos(random_angle) * slice_i + math.sin(random_angle) * slice_j
        )
        new_slice_j = (
            -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
        )
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)


def relu_kernel_transformation(
    data, is_query, proj_mat=None, numerical_stabilizer=0.001
):
    """
    Apply ReLU to the transformed data.
    Args:
        data (torch.Tensor): The input data.
        is_query (bool): If True, the data is a query.
        proj_mat (torch.Tensor): The projection matrix.
        numerical_stabilizer (float): The numerical stabilizer.
    Returns:
        torch.Tensor: The output data
    Explanation:
        bnhd,md->bnhm: batch matrix multiplication with data and proj_mat as the two matrices.
        The output is a tensor of shape (batch_size, num_nodes, num_heads, num_features).

    """
    del is_query
    if proj_mat is None:
        return F.relu(data) + numerical_stabilizer
    ratio = 1.0 / torch.sqrt(torch.tensor(proj_mat.shape[0], dtype=torch.float32))
    data_dash = ratio * torch.einsum("bnhd,md->bnhm", data, proj_mat)
    return F.relu(data_dash) + numerical_stabilizer


def softmax_kernel_transformation(
    data, is_query, proj_mat=None, numerical_stabilizer=1e-6
):
    """
    Apply softmax to the transformed data.
    Args:
        data (torch.Tensor): The input data.
        is_query (bool): If True, the data is a query.
        proj_mat (torch.Tensor): The projection matrix.
        numerical_stabilizer (float): The numerical stabilizer.
    Returns:
        torch.Tensor: The output data
    """
    data_normalizer = 1.0 / torch.sqrt(
        torch.tensor(data.shape[-1], dtype=torch.float32)
    )
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(torch.tensor(proj_mat.shape[0], dtype=torch.float32))
    data_dash = torch.einsum("bnhd,md->bnhm", data, proj_mat)
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=len(data.shape) - 1) / 2.0
    diag_data = diag_data.unsqueeze(len(diag_data.shape) - 1)  # changed
    last_dims_t = len(data_dash.shape) - 1
    attention_dims_t = len(data_dash.shape) - 3
    if is_query:
        data_dash = ratio * (
            torch.exp(
                data_dash
                - diag_data
                - torch.max(data_dash, dim=last_dims_t, keepdim=True)[0]
            )
            + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
            torch.exp(
                data_dash
                - diag_data
                - torch.max(
                    torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                    dim=attention_dims_t,
                    keepdim=True,
                )[0]
            )
            + numerical_stabilizer
        )
    return data_dash


def numerator(query, key, vector):
    """
    Compute the numerator of the kernelized softmax.
    Args:
        query (torch.Tensor): The query tensor.
        key (torch.Tensor): The key tensor.
        vector (torch.Tensor): The vector tensor.
    Returns:
        torch.Tensor: The numerator tensor.
    """
    u_k = torch.einsum("nbhm,nbhd->bhmd", key, vector)
    return torch.einsum("nbhm,bhmd->nbhd", query, u_k)


def denominator(query, key):
    """
    Compute the denominator of the kernelized softmax.
    Args:
        query (torch.Tensor): The query tensor.
        key (torch.Tensor): The key tensor.
    Returns:
        torch.Tensor: The denominator tensor.
    """
    all_ones = torch.ones([key.shape[0]]).to(query.device)
    o_k = torch.einsum("nbhm,n->bhm", key, all_ones)
    return torch.einsum("nbhm,bhm->nbh", query, o_k)


def numerator_gumbel(query, key, vector):
    """
    Compute the numerator of the kernelized softmax with Gumbel transform.
    Args:
        query (torch.Tensor): The query tensor.
        key (torch.Tensor): The key tensor.
        vector (torch.Tensor): The vector tensor.
    Returns:
        torch.Tensor: The numerator tensor.
    """
    u_k = torch.einsum("nbhkm,nbhd->bhkmd", key, vector)
    return torch.einsum("nbhm,bhkmd->nbhkd", query, u_k)


def denominator_gumbel(query, key):
    """
    Compute the denominator of the kernelized softmax with Gumbel transform.
    Args:
        query (torch.Tensor): The query tensor.
        key (torch.Tensor): The key tensor.
        Returns:
        torch.Tensor: The denominator tensor.
    """
    all_ones = torch.ones([key.shape[0]]).to(query.device)
    o_k = torch.einsum("nbhkm,n->bhkm", key, all_ones)
    return torch.einsum("nbhm,bhkm->nbhk", query, o_k)


def kernelized_softmax(
    query,
    key,
    value,
    kernel_transformation,
    proj_mat=None,
    edge_index=None,
    tau=0.25,
    return_attention_weights=True,
):
    """
    Compute the kernelized softmax.
    Args:
        query (torch.Tensor): The query tensor.
        key (torch.Tensor): The key tensor.
        value (torch.Tensor): The value tensor.
        kernel_transformation (function): The kernel transformation function.
        proj_mat (torch.Tensor): The projection matrix.
        edge_index (torch.Tensor): The edge index tensor.
        tau (float): The temperature parameter.
        return_attention_weights (bool): If True, return the attention weights.
    Returns:
        torch.Tensor: The output tensor.
    """
    query /= math.sqrt(tau)
    key /= math.sqrt(tau)
    query_prime = kernel_transformation(
        query, True, proj_mat
    )  # (batch_size, num_nodes, num_heads, num_features)
    key_prime = kernel_transformation(
        key, False, proj_mat
    )  # (batch_size, num_nodes, num_heads, num_features)
    query_prime = query_prime.permute(
        1, 0, 2, 3
    )  # (num_nodes, batch_size, num_heads, num_features)
    key_prime = key_prime.permute(
        1, 0, 2, 3
    )  # (num_nodes, batch_size, num_heads, num_features)
    value = value.permute(
        1, 0, 2, 3
    )  # (num_nodes, batch_size, num_heads, num_features)

    z_numerator = numerator(query_prime, key_prime, value)
    z_denominator = denominator(query_prime, key_prime)

    z_numerator = z_numerator.permute(
        1, 0, 2, 3
    )  # (batch_size, num_nodes, num_heads, num_features)
    z_denominator = z_denominator.permute(1, 0, 2)  # (batch_size, num_nodes, num_heads)
    z_denominator = torch.unsqueeze(
        z_denominator, len(z_denominator.shape)
    )  # (batch_size, num_nodes, num_heads, 1)
    z_output = (
        z_numerator / z_denominator
    )  # (batch_size, num_nodes, num_heads, num_features)

    if return_attention_weights:
        start, end = edge_index
        query_end, key_start = query_prime[end], key_prime[start]
        edge_attention_num = torch.einsum(
            "ebhm, ebhm->ebh", query_end, key_start
        )  # (num_edges, batch_size, num_heads)
        edge_attention_num = edge_attention_num.permute(
            1, 0, 2
        )  # (batch_size, num_edges, num_heads)
        attention_norm = denominator(query_prime, key_prime)
        edge_attention_dem = attention_norm[end]
        edge_attention_dem = edge_attention_dem.permute(1, 0, 2)
        att_weight = edge_attention_num / edge_attention_dem
        return z_output, att_weight
    return z_output


def kernelized_softmax_gumbel(
    query,
    key,
    value,
    kernel_transformation,
    proj_mat=None,
    edge_index=None,
    k=10,
    tau=0.25,
    return_attention_weights=True,
):
    """
    Compute the kernelized softmax with Gumbel transform.
    Args:
        query (torch.Tensor): The query tensor.
        key (torch.Tensor): The key tensor.
        value (torch.Tensor): The value tensor.
        kernel_transformation (function): The kernel transformation function.
        proj_mat (torch.Tensor): The projection matrix.
        edge_index (torch.Tensor): The edge index tensor.
        k (int): The number of Gumbel samples.
        tau (float): The temperature parameter.
        return_attention_weights (bool): If True, return the attention weights.
    Returns:
        torch.Tensor: The output tensor.
    """
    query /= math.sqrt(tau)
    key /= math.sqrt(tau)
    query_prime = kernel_transformation(
        query, True, proj_mat
    )  # (batch_size, num_nodes, num_heads, num_features)
    key_prime = kernel_transformation(
        key, False, proj_mat
    )  # (batch_size, num_nodes, num_heads, num_features)
    query_prime = query_prime.permute(
        1, 0, 2, 3
    )  # (num_nodes, batch_size, num_heads, num_features)
    key_prime = key_prime.permute(
        1, 0, 2, 3
    )  # (num_nodes, batch_size, num_heads, num_features)
    value = value.permute(
        1, 0, 2, 3
    )  # (num_nodes, batch_size, num_heads, num_features)

    gumbels = (
        -torch.empty(
            key_prime.shape[:-1] + (k,), memory_format=torch.legacy_contiguous_format
        )
        .exponential_()
        .log()
    ).to(query.device) / tau
    key_t_gumbel = key_prime.unsqueeze(3) * gumbels.exp().unsqueeze(
        4
    )  # [N, B, H, K, M]
    z_numerator = numerator_gumbel(query_prime, key_t_gumbel, value)
    z_denominator = denominator_gumbel(query_prime, key_t_gumbel)

    z_numerator = z_numerator.permute(
        1, 0, 2, 3, 4
    )  # (batch_size, num_nodes, num_heads, k, num_features)
    z_denominator = z_denominator.permute(1, 0, 2, 3)
    z_denominator = torch.unsqueeze(z_denominator, len(z_denominator.shape))
    z_output = torch.mean(z_numerator / z_denominator, dim=3)  # average over k

    if return_attention_weights:
        start, end = edge_index
        query_end, key_start = query_prime[end], key_prime[start]
        edge_attention_num = torch.einsum(
            "ebhm, ebhkm->ebhk", query_end, key_start.unsqueeze(3)
        )  # (num_edges, batch_size, num_heads, k)
        edge_attention_num = edge_attention_num.permute(1, 0, 2, 3)
        attention_norm = denominator(query_prime, key_prime)
        edge_attention_dem = attention_norm[end]
        edge_attention_dem = edge_attention_dem.permute(1, 0, 2)
        att_weight = edge_attention_num / edge_attention_dem
        return z_output, att_weight
    return z_output


def add_conv_relational_bias(x, edge_index, bias, trans="sigmoid"):
    """
    Compute updated result by the relational bias of input adjacency matrix
    The implementation is similar to the GCN with a shared scalar weight for each edge
    Args:
        x (torch.Tensor): The input tensor.
        edge_index (torch.Tensor): The edge index tensor.
        bias (torch.Tensor): The bias tensor
    Returns:
        torch.Tensor: The output tensor.
    """
    row, col = edge_index
    deg_in = degree(col, x.size(1)).float()
    deg_in_norm = (1.0 / deg_in[col]).sqrt()
    deg_out = degree(row, x.size(1)).float()
    deg_out_norm = (1.0 / deg_out[row]).sqrt()
    conv_output = []
    for i in range(x.shape[2]):
        if trans == "sigmoid":
            bias_i = F.sigmoid(bias[i])
        elif trans == "identity":
            bias_i = bias[i]
        else:
            raise ValueError("Invalid transformation")
        val = torch.ones_like(row) * bias_i * deg_in_norm * deg_out_norm
        adj_i = SparseTensor(
            row=row, col=col, value=val, sparse_sizes=(x.size(1), x.size(1))
        )
        conv_output.append(matmul(adj_i, x[:, :, i]))
    return torch.stack(conv_output, dim=2)


class NodeFormerConv(nn.Module):
    """
    NodeFormer convolution layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        kernel_transformation=softmax_kernel_transformation,
        proj_mat_type="a",
        num_random_features=10,
        use_gumbel=True,
        num_gumbel_samples=10,
        rb_order=0,
        rb_transform="sigmoid",
        use_edge_loss=True,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_heads (int): Number of heads.
            kernel_transformation (function): The kernel transformation function.
            proj_mat_type (str): The type of projection matrix.
            num_random_features (int): Number of random features.
            use_gumbel (bool): If True, use Gumbel transform.
            num_gumbel_samples (int): Number of Gumbel samples.
            rb_order (int): The order of relational bias.
            rb_transform (str): The transformation of relational bias.
            use_edge_loss (bool): If True, use edge loss.
        """
        super(NodeFormerConv, self).__init__()
        self.W_key = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        self.W_query = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        self.W_value = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        self.W_output = nn.Linear(out_channels * num_heads, out_channels)
        if rb_order > 0:
            self.b = nn.Parameter(
                torch.FloatTensor(rb_order, num_heads), requires_grad=True
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel_transformation = kernel_transformation
        self.proj_mat_type = proj_mat_type
        self.num_random_features = num_random_features
        self.use_gumbel = use_gumbel
        self.num_gumbel_samples = num_gumbel_samples
        self.rb_order = rb_order
        self.rb_transform = rb_transform
        self.use_edge_loss = use_edge_loss

        self.reset_parameters()

    def reset_parameters(self):
        """
        Changes:
            1. self.W_key.reset_parameters() -> nn.init.xavier_uniform_(self.W_key.weight)
        """
        nn.init.xavier_uniform_(self.W_key.weight)
        nn.init.xavier_uniform_(self.W_query.weight)
        nn.init.xavier_uniform_(self.W_value.weight)
        nn.init.xavier_uniform_(self.W_output.weight)
        if self.rb_order > 0:
            if self.rb_transform == "sigmoid":
                nn.init.constant_(self.b, 0.1)
            elif self.rb_transform == "identity":
                nn.init.constant_(self.b, 1.0)

    def forward(self, x, adjs, tau):
        b, n = x.size(0), x.size(1)
        query = self.W_query(x).reshape(-1, n, self.num_heads, self.out_channels)
        key = self.W_key(x).reshape(-1, n, self.num_heads, self.out_channels)
        value = self.W_value(x).reshape(-1, n, self.num_heads, self.out_channels)

        if self.proj_mat_type is None:
            proj_mat = None
        else:
            dim = query.shape[-1]
            seed = torch.ceil(torch.abs(torch.sum(query) * 1e8)).to(torch.int32)
            proj_mat = create_projection_matrix(
                dim, self.num_random_features, seed=seed
            ).to(query.device)

        if self.use_gumbel and self.training:
            z_next, weight = kernelized_softmax_gumbel(
                query,
                key,
                value,
                self.kernel_transformation,
                proj_mat,
                adjs[0],
                self.num_gumbel_samples,
                tau,
                self.use_edge_loss,
            )
        else:
            z_next, weight = kernelized_softmax(
                query,
                key,
                value,
                self.kernel_transformation,
                proj_mat,
                adjs[0],
                tau,
                self.use_edge_loss,
            )

            for i in range(self.rb_order):
                z_next += add_conv_relational_bias(
                    value, adjs[i], self.b[i], self.rb_transform
                )
            z_next = self.W_output(z_next.flatten(-2, -1))

            if self.use_edge_loss:
                row, col = adjs[0]
                deg_in = degree(col, query.shape[1]).float()
                deg_norm = 1.0 / deg_in[col]
                deg_norm_ = deg_norm.reshape(1, -1, 1).repeat(1, 1, weight.shape[-1])
                link_loss = torch.mean(weight.log() * deg_norm_)
                return z_next, link_loss
            return z_next


class NodeFormer(nn.Module):
    """
    NodeFormer model.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        kernel_transformation=softmax_kernel_transformation,
        num_random_features=30,
        use_bn=True,
        use_gumbel=True,
        use_residual=True,
        use_act=False,
        use_jk=False,
        num_gumbel_samples=10,
        rb_order=0,
        rb_transform="sigmoid",
        use_edge_loss=True,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            out_channels (int): Number of output channels.
            num_layers (int): Number of layers.
            num_heads (int): Number of heads.
            dropout (float): Dropout probability.
            kernel_transformation (function): The kernel transformation function.
            num_random_features (int): Number of random features.
            use_bn (bool): If True, use batch normalization.
            use_gumbel (bool): If True, use Gumbel transform.
            use_residual (bool): If True, use residual connection.
            use_act (bool): If True, use activation function.
            use_jk (bool): If True, use jump knowledge (concatenation).
            num_gumbel_samples (int): Number of Gumbel samples.
            rb_order (int): The order of relational bias.
            rb_transform (str): The transformation of relational bias.
            use_edge_loss (bool): If True, use edge loss.
        """
        super(NodeFormer, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.batch_norms = nn.ModuleList()
        self.batch_norms.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                NodeFormerConv(
                    hidden_channels,
                    hidden_channels,
                    num_heads,
                    kernel_transformation,
                    num_random_features=num_random_features,
                    use_gumbel=use_gumbel,
                    num_gumbel_samples=num_gumbel_samples,
                    rb_order=rb_order,
                    rb_transform=rb_transform,
                    use_edge_loss=use_edge_loss,
                )
            )
            self.batch_norms.append(nn.LayerNorm(hidden_channels))
        if use_jk:
            self.fcs.append(
                nn.Linear(hidden_channels * num_layers + hidden_channels, out_channels)
            )
        else:
            self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = nn.ReLU()
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act
        self.use_jk = use_jk
        self.use_edge_loss = use_edge_loss

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for fc in self.fcs:
            nn.init.xavier_uniform_(fc.weight)  # changed
        for bn in self.batch_norms:
            bn.reset_parameters()

    def forward(self, x, adjs, tau=1.0):
        x = x.unsqueeze(0)  # [B, N, H, D], B=1 denotes number of graphs
        layer_ = []
        link_loss_ = []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.batch_norms[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            if self.use_edge_loss:
                x, link_loss = conv(x, adjs[i], tau)
                link_loss_.append(link_loss)
            else:
                x = conv(x, adjs[i], tau)
            if self.use_residual:
                x += layer_[i]
            if self.use_bn:
                x = self.batch_norms[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        if self.use_jk:
            x = torch.cat(layer_, dim=-1)

        x = self.fcs[-1](x).squeeze(0)
        if self.use_edge_loss:
            return x, link_loss_
        return x
