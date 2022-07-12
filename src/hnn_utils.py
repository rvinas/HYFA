"""
Utilities for hypergraph neural network
"""
import torch
import torch.nn as nn
import torch_scatter


# Networks
def MLP(in_dim, out_dim, h_dim=None, n_layers=1, dropout=0.1, bias=False, norm='batch', activation='relu', output_activation=None):
    """
    Multi-layer perceptron
    :param in_dim: input dimension
    :param out_dim: output dimension
    :param h_dim: hidden dimension
    :param n_layers: number of hidden layers
    :param dropout: dropout probability
    :param bias: whether to use a bias term in the last layer
    :param norm: whether to use batch/layer normalisation
    :param activation: activation function
    :param output_activation: output activation function
    :return: MLP model
    """

    if h_dim is None:
        h_dim = out_dim

    act_fn = {"swish": nn.SiLU(), "relu": nn.ReLU()}[activation]

    norm_fn = {"layer": nn.LayerNorm, "batch": nn.BatchNorm1d, "none": None}[norm]

    modules = []
    prev_dim = in_dim
    for n in range(n_layers):
        block = []
        block.append(nn.Linear(prev_dim, h_dim))
        block.append(act_fn)
        block.append(nn.Dropout(p=dropout))
        if norm_fn is not None:
            block.append(norm_fn(h_dim))
        modules.extend(block)
        prev_dim = h_dim
    modules.append(nn.Linear(prev_dim, out_dim, bias=bias))
    if output_activation is not None:
        modules.append(output_activation)
    mlp = nn.Sequential(*modules)

    return mlp


# Aggregation
def message_aggregation(messages, idxs, dim_size, aggregators=None):
    """
    Aggregates messages by node using the torch_scatter package. Thankfully,
    the package has pretty much every aggregation you'd ever want to use.
    https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html
    Aggregations are concatenated.
    messages: shape=(nb_messages, feature_dim)
    idxs: int tensor with shape=(nb_messages,)
    dim_size: number of elements in the output tensor
    aggregators: list of aggregation functions (strings)
    """
    if aggregators is None:
        aggregators = ['sum', 'max', 'min', 'mean', 'std']
    agg = []

    if 'sum' in aggregators:
        agg.append(torch_scatter.scatter_add(messages, idxs, dim=0, dim_size=dim_size))
    if 'max' in aggregators:
        agg.append(torch_scatter.scatter_max(messages, idxs, dim=0, dim_size=dim_size)[0])
    if 'min' in aggregators:
        agg.append(torch_scatter.scatter_min(messages, idxs, dim=0, dim_size=dim_size)[0])
    if 'mean' in aggregators:
        agg.append(torch_scatter.scatter_mean(messages, idxs, dim=0, dim_size=dim_size))
    if 'std' in aggregators:
        agg.append(torch_scatter.scatter_std(messages, idxs, dim=0, dim_size=dim_size))

    return torch.cat(agg, dim=-1)


def unique_ids(idxs1, idxs2):
    """
    Given two list of indexes of the same length, return another list with unique ids for each unique pair (i, j) in (idxs1, idxs2)

    Example:
    idxs1 = torch.tensor([2, 2, 0, 0, 0])
    idxs2 = torch.tensor([1, 1, 1, 0, 0])
    unique_ids(idxs1, idxs2) returns torch.tensor([2, 2, 1, 0, 0])
    """
    idx = torch.stack((idxs1, idxs2), dim=-1)
    _, inverse_index = torch.unique(idx, dim=0, return_inverse=True)
    return inverse_index


def masked_softmax(A, mask):
    """
    Computes softmax with a mask
    :param A: tensor of shape=(n, feature_dim)
    :param mask: binary tensor of shape=(n,)
    :return: tensor of shape=(n, feature_dim) after applying masked softmax
    """
    # matrix A is the one you want to do mask softmax at dim=1
    A_max = torch.max(A, dim=1, keepdim=True)[0]
    A_exp = torch.exp(A - A_max)
    A_exp = A_exp * mask.float()  # this step masks
    A_softmax = A_exp / torch.sum(A_exp, dim=1, keepdim=True)
    return A_softmax


def scatter_softmax(x, idxs, dim_size=None):
    """
    Computes softmax activation across a sparse tensor
    :param x: torch tensor of shape=(n, feature_dim)
    :param idxs: softmax indices. The softmax will be taken over elements [0 .. n) that have the same index. Shape=(n,)
    :param dim_size: (currently unused)
    :return: torch tensor of shape=(n, feature_dim)

    Example:

        import torch
        from src.hnn_utils import scatter_softmax

        src = torch.Tensor([[1, 1], [2, 1], [3, 1], [4, 1]])
        index = torch.tensor([2, 1, 1, 1])

        out = scatter_softmax(src, index)

        # tensor([[1.0000, 1.0000],
        #         [0.0900, 0.3333],
        #         [0.2447, 0.3333],
        #         [0.6652, 0.3333]])

    """
    x_max, _ = torch_scatter.scatter_max(x, idxs, dim=0, dim_size=dim_size)
    e_x = torch.exp(x - x_max[idxs])
    e_x_sum = torch_scatter.scatter_add(e_x, idxs, dim=0, dim_size=dim_size)
    return e_x / e_x_sum[idxs]


def meshgrid_2d(a, b):
    """
    Similar to torch.meshgrid, but a and b are 2d tensors of shape (n_genes, n_features)
    :param a: 2d tensors of shape (n_genes, n_features)
    :param b: 2d tensors of shape (n_genes, n_features)
    :return: tensor of shape (n_genes, n_genes, 2*n_features)
    """
    # Similar to torch.meshgrid, but a and b are 2d tensors of shape (n_genes, n_features)
    a_ = torch.tile(a[:, None, :], (1, b.shape[0], 1))
    b_ = torch.tile(b[None, :, :], (a.shape[0], 1, 1))
    return torch.cat((a_, b_), dim=-1)


# Utilities
def expand_features(features_per_node, index):
    """
    Expands the features per node back to the coordinate space. Inverse of "collapse_features"
    :param features_per_node: tensor of shape=(n_nodes, feature_dim)
    :param index: tensor of shape=(space_dim,)
    :return: tensor with features gathered according to index. Shape=(space_dim, feature_dim)
    """
    idxs = torch.tile(index[:, None], dims=[1, features_per_node.shape[-1]])
    features = torch.gather(features_per_node, dim=0, index=idxs)
    return features
