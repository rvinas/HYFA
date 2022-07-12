"""
Encoders mapping genes to metagenes
"""
import torch
import torch.nn as nn
from src.hnn_utils import *


class PlainEncoder(torch.nn.Module):
    """
    Linear mapping
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, out_dim))# , nn.ReLU()

    def forward(self, x, **kwargs):
        return self.encoder(x)


class AttentionEncoder(torch.nn.Module):
    """
    Computes attention coefficients between genes and metagenes
    (not used currently)
    """
    def __init__(self, in_dim, out_dim, metagene_params, d_gene):
        super(AttentiveEncoder, self).__init__()
        self.gene_params = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((in_dim, d_gene))))
        self.metagene_params = metagene_params
        d_metagene = metagene_params.shape[-1]

        hdim = d_gene + d_metagene
        # config.d_edge_attr is the number of heads. Each head is an edge feature for a given metagene
        self.att_mlp = nn.Sequential(nn.Linear(d_gene + d_metagene, hdim, bias=False),
                                               nn.LeakyReLU(0.1),
                                               nn.Linear(hdim, out_dim, bias=False))

    def forward(self, x, return_e=False, **kwargs):
        grid_features = meshgrid_2d(self.gene_params, self.metagene_params)  # Shape=(nb_genes, nb_metagenes, feature_dim)
        e = self.att_mlp(grid_features)  # Shape=(nb_genes, nb_metagenes, d_edge_attr)
        a = torch.nn.Softmax(dim=0)(e)  # Shape=(nb_genes, nb_metagenes, d_edge_attr)
        a = torch.reshape(a, (self.gene_params.shape[0], -1))  # Shape=(nb_genes, nb_metagenes * d_edge_attr)
        metagene_features = x @ a  # Shape=(nb_samples, nb_metagenes * d_edge_attr)

        if return_e:
            return metagene_features, e
        else:
            return metagene_features

